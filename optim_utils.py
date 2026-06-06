import torch
import torch.nn as nn
import math

class TensorExtras:
    def __init__(self, p: torch.Tensor):
        assert p.ndim == 2

        in_shape, out_shape = p.shape
        self.in_shape = in_shape

        self.uuT = torch.eye(in_shape, device=p.device)

        self.count = 0
        self.uut_inv = torch.eye(in_shape, device=p.device)
        self.m = torch.ones(out_shape, device=p.device)

    def update_uuT(self, u: torch.Tensor, m: torch.Tensor):
        """ u.shape = [bs, in_shape]
            m.shape = [bs, out_shape]
            sum(u_i u_i.T) = u.T @ u
        """
        scale = torch.sqrt(torch.trace(u.T @ u))
        self.uuT = 0.9 * self.uuT + 0.1 * u.T @ u * 1/scale

        # m = m.mean(dim=0, dtype=torch.float32)
        self.m = 1 * m

    def get_uuT_inv(self):
        if self.count % 10 == 0:
            self.uuT_inv = torch.pinverse(self.uuT)
        self.count += 1
        return self.uuT_inv


class StateHolder:
    states: dict[torch.Tensor, TensorExtras]
    def __init__(self, muon_params: list[torch.Tensor]):
        self.states = {}
        for p in muon_params:
            self.states[p] = TensorExtras(p)


def get_muon_adam_params(model: nn.Module):
    muon_params, adam_params = [], []
    for n, p in model.named_parameters():
        # Input and output layers use adam
        if "w1" in n or "w4" in n:
            adam_params.append(p)
        # Matrix like parameters use Muon
        elif p.ndim >= 2:
            muon_params.append(p)
        else:
            adam_params.append(p)
    return muon_params, adam_params


class ManualLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, weight, bias=None, save_state: StateHolder|None=None):
        ctx.save_for_backward(u, weight, bias)

        y = u @ weight.T
        if bias is not None:
            y = y + bias

        # Log input
        ctx.save_state = save_state
        if save_state is not None:
            if weight in save_state.states:
                save_state.states[weight].update_uuT(u, y>0)
                # ctx.save_state['output'] = y
        return y

    @staticmethod
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors
        u, weight, bias = saved
        has_bias = bias is not None

        p_states = None
        if ctx.save_state is not None:
            save_state = ctx.save_state
            if weight in save_state.states:
                p_states = save_state.states[weight]

        v = grad_output
        # Log backward input gradient

        if p_states is not None:
            lam = 0.05
            m = p_states.m
            uuT_inv = p_states.get_uuT_inv()

            # grad_weight = v.T @ u @ uuT_inv
            # for _ in range(10):
            #     pred = u @ grad_weight.T
            #     resid = m * (pred - v)
            #
            #     grad = 2 * resid.T @ u / m.sum() + 2 * lam * grad_weight
            #
            #     grad_weight = grad_weight - 0.01 * grad

            u_scale = (u @ uuT_inv * u).sum(dim=1).sqrt()
            u_scale = u_scale / u_scale.mean()
            u = u * u_scale.unsqueeze(-1)

            grad_weight = v.T @ u

        else:
            grad_weight = v.T @ u

        grad_u = v @ weight

        if has_bias:
            grad_bias = v.sum(dim=0)
        else:
            grad_bias = None



        return grad_u, grad_weight, grad_bias, None


# ---------------- Muon ------------------------
@torch.compile(mode='reduce-overhead', fullgraph=True, dynamic=False)
def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2  # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    # X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    X.div_(X.norm().clamp(min=1e-7))
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A  # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT

    return X


def muon_update(grad, p_states: TensorExtras, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= 0.2 * max(grad.size(-1),
                        grad.size(-2)) ** 0.5  # Scale update to be roughly spectral norm of grad (Kimi version)

    # if p_states is not None:
    #     update = update.float() @ p_states.uuT
    return update


# ---------------- ADAM ------------------------
@torch.compile(dynamic=True)
def adam_update(grad, buf1, buf2, step, betas, eps):
    beta1, beta2 = betas

    # m_t = beta1 * m + (1 - beta1) * grad
    buf1.lerp_(grad, 1 - beta1)
    # v_t = beta2 * v + (1 - beta2) * grad^2
    buf2.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

    bias_correction1 = 1 - beta1 ** step
    bias_correction2 = 1 - beta2 ** step

    step_size = math.sqrt(bias_correction2) / bias_correction1
    return buf1.div(buf2.sqrt().add_(eps)).mul_(step_size)


class MuonWithAuxAdam(torch.optim.Optimizer):
    """
    Non-distributed variant of MuonWithAuxAdam.
    """

    def __init__(self, param_groups, tracking: StateHolder):
        self.tracking = tracking

        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon"])
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)

                    # Preconditioner
                    p_sates = None
                    if p in self.tracking.states:
                        p_sates = self.tracking.states[p]

                    update = muon_update(p.grad, p_sates, state["momentum_buffer"], beta=group["momentum"])
                    # weight decay
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    # AdamW weight decay
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss