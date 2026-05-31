import torch


class ManualLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, weight, bias=None, save_state: dict|None=None):
        ctx.save_state = save_state
        ctx.save_for_backward(u, weight, bias)

        y = u @ weight.T
        if bias is not None:
            y = y + bias

        # Log input
        ctx.save_state["input"] = u.to(torch.bfloat16)
        ctx.save_state['output'] = y.to(torch.bfloat16)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors

        u, weight, bias = saved
        has_bias = bias is not None

        v = grad_output

        grad_u = v @ weight
        grad_weight = v.T @ u

        if has_bias:
            grad_bias = v.sum(dim=0)
        else:
            grad_bias = None

        # Log backward input gradient
        ctx.save_state['input_grad'] = v.to(torch.bfloat16)

        return grad_u, grad_weight, grad_bias, None

