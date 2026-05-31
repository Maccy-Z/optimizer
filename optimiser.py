import torch
from torch.optim.optimizer import Optimizer

# class SGD(Optimizer):
#     def __init__(self, params, lr, momentum=0):
#         if lr < 0.0:
#             raise ValueError(f"Invalid learning rate: {lr}")
#         if momentum < 0.0:
#             raise ValueError(f"Invalid momentum value: {momentum}")
#
#         defaults = dict(lr=lr, momentum=momentum)
#         super(SGD, self).__init__(params, defaults)
#
#     @torch.no_grad()
#     def step(self, closure=None):
#         loss = None
#         if closure is not None:
#             with torch.enable_grad():
#                 loss = closure()
#
#         for group in self.param_groups:
#             momentum = group['momentum']
#             lr = group['lr']
#
#             for p in group['params']:
#                 if p.grad is None:
#                     continue
#
#                 d_p = p.grad
#
#                 if momentum != 0:
#                     param_state = self.state[p]
#                     if 'momentum_buffer' not in param_state:
#                         buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
#                     else:
#                         buf = param_state['momentum_buffer']
#                         buf.mul_(momentum).add_(d_p)
#                     d_p = buf
#
#                 p.add_(d_p, alpha=-lr)
#
#         return loss


class SGD(Optimizer):
    def __init__(self, params, lr, momentum=0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum <= 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")

        defaults = dict(lr=lr, momentum=momentum)
        super(SGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad
                p.add_(d_p, alpha=-lr)

        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        pass
