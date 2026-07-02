import torch
from torch.autograd import Function
import torch.nn as nn


class LinearReLUFunction(Function):
    @staticmethod
    def forward(ctx, input, weight):
        """
        input:  (batch_size, in_features)
        weight: (out_features, in_features)

        Forward:  z = input @ W^T,  output = relu(z) = clamp(z, min=0)

        returns:
            output: (batch_size, out_features)
        """
        z = input @ weight.t()
        output = torch.clamp(z, min=0)

        ctx.save_for_backward(input, weight, z)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        returns gradients for:
            input, weight

        Backward:
            grad_z = grad_output * (z > 0)                       # relu mask
            grad_input  = grad_z @ W                              # (B, out) @ (out, in) -> (B, in)
            grad_weight = grad_z^T @ input                        # (out, B) @ (B, in) -> (out, in)
        """
        input, weight, z = ctx.saved_tensors

        grad_z = grad_output.clone()
        grad_z[z <= 0] = 0

        grad_input = grad_z @ weight
        grad_weight = grad_z.t() @ input

        return grad_input, grad_weight


class Model(nn.Module):
    def __init__(self, in_features, out_features, hidden_size=64, num_layers=2, device=None, generator=None):
        super().__init__()

        # Build the list of (in_features, out_features) for each layer
        if num_layers == 1:
            shapes = [(in_features, out_features)]
        else:
            shapes = [(in_features, hidden_size)]
            shapes += [(hidden_size, hidden_size)] * (num_layers - 2)
            shapes += [(hidden_size, out_features)]

        weights = []
        for fan_in, fan_out in shapes:
            w = nn.Parameter(torch.empty(fan_out, fan_in, device=device))
            nn.init.kaiming_uniform_(w, a=5 ** 0.5, generator=generator)
            weights.append(w)

        self.weights = nn.ParameterList(weights)

    def forward(self, input):
        x = input
        for i, weight in enumerate(self.weights):
            if i < len(self.weights) - 1:
                x = LinearReLUFunction.apply(x, weight)
            else:
                # Output layer: linear projection (no activation)
                x = x @ weight.t()
        return x


def main():
    device = "cuda"
    dim = 1024
    bs = 2048

    gen = torch.Generator(device=device).manual_seed(42)
    x = torch.randn(bs, dim, requires_grad=True, device=device, generator=gen)
    layer = Model(dim, dim, hidden_size=dim*4, num_layers=2, device=device, generator=gen)

    y = layer(x)
    loss = y.sum()
    loss.backward()
    print(loss)
    print(f'{x.grad.shape = }')  # torch.Size([4, 10])
    for i, w in enumerate(layer.weights):
        print(f'layer.weights[{i}].grad.shape = {w.grad.shape}')


if __name__ == '__main__':
    main()
