import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ManualLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, weight, bias=None):
        print(ctx)
        ctx.has_bias = bias is not None

        if bias is None:
            ctx.save_for_backward(u, weight)
        else:
            ctx.save_for_backward(u, weight, bias)

        y = u @ weight.T
        if bias is not None:
            y = y + bias

        return y

    @staticmethod
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors

        if ctx.has_bias:
            u, weight, bias = saved
        else:
            u, weight = saved

        v = grad_output

        grad_u = v @ weight
        grad_weight = v.T @ u
        grad_bias = v.sum(dim=0) if ctx.has_bias else None

        return grad_u, grad_weight, grad_bias


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(10, 1))
        self.W2 = nn.Parameter(torch.randn(10, 10))
        self.W3 = nn.Parameter(torch.randn(1, 10))
        self.b1 = nn.Parameter(torch.randn(10))

    def forward(self, x):
        x = F.linear(x, self.W1, self.b1)
        x = F.gelu(x)
        x = ManualLinear.apply(x, self.W2)
        x = F.gelu(x)
        x = F.linear(x, self.W3)
        return x


Xs = torch.randn((50, 1), device=device)
Ys = 2 * Xs**2 + 1

model = Model().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0)
for t in range(1000):
    optimizer.zero_grad()
    y_pred = model(Xs)
    loss = criterion(y_pred, Ys)
    loss.backward()
    optimizer.step()
    if t % 100 == 0:
        print(t, loss.item())

# Plotting
from matplotlib import pyplot as plt
with torch.no_grad():
    Xs_test = torch.randn((100, 1), device=device)
    Ys_test = 2 * Xs_test ** 2 + 1

    y_pred = model(Xs_test)
    plt.plot(Xs_test.cpu().numpy(), Ys_test.cpu().numpy(), 'o')
    plt.plot(Xs_test.cpu().numpy(), y_pred.cpu().numpy(), 'o')
    plt.show()
