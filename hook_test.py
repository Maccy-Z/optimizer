import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class HookOptimiser:
    def __init__(self, model, lr=0.01, momentum=0.9):
        self.model = model
        self.lr = lr
        self.momentum = momentum
        self.mom_buffers = {p: torch.zeros_like(p, device=device) for p in model.parameters()}

        self.setup_hooks(model)

    def generate_hook(self, param):
        mom_buffer = self.mom_buffers[param]

        def hook(w):
            mom_buffer.mul_(0.9).add_(w.grad)
            w.add_(w.grad, alpha=-0.01)
            return

        return hook

    def setup_hooks(self, model):
        for n, p in model.named_parameters():
            p.register_post_accumulate_grad_hook(self.generate_hook(p))

    def start_step(self):
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad = None

    def end_step(self):
        pass


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
        x = F.linear(x, self.W2)
        x = F.gelu(x)
        x = F.linear(x, self.W3)
        return x


def main():
    Xs = torch.randn((50, 1), device=device)
    Ys = 2 * Xs**2 + 1

    model = Model().to(device)
    criterion = nn.MSELoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = HookOptimiser(model, lr=0.01, momentum=0.9)

    for t in range(1000):
        optimizer.start_step()
        # for n, p in model.named_parameters():
        #     if p.grad is not None:
        #         p.grad.mul_(0.9)

        y_pred = model(Xs)
        loss = criterion(y_pred, Ys)
        loss.backward()
        optimizer.end_step()

        if t % 100 == 0:
            print(t, loss.item())

        # exit(4)

    # Plotting
    from matplotlib import pyplot as plt
    with torch.no_grad():
        Xs_test = torch.randn((100, 1), device=device)
        Ys_test = 2 * Xs_test ** 2 + 1

        y_pred = model(Xs_test)
        plt.plot(Xs_test.cpu().numpy(), Ys_test.cpu().numpy(), 'o')
        plt.plot(Xs_test.cpu().numpy(), y_pred.cpu().numpy(), 'o')
        plt.show()


if __name__ == "__main__":
    main()

