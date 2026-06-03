import torch
import torch.nn as nn
import torch.optim as optim


class AUSigmoidBU(nn.Module):
    """
    Fits: v = A @ (u * sigmoid(B @ u))

    u: shape (n_samples, in_dim)
    v: shape (n_samples, out_dim)
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.A = nn.Parameter(torch.randn(out_dim, in_dim) * 0.01)
        self.B = nn.Parameter(torch.randn(in_dim, in_dim) * 0.01)

    def forward(self, u, m, a):
        return ((torch.sigmoid(u @ self.B) * u) @ self.A.T) * m

    def fit(
        self,
        u, v, m, a,
        lr=1e-3, steps=1000, weight_decay=0.0, verbose=False,
    ):
        optimizer = optim.Adam(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        loss_fn = nn.MSELoss()

        for step in range(steps):
            optimizer.zero_grad()

            v_hat = self(u, m, a)
            loss = loss_fn(v_hat, v)

            loss.backward()
            optimizer.step()

            if verbose and step % 500 == 0:
                print(f"step {step:5d} | loss {loss.item():.6g}")

        return self
