import torch
import torch.nn as nn
import torch.nn.functional as F


class LikelihoodModel(nn.Module):
    """Abstract base class for likelihood-based generative models.

    Subclasses must implement:
      - log_prob(x) -> Tensor of shape (batch,)
      - sample(n_samples) -> Tensor of shape (n_samples, dim)
      - _init_params_from_data(cls, X) -> dict of __init__ kwargs
    """

    def log_prob(self, x):
        raise NotImplementedError

    def sample(self, n_samples=1):
        raise NotImplementedError

    @classmethod
    def _init_params_from_data(cls, X):
        """Return kwargs to initialize the model from data."""
        raise NotImplementedError

    @classmethod
    def from_data(
        cls,
        X,
        max_iter=100,
        lr=0.5,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        verbose=True,
    ):
        X = torch.as_tensor(X, dtype=torch.float32).contiguous()

        init_kwargs = cls._init_params_from_data(X)
        model = cls(**init_kwargs)

        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=lr,
            max_iter=max_iter,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad(set_to_none=True)
            loss = -model.log_prob(X).mean()
            loss.backward()

            for p in model.parameters():
                if p.grad is not None:
                    p.grad = p.grad.contiguous()

            if verbose:
                print(f"NLL: {loss.item():.6f}")

            return loss

        optimizer.step(closure)

        return model


class GaussianModel:
    def __init__(self, mean, covariance):
        self.mean = mean
        self.covariance = covariance

    @classmethod
    def from_data(cls, X, jitter=1e-6):
        X = torch.as_tensor(X, dtype=torch.float32)

        mean = X.mean(dim=0)
        X_centered = X - mean

        covariance = X_centered.T @ X_centered / (X.shape[0] - 1)
        covariance = covariance + jitter * torch.eye(X.shape[1])

        return cls(mean, covariance)

    def distribution(self):
        return torch.distributions.MultivariateNormal(
            loc=self.mean,
            covariance_matrix=self.covariance,
        )

    def log_prob(self, x):
        return self.distribution().log_prob(x)

    def sample(self, n_samples=1):
        sample = self.distribution().sample((n_samples,))
        # sample = sample.clamp(min=-0.5)
        return sample


class ZeroInflatedPositiveGaussianModel(LikelihoodModel):
    """
    Fast zero-inflated positive Gaussian model.

    For each feature j:
        x_j = 0 with probability p_zero_j
        x_j = softplus(z_j), z_j ~ Normal(mean_j, std_j), otherwise

    This is fully vectorized (diagonal covariance).
    """

    def __init__(self, mean, log_std, logits_positive):
        super().__init__()

        self.mean = nn.Parameter(torch.as_tensor(mean, dtype=torch.float32))
        self.log_std = nn.Parameter(torch.as_tensor(log_std, dtype=torch.float32))
        self.logits_positive = nn.Parameter(
            torch.as_tensor(logits_positive, dtype=torch.float32)
        )

    @staticmethod
    def inverse_softplus(x, eps=1e-8):
        x = torch.clamp(x, min=eps)
        return x + torch.log(-torch.expm1(-x))

    @property
    def std(self):
        return torch.exp(self.log_std) + 1e-6

    @property
    def p_positive(self):
        return torch.sigmoid(self.logits_positive)

    def log_prob(self, x):
        x = torch.as_tensor(x, dtype=self.mean.dtype, device=self.mean.device)

        if x.ndim == 1:
            x = x.unsqueeze(0)

        if torch.any(x < 0):
            raise ValueError("All values must be non-negative.")

        positive = x > 0

        log_p_positive = F.logsigmoid(self.logits_positive)
        log_p_zero = F.logsigmoid(-self.logits_positive)

        discrete_lp = torch.where(
            positive,
            log_p_positive,
            log_p_zero,
        )

        z = self.inverse_softplus(torch.clamp(x, min=1e-8))

        var = self.std.square()
        normal_lp = (
            -0.5 * torch.log(2 * torch.pi * var)
            - 0.5 * ((z - self.mean) ** 2) / var
        )

        # Change-of-variables correction:
        # x = softplus(z), dx/dz = sigmoid(z)
        jacobian_lp = -F.logsigmoid(z)

        positive_lp = normal_lp + jacobian_lp

        total_lp = torch.where(
            positive,
            discrete_lp + positive_lp,
            discrete_lp,
        )

        return total_lp.sum(dim=-1)

    def sample(self, n_samples=1):
        with torch.no_grad():
            shape = (n_samples, self.mean.numel())

            positive = torch.bernoulli(
                self.p_positive.expand(shape)
            ).bool()

            z = self.mean + self.std * torch.randn(
                shape,
                dtype=self.mean.dtype,
                device=self.mean.device,
            )

            x_positive = F.softplus(z)
            x = torch.zeros_like(x_positive)
            x[positive] = x_positive[positive]

            return x

    @classmethod
    def _init_params_from_data(cls, X):
        """Heuristic initialization from data before SGD fine-tuning.

        Steps:
          1. Estimate zero-inflation probability (p_positive) per feature.
          2. Inverse-softplus the positive values to get latent z-space.
          3. Compute per-feature mean/std of z-scores (diagonal Gaussian).
          4. Parameterize log_std via inverse-softplus for unconstrained optimization.
        """
        X = torch.as_tensor(X, dtype=torch.float32)

        if X.ndim != 2:
            raise ValueError("X must have shape (n_samples, n_features).")

        if torch.any(X < 0):
            raise ValueError("All values must be non-negative.")

        positive = X > 0
        n, d = X.shape

        p_positive = positive.float().mean(dim=0)
        p_positive = p_positive.clamp(1e-5, 1 - 1e-5)
        logits_positive = torch.logit(p_positive)

        Z = cls.inverse_softplus(torch.clamp(X, min=1e-8))

        mean = torch.zeros(d)
        std = torch.ones(d)

        for j in range(d):
            zj = Z[positive[:, j], j]

            if zj.numel() >= 2:
                mean[j] = zj.mean()
                std[j] = zj.std(unbiased=True).clamp_min(1e-3)
            elif zj.numel() == 1:
                mean[j] = zj[0]
                std[j] = 1.0
            else:
                mean[j] = 0.0
                std[j] = 1.0

        # log_std = log(σ)
        log_std = torch.log(std.clamp_min(1e-6))

        return dict(mean=mean, log_std=log_std, logits_positive=logits_positive)


class TruncatedGaussianModel(LikelihoodModel):
    """
    Rectified Gaussian model (diagonal covariance, clamp at 0).

    Each feature j is drawn from a Gaussian then clamped to [0, ∞):
        x_j = max(0, z_j),  z_j ~ Normal(mean_j, std_j)

    This gives a mixed discrete-continuous distribution:
      - Continuous density for x > 0 (Gaussian pdf)
      - Point mass at x = 0 (Gaussian cdf(0))
    """

    def __init__(self, mean, log_std):
        super().__init__()

        self.mean = nn.Parameter(torch.as_tensor(mean, dtype=torch.float32))
        self.log_std = nn.Parameter(torch.as_tensor(log_std, dtype=torch.float32))

    @property
    def std(self):
        return torch.exp(self.log_std) + 1e-6

    def log_prob(self, x):
        x = torch.as_tensor(x, dtype=self.mean.dtype, device=self.mean.device)

        if x.ndim == 1:
            x = x.unsqueeze(0)

        if torch.any(x < 0):
            raise ValueError("All values must be non-negative.")

        positive = x > 0

        var = self.std.square()
        normal_lp = (
            -0.5 * torch.log(2 * torch.pi * var)
            - 0.5 * ((x - self.mean) ** 2) / var
        )

        # Point mass at zero: probability that the underlying Gaussian is ≤ 0
        std = self.std
        cdf_zero = 0.5 * (1 + torch.erf(-self.mean / (std * (2 ** 0.5))))
        log_cdf_zero = torch.log(cdf_zero.clamp_min(1e-10))

        total_lp = torch.where(
            positive,
            normal_lp,
            log_cdf_zero,
        )

        return total_lp.sum(dim=-1)

    def sample(self, n_samples=1):
        with torch.no_grad():
            shape = (n_samples, self.mean.numel())

            z = self.mean + self.std * torch.randn(
                shape,
                dtype=self.mean.dtype,
                device=self.mean.device,
            )

            return torch.clamp(z, min=0)

    @classmethod
    def _init_params_from_data(cls, X):
        """Heuristic initialization accounting for the clamp-at-zero truncation.

        For a rectified Gaussian X = max(0, Z) with Z ~ N(μ, σ²):
          P(X=0) = Φ(-μ/σ)  →  μ = -σ · Φ⁻¹(p_zero)

        Steps:
          1. Estimate σ from positive values' std (zeros carry no variance info).
          2. Derive μ from the zero proportion via the inverse probit link.
          3. Parameterize log_std = log(σ) for unconstrained optimization.
        """
        X = torch.as_tensor(X, dtype=torch.float32)

        if X.ndim != 2:
            raise ValueError("X must have shape (n_samples, n_features).")

        if torch.any(X < 0):
            raise ValueError("All values must be non-negative.")

        n, d = X.shape
        positive = X > 0

        mean = torch.zeros(d)
        std = torch.ones(d)

        for j in range(d):
            xj_pos = X[positive[:, j], j]
            p_zero = 1.0 - positive[:, j].float().mean().item()

            if xj_pos.numel() >= 2:
                # Estimate σ from positive values only
                std[j] = xj_pos.std(unbiased=True).clamp_min(1e-3)
            elif xj_pos.numel() == 1:
                std[j] = 1.0

            # Derive μ from p_zero:  μ = -σ · Φ⁻¹(p_zero)
            p_zero_clamped = min(max(p_zero, 1e-5), 1 - 1e-5)
            if 1e-5 < p_zero_clamped < 1 - 1e-5:
                # Φ⁻¹(p) = sqrt(2) * erfinv(2p - 1)
                z_score = (2 ** 0.5) * torch.erfinv(
                    torch.tensor(2 * p_zero_clamped - 1)
                )
                mean[j] = -std[j] * z_score.item()
            elif p_zero_clamped <= 1e-5:
                # Almost no zeros — use empirical mean directly
                mean[j] = X[:, j].mean()
            else:
                # Almost all zeros — push mean far negative
                mean[j] = -3.0 * std[j]

        # log_std = log(σ)
        log_std = torch.log(std.clamp_min(1e-6))

        return dict(mean=mean, log_std=log_std)

