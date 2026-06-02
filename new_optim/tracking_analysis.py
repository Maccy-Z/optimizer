import torch
from matplotlib import pyplot as plt
import scipy.stats as stats
import numpy as np
import torch.nn.functional as F

from utils import plot_svd_spectrum
from models import GaussianModel, ZeroInflatedPositiveGaussianModel, TruncatedGaussianModel
from distances import energy_distance
from fit_model import AUSigmoidBU


def load_data():
    tracking = torch.load("tracking.pt", map_location=torch.device('cpu'))

    inputs, acts, grads = [], [], []

    for save in tracking:
        inputs.append(save["input"])
        acts.append(save["output"])
        grads.append(save["input_grad"])

    u = torch.stack(inputs).float()
    acts = torch.stack(acts).float()
    v = torch.stack(grads).float()

    v = v * 1e4
    print(f'Total data points: {u.shape[0], u.shape[1]}')
    return u, v, acts


def regress(u, v, m, tol=1e-8, eps=1e-8):
    """Regress v = (A u + eps) * m, where m can be float.

    Uses rows where m[:, j] is nonzero, solves:
        v[:, j] / m[:, j] = u @ A[:, j] + eps

    Scales u and Vj after masking, separately for each output j.
    No mean subtraction.
    """
    n, d_u = u.shape
    d_v = v.shape[1]

    A_hat = torch.zeros(d_u, d_v, dtype=u.dtype, device=u.device)

    for j in range(d_v):
        mj = m[:, j]
        keep = mj.abs() > tol

        if keep.sum() == 0:
            continue

        Uj = u[keep]
        Vj = v[keep, j] / mj[keep]

        # Scale after masking, specific to this output j
        v_scale = Vj.std().clamp_min(eps)

        Vj_scaled = Vj / v_scale

        beta_scaled = torch.linalg.lstsq(
            Uj,
            Vj_scaled,
            driver="gelsd",
        ).solution

        # Convert coefficients back to original scale
        A_hat[:, j] = beta_scaled * v_scale

    return A_hat


def fit_model_u(u_reg):
    # Fit model on u
    print()
    model_zipg = TruncatedGaussianModel.from_data(u_reg)
    model_gaussian = GaussianModel.from_data(u_reg)


    n_samples = 10000
    idxs = slice(0, 1000)
    sample_zipg = model_zipg.sample(n_samples)
    energy_zipg = energy_distance(targets=u_reg[:, idxs], preds=sample_zipg[:, idxs])
    sample_gaussian = model_gaussian.sample(n_samples)
    energy_gaussian = energy_distance(targets=u_reg[:, idxs], preds=sample_gaussian[:, idxs])
    print(f'{energy_zipg = }, ')

    print(f'{energy_gaussian = }, ')

    plt.scatter(sample_gaussian[:, 18], sample_gaussian[:, 19], label='Sample')
    plt.scatter(u_reg[:, 18], u_reg[:, 19], label="True")
    plt.legend()
    plt.title("sample_gaussian")
    plt.show()


    plt.scatter(sample_zipg[:, 18], sample_zipg[:, 19], label='Sample')
    plt.scatter(u_reg[:, 18], u_reg[:, 19], label="True")
    plt.legend()
    plt.title("zero inflated positive gaussian")
    plt.show()

    print(u_reg[:, 19].mean())
    print(sample_gaussian[:, 19].mean())
    print(sample_zipg[:, 19].mean())


def main():
    u, v, acts = load_data()
    st, end = 1000, 1050

    u_reg, v_reg, acts_reg = u[st:end], v[st:end], acts[st:end]
    u_reg, v_reg, acts_reg = u_reg.flatten(0, 1), v_reg.flatten(0, 1), acts_reg.flatten(0, 1)
    # v_reg = v_reg[:, 2:3]
    # acts_reg = acts_reg[:, 2:3]

    print(f'{u_reg.shape = }, {v_reg.shape = }')

    # # Plotting
    # from matplotlib import pyplot as plt
    # # plt.hist(v_reg[:, 2], bins=20)
    # plt.scatter(u_reg[:, 5], v_reg[:, 5], cmap='viridis')
    # plt.show()


    # Append on bias column for u
    u_reg = torch.cat([u_reg, torch.ones((u_reg.shape[0], 1))], dim=1)

    # Filter out zero values if act<0
    mask = (acts_reg>0).float()
    # mask = torch.ones_like(mask)

    A_hat = regress(u_reg, v_reg, mask)

    print(f'{A_hat.shape = }')

    v_hat = u_reg @ A_hat * mask
    resid = v_reg - v_hat
    mse = (resid ** 2).mean()
    r2 = 1 - mse / v_reg.var()
    print(f'{mse = }, {r2 = }')


    plt.scatter(v_hat[:, 3], resid[:, 3], alpha=0.1, label='Residual')
    plt.show()

    # Try fitting more complex model
    u_reg2 = torch.cat([u_reg,], dim=1)
    # print(f'{u_reg2.shape = }')
    model2 = AUSigmoidBU(u_reg2.shape[1], v_reg.shape[1])
    model2.fit(u_reg2, v_reg, mask, acts_reg)
    v_hat2 = model2.forward(u_reg2, mask, acts_reg).detach()
    resid2 = v_reg - v_hat2
    mse2 = (resid2 ** 2).mean()
    r22 = 1 - mse2 / v_reg.var()
    print(f'{mse2 = }, {r22 = }')

    plt.scatter(v_hat2[:, 3], resid2[:, 3], alpha=0.1, label='Residual')
    plt.show()

    A_hat2 = model2.A.detach()
    # Plot SVD spectrum of A
    plot_svd_spectrum(A_hat, title="SVD spectrum of A")
    plot_svd_spectrum(A_hat2, title="SVD spectrum of A2")

    # g = u_reg.T @ v_reg
    # plot_svd_spectrum(g, title="SVD spectrum of G")
    # c = u_reg.T @ u_reg
    # print(f'{c.shape = }')
    # plot_svd_spectrum(c, title="SVD spectrum of C")
    # print(g.shape)


if __name__ == "__main__":
    main()

