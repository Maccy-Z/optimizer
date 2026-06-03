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

        if keep.sum() <= 1:
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
    train_st1, train_end1 = 700, 712
    val_st, val_end = 712, 714
    train_st2, train_end2 = 714, 730

    u_train = torch.cat([u[train_st1:train_end1], u[train_st2:train_end2]], dim=0)
    v_train = torch.cat([v[train_st1:train_end1], v[train_st2:train_end2]], dim=0)
    acts_train = torch.cat([acts[train_st1:train_end1], acts[train_st2:train_end2]], dim=0)

    u_train, v_train, acts_train = u_train.flatten(0, 1), v_train.flatten(0, 1), acts_train.flatten(0, 1)

    u_val, v_val, acts_val = u[val_st:val_end], v[val_st:val_end], acts[val_st:val_end]
    u_val, v_val, acts_val = u_val.flatten(0, 1), v_val.flatten(0, 1), acts_val.flatten(0, 1)

    print(f'{u_train.shape = }, {v_train.shape = }')
    print(f'{u_val.shape = }, {v_val.shape = }')

    # Append on bias column for u
    u_train = torch.cat([u_train, torch.ones((u_train.shape[0], 1))], dim=1)
    u_val = torch.cat([u_val, torch.ones((u_val.shape[0], 1))], dim=1)

    # Filter out zero values if act<0
    mask_train = (acts_train>0).float()
    mask_val = (acts_val>0).float()
    # mask_train, mask_val = torch.ones_like(mask_train), torch.ones_like(mask_val)

    A_hat = regress(u_train, v_train, mask_train)

    print(f'{A_hat.shape = }')

    v_hat_train = u_train @ A_hat * mask_train
    resid_train = v_train - v_hat_train
    mse_train = (resid_train ** 2).mean()
    r2_train = 1 - mse_train / v_train.var()
    print(f'Train Simple: {mse_train = }, {r2_train = }')

    v_hat_val = u_val @ A_hat * mask_val
    resid_val = v_val - v_hat_val
    mse_val = (resid_val ** 2).mean()
    r2_val = 1 - mse_val / v_val.var()
    print(f'Val Simple: {mse_val = }, {r2_val = }')

    plt.scatter(v_hat_train[:, 0], resid_train[:, 0], alpha=0.1, label='Train Residual')
    plt.scatter(v_hat_val[:, 0], resid_val[:, 0], alpha=0.1, label='Val Residual')
    plt.legend()
    plt.show()

    # # Try fitting more complex model
    # u_train2 = torch.cat([u_train,], dim=1)
    # u_val2 = torch.cat([u_val,], dim=1)
    # # print(f'{u_train2.shape = }')
    # model2 = AUSigmoidBU(u_train2.shape[1], v_train.shape[1])
    # model2.fit(u_train2, v_train, mask_train, acts_train)
    #
    # v_hat2_train = model2.forward(u_train2, mask_train, acts_train).detach()
    # resid2_train = v_train - v_hat2_train
    # mse2_train = (resid2_train ** 2).mean()
    # r22_train = 1 - mse2_train / v_train.var()
    # print(f'Train Complex: mse = {mse2_train}, r2 = {r22_train}')
    #
    # v_hat2_val = model2.forward(u_val2, mask_val, acts_val).detach()
    # resid2_val = v_val - v_hat2_val
    # mse2_val = (resid2_val ** 2).mean()
    # r22_val = 1 - mse2_val / v_val.var()
    # print(f'Val Complex: mse = {mse2_val}, r2 = {r22_val}')
    #
    # plt.scatter(v_hat2_train[:, 0], resid2_train[:, 0], alpha=0.1, label='Train Residual')
    # plt.scatter(v_hat2_val[:, 0], resid2_val[:, 0], alpha=0.1, label='Val Residual')
    # plt.legend()
    # plt.show()

    # A_hat2 = model2.A.detach()
    # # Plot SVD spectrum of A
    # plot_svd_spectrum(A_hat, title="SVD spectrum of A")
    # plot_svd_spectrum(A_hat2, title="SVD spectrum of A2")


if __name__ == "__main__":
    main()
