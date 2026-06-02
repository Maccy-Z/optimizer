import torch


def energy_distance(targets, preds, beta=1.0, scaled=True, eps=1e-8):
    """
    Energy distance between empirical samples.

    targets: (n, d)
    preds:   (m, d)
    beta:    exponent, usually 1.0; should satisfy 0 < beta < 2
    """

    if not (0 < beta < 2):
        raise ValueError("beta should satisfy 0 < beta < 2")

    n = targets.shape[0]
    m = preds.shape[0]

    d_xy = torch.cdist(preds, targets, p=2).pow(beta)
    s1 = d_xy.mean()

    d_xx = torch.cdist(preds, preds, p=2).pow(beta)
    d_yy = torch.cdist(targets, targets, p=2).pow(beta)

    s2 = d_xx.sum() / (m * (m - 1))
    s3 = d_yy.sum() / (n * (n - 1))

    ed = 2 * s1 - s2 - s3

    ed = ed / (s3 + eps)

    return ed