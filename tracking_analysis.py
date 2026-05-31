import torch
from utils import plot_svd_spectrum

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

    print(f'Total data points: {u.shape[0], u.shape[1]}')
    return u, v, acts


def main():
    u, v, acts = load_data()
    st, end = 1000, 1010

    u_reg, v_reg, acts_reg = u[st:end], v[st:end], acts[st:end]
    u_reg, v_reg, acts_reg = u_reg.flatten(0, 1), v_reg.flatten(0, 1), acts_reg.flatten(0, 1)
    print(f'{u_reg.shape = }, {v_reg.shape = }')

    # Plotting
    from matplotlib import pyplot as plt
    # plt.hist(v_reg[:, 2], bins=20)
    plt.scatter(acts_reg[:, 5], v_reg[:, 5], cmap='viridis')
    plt.show()

    # Append on bias column for u
    u_reg = torch.cat([u_reg, torch.ones((u_reg.shape[0], 1))], dim=1)

    # Filter out zero values if act<0
    mask = (acts_reg>0)

    # Solve for A
    A_hat = torch.zeros(u_reg.shape[1], v_reg.shape[1])
    for j in range(v_reg.shape[1]):
        mj = mask[:, j]
        Uj = u_reg[mj]
        Vj = v_reg[mj, j]

        A_hat[:, j] = torch.linalg.lstsq(Uj, Vj).solution

    v_hat = u_reg @ A_hat * (acts_reg>0).float()
    mse = ((v_hat - v_reg) ** 2).mean()
    r2 = 1 - mse / v_reg.var()
    print(f'{mse = }, {r2 = }')


    # Plot SVD spectrum of A
    plot_svd_spectrum(A_hat, title="SVD spectrum of A")
    g = u_reg.T @ v_reg
    plot_svd_spectrum(g, title="SVD spectrum of G")
    c = u_reg.T @ u_reg
    print(f'{c.shape = }')
    plot_svd_spectrum(c, title="SVD spectrum of C")
    print(g.shape)

if __name__ == "__main__":
    main()

