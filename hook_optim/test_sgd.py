import torch
import torch.nn as nn
from optimiser import SGD as MySGD
from torch.optim import SGD as PyTorchSGD
# Set up simple linear model
torch.manual_seed(0)
model1 = nn.Linear(10, 2)
model2 = nn.Linear(10, 2)
model2.load_state_dict(model1.state_dict())
# Dummy data
x = torch.randn(5, 10)
y = torch.randn(5, 2)
opt1 = MySGD(model1.parameters(), lr=0.1, momentum=0.9)
opt2 = PyTorchSGD(model2.parameters(), lr=0.1, momentum=0.9)
for _ in range(5):
    opt1.zero_grad()
    loss1 = nn.functional.mse_loss(model1(x), y)
    loss1.backward()
    opt1.step()
    opt2.zero_grad()
    loss2 = nn.functional.mse_loss(model2(x), y)
    loss2.backward()
    opt2.step()
for p1, p2 in zip(model1.parameters(), model2.parameters()):
    if not torch.allclose(p1, p2):
        print("Mismatch!")
        exit(1)
print("All matches!")
