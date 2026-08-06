import torch

X = torch.tensor([
    [1.0, 1.0, 7.0, 150.0],
    [1.0, 1.0, 10.0, 190.0],
    [0.0, 1.0, 9.0, 130.0]
])

W1 = torch.tensor([
    [0.2, 0.1, 0.9, 0.3],
    [0.5, 0.2, 0.4, 0.1],
    [0.1, 0.8, 0.6, 0.2]
])

H = X @ W1.T
print("H:\n", H)

W2 = torch.tensor([
    [0.2, 0.4, 0.3],
    [0.5, 0.1, 0.2]
])

Y = H @ W2.T
print("Y:\n", Y)

W_eff = W2 @ W1
print("W_eff:\n", W_eff)

Y_eff = X @ W_eff.T
print("Y_eff:\n", Y_eff)

prob = torch.softmax(Y, dim=1)
print("Softmax(Y):\n", prob)