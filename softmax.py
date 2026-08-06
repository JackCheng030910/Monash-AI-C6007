import torch

X = torch.tensor([
    [1.0,1.0,7.0,150.0],
    [1.0,1.0,10.0,190.0],
    [0.0,1.0,9.0,130.0]
])

W = torch.tensor([
    [0.2,0.1,0.9,0.3],
    [0.5,0.2,0.4,0.1],
    [0.1,0.6,0.8,0.2]
])

logits = X @ W.T

probabilities = torch.softmax(logits, dim=1)

print(probabilities)