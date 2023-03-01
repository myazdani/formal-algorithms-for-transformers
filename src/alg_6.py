import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim # embedding dimension
        self.scale = nn.Parameter(torch.FloatTensor(self.dim))
        self.offset = nn.Parameter(torch.FloatTensor(self.dim))

    def forward(self, x):
        m = x.mean()
        s = x.std()
        x_hat = ((x - m)/s)*self.scale + self.offset
        return x_hat


if __name__ == "__main__":
    dim = 50
    activations = torch.randn(dim)
    layer_norm = LayerNorm(dim)
    activations_hat = layer_norm(activations)
    print(activations_hat.shape)