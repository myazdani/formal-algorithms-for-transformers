import torch
import torch.nn as nn
from torch.nn import functional as F

class PositionEmbedding(nn.Module):
    def __init__(self, lmax, embed_dim, **kwargs):
        super().__init__()
        self.lmax = lmax # position of a token in the sequence
        self.embed_dim = embed_dim # embedding dimension
        self.emb = nn.Parameter(torch.zeros(self.lmax, self.embed_dim))

    def forward(self, t):
        return self.emb[:t,:]

    


if __name__ == "__main__":
    max_seq_len = 512
    embed_dim = 50
    pos_emb = PositionEmbedding(max_seq_len, embed_dim)
    position_embeddings = pos_emb(123)
    print(position_embeddings.shape)
