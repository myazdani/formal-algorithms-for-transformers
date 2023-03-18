import torch
import torch.nn as nn
from torch.nn import functional as F

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size # number of tokens
        self.embed_dim = embed_dim # embedding dimension
        self.emb = nn.Embedding(self.vocab_size, self.embed_dim, **kwargs)

    def forward(self, idx):
        return self.emb(idx)



if __name__ == "__main__":
    vocab_size = 10000
    embed_dim = 50
    token_emb = TokenEmbedding(vocab_size, embed_dim)
    vocab_size=100
    batch_size = 32
    seq_len=128
    idx = torch.randint(0,vocab_size, size = (batch_size, seq_len))
    idx_embeding = token_emb(idx)
    print(idx_embeding.shape)
