import torch
import torch.nn as nn
from torch.nn import functional as F

class TokenUnembedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size # number of tokens
        self.embed_dim = embed_dim # embedding dimension
        self.un_emb = nn.Linear(self.embed_dim, self.vocab_size, **kwargs)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, embedding):
        logits = self.un_emb(embedding)
        return self.softmax(logits)

    


if __name__ == "__main__":
    vocab_size = 10000
    embed_dim = 50
    token_unemb = TokenUnembedding(vocab_size, embed_dim)
    emb = torch.randn(32, embed_dim)
    unemb = token_unemb(emb)
    print(unemb.shape)
