import torch
import torch.nn as nn
from torch.nn import functional as F
from position_embedding import PositionEmbedding
from token_embedding import TokenEmbedding
import numpy as np


class SingleQueryAttention(nn.Module):
    def __init__(self, input_dim, atten_dim, output_dim, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.atten_dim = atten_dim
        self.output_dim = output_dim
        self.key = nn.Linear(input_dim, atten_dim)
        self.query = nn.Linear(input_dim, atten_dim)
        self.value = nn.Linear(input_dim, output_dim)

    def forward(self, current_tokens, context_tokens):
        q = self.query(current_tokens).T
        k = self.key(context_tokens).T
        v = self.value(context_tokens).T

        att = q.T @ k / np.sqrt(self.atten_dim)
        att = F.softmax(att, dim=-1)
        v = v @ att
        return v.T

def main():
    max_seq_len = 512
    embed_dim = 50
    vocab_size = 10000
    token_emb = TokenEmbedding(vocab_size, embed_dim)
    pos_emb = PositionEmbedding(max_seq_len, embed_dim)
    attention = SingleQueryAttention(50, 50, 50)

    idx = torch.tensor([43, 32])  
    token_embeddings = token_emb(idx)
    position_embeddings = pos_emb(2)
    x = position_embeddings + token_embeddings
    print(x.shape)
    x_emb = attention(x,x)
    print(x_emb.shape)


if __name__ == "__main__":
    main()