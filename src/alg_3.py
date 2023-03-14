import torch
import torch.nn as nn
from torch.nn import functional as F
from alg_2 import PositionEmbedding
from alg_1 import TokenEmbedding
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

    def forward(self, current_token, context_tokens):
        q = self.query(current_token)
        k = self.key(context_tokens)
        v = self.value(context_tokens)

        att = torch.einsum('ijk,ilk->ilk', [q,k]) / np.sqrt(self.atten_dim)
        att = F.softmax(att, dim=-1)
        v = torch.einsum('ijk,ijk->ik', [att, v])
        return v[:,None,:]

def main():
    max_seq_len = 512
    embed_dim = 50
    vocab_size = 10000
    token_emb = TokenEmbedding(vocab_size, embed_dim)
    pos_emb = PositionEmbedding(max_seq_len, embed_dim)
    attention = SingleQueryAttention(embed_dim, embed_dim, embed_dim)

    batch_size=32
    current_token = torch.randint(0,vocab_size, size = (batch_size, 1)) 
    context_tokens = torch.randint(0,vocab_size, size = (batch_size, max_seq_len)) 
    current_token_embeddings = token_emb(current_token) + pos_emb(1)
    context_token_embeddings = token_emb(context_tokens) + pos_emb(max_seq_len)
    x_emb = attention(current_token_embeddings,context_token_embeddings)
    print(x_emb.shape)


if __name__ == "__main__":
    main()