import torch
import torch.nn as nn
from torch.nn import functional as F
import math

from alg_2 import PositionEmbedding
from alg_1 import TokenEmbedding
from alg_4 import Attention

class MHAttentionIneffcient(nn.Module):
    '''
    An inefficient, but faithful, implementation of multi-headed attention
    '''
    def __init__(self, num_heads, input_dim, atten_dim, output_dim, **kwargs):
        super().__init__()
        self.heads =nn.ModuleList()
        for _ in range(num_heads):
            self.heads.append(Attention(input_dim, atten_dim, output_dim, **kwargs))
        self.out_proj = nn.Linear(num_heads*output_dim, output_dim)
        self.num_heads = num_heads
        self.output_dim = output_dim


    def forward(self,current_tokens, context_tokens, attention_mask=None):
        ys = []
        for head in self.heads:
            ys.append(head(current_tokens, context_tokens, attention_mask))
        y = self.out_proj(torch.cat(ys).view(-1, self.output_dim*self.num_heads))

        return y

def main():
    max_seq_len = 512
    embed_dim = 50
    vocab_size = 10000
    token_emb = TokenEmbedding(vocab_size, embed_dim)
    pos_emb = PositionEmbedding(max_seq_len, embed_dim)
    attention = MHAttentionIneffcient(3, 50, 50, 50)

    idx = torch.tensor([43, 32])  
    token_embeddings = token_emb(idx)
    position_embeddings = pos_emb(2)
    x = position_embeddings + token_embeddings
    print(x.shape)
    x_emb = attention(x,x)
    print(x_emb.shape)


if __name__ == "__main__":
    main()