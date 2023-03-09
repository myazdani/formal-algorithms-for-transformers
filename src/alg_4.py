import torch
import torch.nn as nn
from torch.nn import functional as F
import math

from alg_2 import PositionEmbedding
from alg_1 import TokenEmbedding
from alg_3 import SingleQueryAttention


class Attention(SingleQueryAttention):
    ## TODO : clean up attention mask specification API
    def __init__(self, input_dim, atten_dim, output_dim, **kwargs):
        super().__init__(input_dim, atten_dim, output_dim, **kwargs)
        # self.register_buffer("masked_bias", torch.tensor(-1e4))

    def forward(self, current_tokens, context_tokens, attention_mask=None):
        q = self.query(current_tokens).T
        k = self.key(context_tokens).T
        v = self.value(context_tokens).T

        att = q.T @ k / math.sqrt(self.atten_dim)
        # hugging face implementation: https://tinyurl.com/22f9b6y
        if attention_mask is not None:
            # Apply the attention mask
            # att = torch.where(mask, att, self.masked_bias)
            att = att + attention_mask
        
        att = F.softmax(att, dim=-1)
        v = v @ att
        return v.T

def main():
    max_seq_len = 512
    embed_dim = 50
    vocab_size = 10000
    token_emb = TokenEmbedding(vocab_size, embed_dim)
    pos_emb = PositionEmbedding(max_seq_len, embed_dim)
    attention = Attention(50, 50, 50)

    idx = torch.tensor([43, 32])  
    token_embeddings = token_emb(idx)
    position_embeddings = pos_emb(2)
    x = position_embeddings + token_embeddings
    print(x.shape)
    mask = torch.tril(torch.ones(len(idx), len(idx)))
    x_emb = attention(x,x, mask.masked_fill(mask==0, float('-inf')))
    print(x_emb.shape)
    print(x_emb)


if __name__ == "__main__":
    main()