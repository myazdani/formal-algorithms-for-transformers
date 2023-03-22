import torch
import torch.nn as nn
from torch.nn import functional as F
import math

from src.alg_2 import PositionEmbedding
from src.alg_1 import TokenEmbedding
from src.alg_3 import SingleQueryAttention


class Attention(SingleQueryAttention):
    ## TODO : clean up attention mask specification API
    def __init__(self, input_dim, atten_dim, output_dim, **kwargs):
        super().__init__(input_dim, atten_dim, output_dim, **kwargs)
        # self.register_buffer("masked_bias", torch.tensor(-1e4))

    def forward(self, current_tokens, context_tokens, attention_mask=None):
        q = self.query(current_tokens)
        k = self.key(context_tokens)
        v = self.value(context_tokens)
        att = torch.einsum('ijk,ilk->ijl', [q,k]) / math.sqrt(self.atten_dim)
        # hugging face implementation: https://tinyurl.com/22f9b6y
        if attention_mask is not None:
            # Apply the attention mask
            # att = torch.where(mask, att, self.masked_bias)
            att = att + attention_mask[None,:,:]
        
        att = F.softmax(att, dim=-1)
        v = torch.einsum('ijk,ilm->ilk',[v, att])
        return v

def main():
    max_seq_len = 512
    embed_dim = 50
    vocab_size = 10000
    token_emb = TokenEmbedding(vocab_size, embed_dim)
    pos_emb = PositionEmbedding(max_seq_len, embed_dim)
    attention = Attention(embed_dim, embed_dim, embed_dim)
    batch_size=32
    idx = torch.randint(0,vocab_size, size = (batch_size, max_seq_len)) 
    idz = torch.randint(0,vocab_size, size = (batch_size, max_seq_len//2)) 
    x = token_emb(idx) + pos_emb(max_seq_len) # current token representations
    z = token_emb(idz) + pos_emb(max_seq_len//2) # context token reps.    
    print(f"x shape: {x.shape}")
    print(f"z shape: {z.shape}")
    mask = torch.tril(torch.ones(max_seq_len, max_seq_len//2))
    # updated representation of x folding in information from z
    x_emb = attention(x,z, mask.masked_fill(mask==0, float('-inf'))) 

    print(x_emb.shape)


if __name__ == "__main__":
    main()