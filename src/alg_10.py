import torch
import torch.nn as nn
from torch.nn import functional as F

from alg_2 import PositionEmbedding
from alg_1 import TokenEmbedding
from alg_5 import MHAttentionInefficient
from alg_6 import LayerNorm
from alg_7 import TokenUnembedding


class DTransformer(nn.Module):

    class ResNet(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, x, y=None, mask=None):
            if (y is not None) and (mask is None):
                return self.module(x,y) + x
            elif (y is not None) and (mask is not None):
                return self.module(x,y, mask) + x
            else:
                return self.module(x) + x


    def __init__(self, embed_dim, mlp_dim, max_seq_len, L_dec, vocab_size, num_heads):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.token_emb = TokenEmbedding(vocab_size, embed_dim)
        self.pos_emb = PositionEmbedding(max_seq_len, embed_dim)
        # setup decoder
        self.decoder_layers = nn.ModuleList()
        for i in range(L_dec):
            self.decoder_layers.add_module(f"dec_layer_norm1_{i}", LayerNorm(embed_dim))
            multi_head_attention = MHAttentionInefficient(num_heads=num_heads,
                                                          input_dim=embed_dim,
                                                          atten_dim=embed_dim,
                                                          output_dim=embed_dim)   
            self.decoder_layers.add_module(f"dec_attention_layer_{i}",DTransformer.ResNet(multi_head_attention))
            self.decoder_layers.add_module(f"dec_layer_norm2_{i}", LayerNorm(embed_dim))
            mlp = nn.Sequential(
                nn.Linear(embed_dim, mlp_dim),
                nn.GELU(),
                nn.Linear(mlp_dim, embed_dim)
            )
            self.decoder_layers.add_module(f"dec_mlp_layer_{i}", DTransformer.ResNet(mlp))

        self.layer_norm = LayerNorm(embed_dim)
        self.unembed = TokenUnembedding(vocab_size, embed_dim)


    def forward(self, x):
        x = self.token_emb(x) + self.pos_emb(len(x))
        lx = len(x)
        for name, layer in self.decoder_layers.named_children():
            if "dec_attention_layer_" in name:
                mask = torch.tril(torch.ones(lx, lx))
                x = layer(x,x, mask.masked_fill(mask==0, float('-inf'))) 
            else:
                x = layer(x)
        x = self.layer_norm(x)
        return self.unembed(x) 


if __name__ == "__main__":
    max_seq_len = 512
    embed_dim = 50
    vocab_size = 10000
    ed_seq2seq = DTransformer(embed_dim=embed_dim, mlp_dim=32, max_seq_len=max_seq_len,
                                L_dec=3, vocab_size=vocab_size, num_heads=3)

    x_ids = torch.tensor([4, 13, 1])  
    print(ed_seq2seq(x_ids))
