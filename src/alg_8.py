import torch
import torch.nn as nn
from torch.nn import functional as F

from src.alg_2 import PositionEmbedding
from src.alg_1 import TokenEmbedding
from src.alg_5 import MHAttentionInefficient
from src.alg_6 import LayerNorm
from src.alg_7 import TokenUnembedding


class EDTransformer(nn.Module):

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


    def __init__(self, embed_dim, mlp_dim, max_seq_len, L_enc, L_dec, vocab_size, num_heads):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.token_emb = TokenEmbedding(vocab_size, embed_dim)
        self.pos_emb = PositionEmbedding(max_seq_len, embed_dim)
        # setup encoder
        self.encoder_layers = nn.ModuleList()
        for i in range(L_enc):
            multi_head_attention = MHAttentionInefficient(num_heads=num_heads,
                                                          input_dim=embed_dim,
                                                          atten_dim=embed_dim,
                                                          output_dim=embed_dim)
            self.encoder_layers.add_module(f"enc_attention_layer_{i}",EDTransformer.ResNet(multi_head_attention))
            layer_norm_1 = LayerNorm(embed_dim)
            self.encoder_layers.add_module(f"enc_layer_norm1_layer_{i}", layer_norm_1)
            mlp = nn.Sequential(
                nn.Linear(embed_dim, mlp_dim),
                nn.ReLU(),
                nn.Linear(mlp_dim, embed_dim)
            )
            self.encoder_layers.add_module(f"enc_mlp_layer_{i}", EDTransformer.ResNet(mlp))
            layer_norm_2 = LayerNorm(embed_dim)
            self.encoder_layers.add_module(f"enc_layer_norm2_layer_{i}", layer_norm_2)
        # setup decoder
        self.decoder_layers = nn.ModuleList()
        for i in range(L_dec):
            multi_head_attention = MHAttentionInefficient(num_heads=num_heads,
                                                          input_dim=embed_dim,
                                                          atten_dim=embed_dim,
                                                          output_dim=embed_dim)   
            self.decoder_layers.add_module(f"dec_attention1_layer_{i}",EDTransformer.ResNet(multi_head_attention))
            layer_norm_3 = LayerNorm(embed_dim)
            self.decoder_layers.add_module(f"dec_layer_norm1_{i}", layer_norm_3)
            multi_head_attention = MHAttentionInefficient(num_heads=num_heads,
                                                          input_dim=embed_dim,
                                                          atten_dim=embed_dim,
                                                          output_dim=embed_dim)  
            self.decoder_layers.add_module(f"dec_attention2_layer_{i}", EDTransformer.ResNet(multi_head_attention))
            layer_norm_4 = LayerNorm(embed_dim)
            self.decoder_layers.add_module(f"dec_layer_norm2_layer_{i}", layer_norm_4)
            mlp = nn.Sequential(
                nn.Linear(embed_dim, mlp_dim),
                nn.ReLU(),
                nn.Linear(mlp_dim, embed_dim)
            )
            self.decoder_layers.add_module(f"dec_mlp_layer_{i}", EDTransformer.ResNet(mlp))
            layer_norm_5 = LayerNorm(embed_dim)
            self.decoder_layers.add_module(f"dec_layer_norm3_layer_{i}", layer_norm_5)

        self.unembed = TokenUnembedding(vocab_size, embed_dim)


    def forward(self, z, x=None):
        lz = z.size()[1] #max seq len
        z = self.token_emb(z) + self.pos_emb(lz)[None,:,:]
        for name, layer in self.encoder_layers.named_children():
            if "attention" in name:
                z = layer(z, z)
            else:
                z = layer(z)
        lx = x.size()[1] #max seq len
        x = self.token_emb(x) + self.pos_emb(lx)[None,:,:]
        for name, layer in self.decoder_layers.named_children():
            if "dec_attention1" in name:
                mask = torch.tril(torch.ones(lx, lx))
                x = layer(x,x, mask.masked_fill(mask==0, float('-inf'))) 
            elif "dec_attention2" in name:
                x = layer(x,z)
            else:
                x = layer(x)
        return self.unembed(x) 


if __name__ == "__main__":
    max_seq_len = 512
    embed_dim = 50
    vocab_size = 10000
    ed_seq2seq = EDTransformer(embed_dim=embed_dim, mlp_dim=32, max_seq_len=max_seq_len,
                                L_dec=3, L_enc=3, vocab_size=vocab_size, num_heads=3)

    bs = 32
    z_ids = torch.randint(0,vocab_size, size = (bs*2, max_seq_len)) 
    x_ids = torch.randint(0,vocab_size, size = (bs*2, 1))
    output = ed_seq2seq(z_ids, x_ids)
    print(output.size())
    probs = output.gather(dim=2, index=x_ids.unsqueeze(-1)).squeeze(-1)
    print(probs.size())