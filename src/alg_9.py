import torch
import torch.nn as nn
from torch.nn import functional as F

from alg_2 import PositionEmbedding
from alg_1 import TokenEmbedding
from alg_5 import MHAttentionInefficient
from alg_6 import LayerNorm
from alg_7 import TokenUnembedding

class ETransformer(nn.Module):

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

    def __init__(self, embed_dim, mlp_dim, output_dim, max_seq_len, L_enc, vocab_size, num_heads):
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
            self.encoder_layers.add_module(f"enc_attention_layer_{i}",ETransformer.ResNet(multi_head_attention))
            layer_norm_1 = LayerNorm(embed_dim)
            self.encoder_layers.add_module(f"enc_layer_norm1_layer_{i}", layer_norm_1)
            mlp = nn.Sequential(
                nn.Linear(embed_dim, mlp_dim),
                nn.GELU(),
                nn.Linear(mlp_dim, embed_dim)
            )
            self.encoder_layers.add_module(f"enc_mlp_layer_{i}", ETransformer.ResNet(mlp))
            layer_norm_2 = LayerNorm(embed_dim)
            self.encoder_layers.add_module(f"enc_layer_norm2_layer_{i}", layer_norm_2)
        self.fc = nn.Linear(embed_dim, output_dim)
        self.gelu = nn.GELU()
        self.output_layer_norm =  LayerNorm(output_dim)
        self.unembed = TokenUnembedding(vocab_size, output_dim)

    def forward(self, x):
        lx = x.size()[1] #max seq len
        x = self.token_emb(x) + self.pos_emb(lx)[None,:,:]
        for name, layer in self.encoder_layers.named_children():
            if "attention" in name:
                x = layer(x, x)
            else:
                x = layer(x)
        x = self.gelu(self.fc(x))
        x = self.output_layer_norm(x)
        return self.unembed(x) 

if __name__ == "__main__":
    max_seq_len = 512
    embed_dim = 50
    vocab_size = 10000
    encoder_transformer = ETransformer(embed_dim=embed_dim, mlp_dim=32, 
                                       max_seq_len=max_seq_len,
                                       L_enc=3, vocab_size=vocab_size, 
                                       output_dim=16,num_heads=3)

    bs = 32
    x_ids = torch.randint(0,vocab_size, size = (bs, max_seq_len)) 
    print(encoder_transformer(x_ids))        