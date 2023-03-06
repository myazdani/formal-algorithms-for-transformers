import torch
import torch.nn as nn
from torch.nn import functional as F

from alg_2 import PositionEmbedding
from alg_1 import TokenEmbedding
from alg_5 import MHAttentionInefficient
from alg_6 import LayerNorm
from alg_7 import TokenUnembedding


class EDTransformer(nn.Module):

    class ResNet(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inputs):
            return self.module(inputs) + inputs


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


    def forward(self, z):
        z = self.token_emb(z) #+ self.pos_emb(z)
        for name, layer in self.encoder_layers.named_children():
            print(layer)
            import pdb
            pdb.set_trace()
            break


if __name__ == "__main__":
    max_seq_len = 512
    embed_dim = 50
    vocab_size = 10000
    ed_seq2seq = EDTransformer(embed_dim=embed_dim, mlp_dim=32, max_seq_len=max_seq_len,
                                L_dec=3, L_enc=3, vocab_size=vocab_size, num_heads=3)

    idx = torch.tensor([43, 32])  
