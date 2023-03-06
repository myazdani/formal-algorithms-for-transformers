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
        encoder_layers = []
        for i in range(L_enc):
            multi_head_attention = MHAttentionInefficient(num_heads=num_heads,
                                                          input_dim=embed_dim,
                                                          attention_dim=embed_dim,
                                                          output_dim=embed_dim)
            encoder_layers.append(EDTransformer.ResNet(multi_head_attention))
            layer_norm_1 = LayerNorm(embed_dim)
            encoder_layers.append(layer_norm_1)
            mlp = nn.Sequential(
                nn.Linear(embed_dim, mlp_dim),
                nn.ReLU(),
                nn.Linear(mlp_dim, embed_dim)
            )
            encoder_layers.append(EDTransformer.ResNet(mlp))
            layer_norm_2 = LayerNorm(embed_dim)
            encoder_layers.append(layer_norm_2)
        self.encoder = nn.Sequential(*encoder_layers)
        # setup decoder
        decoder_layers = []
        for i in range(L_dec):
            multi_head_attention = MHAttentionInefficient(num_heads=num_heads,
                                                          input_dim=embed_dim,
                                                          attention_dim=embed_dim,
                                                          output_dim=embed_dim)   
            decoder_layers.append(EDTransformer.ResNet(multi_head_attention))
            layer_norm_3 = LayerNorm(embed_dim)
            decoder_layers.append(layer_norm_3)
            multi_head_attention = MHAttentionInefficient(num_heads=num_heads,
                                                          input_dim=embed_dim,
                                                          attention_dim=embed_dim,
                                                          output_dim=embed_dim)  
            decoder_layers.append(EDTransformer.ResNet(multi_head_attention))
            layer_norm_4 = LayerNorm(embed_dim)
            decoder_layers.append(layer_norm_4)
            mlp = nn.Sequential(
                nn.Linear(embed_dim, mlp_dim),
                nn.ReLU(),
                nn.Linear(mlp_dim, embed_dim)
            )
            decoder_layers.append(EDTransformer.ResNet(mlp))
            layer_norm_5 = LayerNorm(embed_dim)
            decoder_layers.append(layer_norm_5)
        self.decoder = nn.Sequential(*decoder_layers)

        self.unembed = TokenUnembedding(vocab_size, embed_dim)


    def forward(self, z):
        z = self.token_emb(z) + self.pos_emb(z)
        