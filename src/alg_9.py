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

        def forward(self, inputs):
            return self.module(inputs) + inputs

    def __init__(self, embed_dim, mlp_dim, output_dim, max_seq_len, L_enc, vocab_size, num_heads):
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
            encoder_layers.append(ETransformer.ResNet(multi_head_attention))
            layer_norm_1 = LayerNorm(embed_dim)
            encoder_layers.append(layer_norm_1)
            mlp = nn.Sequential(
                nn.Linear(embed_dim, mlp_dim),
                nn.ReLU(),
                nn.Linear(mlp_dim, embed_dim)
            )
            encoder_layers.append(ETransformer.ResNet(mlp))
            layer_norm_2 = LayerNorm(embed_dim)
            encoder_layers.append(layer_norm_2)
        self.encoder = nn.Sequential(*encoder_layers)     
        self.fc = nn.Linear(embed_dim, output_dim)
        self.gelu = nn.GELU()
        self.output_layer_norm =  LayerNorm(output_dim)
        self.unembed = TokenUnembedding(vocab_size, output_dim)