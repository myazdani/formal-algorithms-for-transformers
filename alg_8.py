import torch
import torch.nn as nn
from torch.nn import functional as F

from alg_2 import PositionEmbedding
from alg_1 import TokenEmbedding
from alg_5 import MHAttentionInefficient
from alg_6 import LayerNorm


class EDTransformer(nn.Module):
    def __init__(self, embed_dim, mlp_dim, max_seq_len, L_enc, L_dec, vocab_size, num_heads):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.token_emb = TokenEmbedding(vocab_size, embed_dim)
        self.pos_emb = PositionEmbedding(max_seq_len, embed_dim)
        encoder_layers = []
        for i in range(L_enc):
            multi_head_attention = MHAttentionInefficient(num_heads=num_heads,
                                                          input_dim=embed_dim,
                                                          attention_dim=embed_dim,
                                                          output_dim=embed_dim)
            encoder_layers.append(multi_head_attention)
            layer_norm_1 = LayerNorm(embed_dim)
            encoder_layers.append(layer_norm_1)
            fc1 = nn.Linear(embed_dim, mlp_dim)
            encoder_layers.append(fc1)
            encoder_layers.append(nn.ReLU())
            fc2 = nn.Linear(mlp_dim, embed_dim)
            encoder_layers.append(fc2)
            layer_norm_2 = LayerNorm(embed_dim)
            encoder_layers.append(layer_norm_2)
    
        self.encoder = nn.Sequential(*encoder_layers)
        

