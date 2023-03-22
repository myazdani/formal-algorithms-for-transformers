import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import unittest

from src.alg_2 import PositionEmbedding
from src.alg_1 import TokenEmbedding
from src.alg_4 import Attention

class MHAttentionInefficient(nn.Module):
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
        y = self.out_proj(torch.cat(ys,axis=2))

        return y