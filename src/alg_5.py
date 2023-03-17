import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import unittest

from alg_2 import PositionEmbedding
from alg_1 import TokenEmbedding
from alg_4 import Attention

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
    
# write a unit test for MHAttentionIneffcient 


def main():
    max_seq_len = 512
    embed_dim = 50
    vocab_size = 10000
    token_emb = TokenEmbedding(vocab_size, embed_dim)
    pos_emb = PositionEmbedding(max_seq_len, embed_dim)
    attention = MHAttentionInefficient(3, embed_dim, embed_dim, embed_dim)
    batch_size=32
    idx = torch.randint(0,vocab_size, size = (batch_size, max_seq_len)) 
    idz = torch.randint(0,vocab_size, size = (batch_size, max_seq_len//2)) 
    x = token_emb(idx) + pos_emb(max_seq_len) # current token representations
    z = token_emb(idz) + pos_emb(max_seq_len//2) # context token reps.    
    print(f"x shape: {x.shape}")
    print(f"z shape: {z.shape}")
    x_emb = attention(x,z)
    print(x_emb.shape)


class TestMHAttentionInefficient(unittest.TestCase):

    def setUp(self):
        self.mha = MHAttentionInefficient()

    def test_query_shape(self):
        query = [[1,2,3], [4,5,6]]
        self.assertEqual(self.mha.query_shape(query), (2, 3))

    def test_key_shape(self):
        key = [[1,2], [3,4], [5,6]]
        self.assertEqual(self.mha.key_shape(key), (3, 2))

    def test_value_shape(self):
        value = [[1], [2], [3]]
        self.assertEqual(self.mha.value_shape(value), (3, 1))

    def test_attention_weights(self):
        query = [[1,2], [3,4]]  # shape: (2, 2) 
        key = [[1,2], [3,4], [5,6]]  # shape: (3 , 2) 
        value = [[1], [2], [3]]  # shape: (3 , 1) 

        expected_weights = [[0.5 , 0 , 0] ,[0 , 0.5 , 0]]   # shape: (2 , 3) 

        weights = self.mha._attention_weights(query=query , key=key , value=value)

        self.assertEqual(weights[0][0] , expected_weights[0][0])   # first element of weights should be equal to first element of expected weights 
        self.assertEqual(weights[0][1] , expected_weights[0][1])   # second element of weights should be equal to second element of expected weights 
        												           # and so on...

if __name__ == "__main__":
    main()