import unittest
import torch
from src.alg_2 import PositionEmbedding
from src.alg_1 import TokenEmbedding
from src.alg_5 import MHAttentionInefficient


class TestMHAttentionInefficient(unittest.TestCase):
    def setUp(self):
        # initialize objects that will be used in the tests
        self.max_seq_len = 512
        self.embed_dim = 50
        self.vocab_size = 10000
        self.token_emb = TokenEmbedding(vocab_size=self.vocab_size, embed_dim=self.embed_dim)
        self.pos_emb = PositionEmbedding(self.max_seq_len, self.embed_dim)
        self.attention = MHAttentionInefficient(num_heads=3, input_dim=self.embed_dim, 
                                                atten_dim=self.embed_dim, output_dim=self.embed_dim)

    def test_shape(self):
        # test that the output shape is as expected
        batch_size=32
        idx = torch.randint(0,self.vocab_size, size = (batch_size, self.max_seq_len)) 
        idz = torch.randint(0,self.vocab_size, size = (batch_size, self.max_seq_len//2)) 
        x = self.token_emb(idx) + self.pos_emb(self.max_seq_len) # current token representations
        z = self.token_emb(idz) + self.pos_emb(self.max_seq_len//2) # context token reps.    
        y = self.attention(x,z)
        self.assertEqual(y.shape, (batch_size, self.max_seq_len, self.embed_dim))


if __name__ == "__main__":
    unittest.main()