"""
\begin{algorithm}[h] 
    \DontPrintSemicolon
    \KwIn{$v∈V\cong[N_\t{V}]$, a token ID.}
    \KwOut{$\v{e}∈ℝ^{d_\t{e}}$, the vector representation of the token.}
    \KwParam{$\m{W_e}∈ℝ^{d_\t{e}×N_\t{V}}$, the token embedding matrix.}
    \Return $\v{e} = \m{W_e}[:,v]$
    \caption{Token embedding.}
    \label{algo:token_embedding}
\end{algorithm}
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size # number of tokens
        self.embed_dim = embed_dim # embedding dimension
        self.emb = nn.Embedding(self.vocab_size, self.embed_dim, **kwargs)

    def forward(self, idx):
        return self.emb(idx)



if __name__ == "__main__":
    vocab_size = 10000
    embed_dim = 50
    token_emb = TokenEmbedding(vocab_size, embed_dim)
    idx = torch.tensor([1, 2])
    idx_embeding = token_emb(idx)
    print(idx_embeding.shape)
