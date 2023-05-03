# Formal Algorithms For Transformers

PyTorch implementation of transformer algorithms described in "Formal Algorithms for Transformers" by Mary Phuong and Marcus Hutter:  https://arxiv.org/abs/2207.09238

[Algorithm 1](https://github.com/myazdani/formal-algorithms-for-transformers/blob/main/src/alg_1.py): token embedding
Algorithm 2: positional embedding
Algorithm 3: Basic single-query attention
Algorithm 4: 𝑽˜ ← Attention(𝑿, 𝒁|W𝒒𝒌𝒗, Mask)
Algorithm 5: 𝑽˜ ← MHAttention(𝑿, 𝒁|W, Mask)
Algorithm 6: ˆ𝒆 ← layer_norm(𝒆|𝜸, 𝜷)
Algorithm 7: Unembedding.
Algorithm 8: 𝑷 ← EDTransformer(𝒛, 𝒙|𝜽)
Algorithm 9: 𝑷 ← ETransformer(𝒙|�
Algorithm 10: 𝑷 ← DTransformer(𝒙|𝜽)
Algorithm 11: 𝜽ˆ ← EDTraining(𝒛1:𝑁data , 𝒙1:𝑁data , 𝜽)
Algorithm 12: 𝜽ˆ ← ETraining(𝒙1:𝑁data , 𝜽)
Algorithm 13: 𝜽ˆ ← DTraining(𝒙1:𝑁data , 𝜽)
Algorithm 14: 𝒚 ← DInference(𝒙, 𝜽ˆ)
