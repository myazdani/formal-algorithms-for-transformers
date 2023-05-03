# Formal Algorithms For Transformers

PyTorch implementation of transformer algorithms described in "Formal Algorithms for Transformers" by Mary Phuong and Marcus Hutter:  https://arxiv.org/abs/2207.09238

[Algorithm 1](https://github.com/myazdani/formal-algorithms-for-transformers/blob/main/src/alg_1.py): token embedding

[Algorithm 2](https://github.com/myazdani/formal-algorithms-for-transformers/blob/main/src/alg_2.py): positional embedding

[Algorithm 3](https://github.com/myazdani/formal-algorithms-for-transformers/blob/main/src/alg_3.py): Basic single-query attention

[Algorithm 4](https://github.com/myazdani/formal-algorithms-for-transformers/blob/main/src/alg_4.py): 𝑽˜ ← Attention(𝑿, 𝒁|W𝒒𝒌𝒗, Mask)

[Algorithm 5](https://github.com/myazdani/formal-algorithms-for-transformers/blob/main/src/alg_5.py): 𝑽˜ ← MHAttention(𝑿, 𝒁|W, Mask)

[Algorithm 6](https://github.com/myazdani/formal-algorithms-for-transformers/blob/main/src/alg_6.py): ˆ𝒆 ← layer_norm(𝒆|𝜸, 𝜷)

[Algorithm 7](https://github.com/myazdani/formal-algorithms-for-transformers/blob/main/src/alg_7.py): Unembedding.

[Algorithm 8](https://github.com/myazdani/formal-algorithms-for-transformers/blob/main/src/alg_8.py): 𝑷 ← EDTransformer(𝒛, 𝒙|𝜽)

[Algorithm 9](https://github.com/myazdani/formal-algorithms-for-transformers/blob/main/src/alg_9.py): 𝑷 ← ETransformer(𝒙|𝜽)

[Algorithm 10](https://github.com/myazdani/formal-algorithms-for-transformers/blob/main/src/alg_10.py): 𝑷 ← DTransformer(𝒙|𝜽)

Algorithm 11: 𝜽ˆ ← EDTraining(𝒛1:𝑁data , 𝒙1:𝑁data , 𝜽)

Algorithm 12: 𝜽ˆ ← ETraining(𝒙1:𝑁data , 𝜽)

Algorithm 13: 𝜽ˆ ← DTraining(𝒙1:𝑁data , 𝜽)

Algorithm 14: 𝒚 ← DInference(𝒙, 𝜽ˆ)
