import torch
from model import InputEmbeddings

d_model = 8
vocab_size = 100
embeddings = InputEmbeddings(d_model, vocab_size)

x = torch.tensor([[1, 5, 9], [2, 4, 7]], dtype=torch.long)
y = embeddings(x)

print("input shape:", x.shape)
print("output shape:", y.shape)
print(y)