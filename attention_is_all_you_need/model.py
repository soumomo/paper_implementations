import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model) #num_embeddings, embedding_dim

    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model)
    #In the embedding layers, we multiply those weights by √d_model.


class PositionalEncodings(nn.Module):

    """
        Compute the frequency scaling factors in log-space:
        Instead of directly calculating 1/(10000^(2i/d_model)), use the identity a^b = exp(b * log(a)).
        This produces exactly the same values, but is more numerically stable and efficient for vectorized computation. The final
        positional encodings are still ordinary sin() and cos() values, not logarithms.
    """

    def __init__(self, d_model: int, seq_length: int , dropout: float) -> None: #dropout is for the model to be less overfit
        super().__init__()
        self.d_model = d_model
        self.seq_length = seq_length
        self.dropout = nn.Dropout(dropout)
        
        # create a matrix of shape (seq_lenth ,d_model)
        pe = torch.zeros(seq_length, d_model)
        #create a vector of shape (seq_length, 1)
        position = torch.arange(0, seq_length,dtype=float).unsqueeze(1) # unsqueeze is used for converting a 1D vector into 2D column matrix
        div_term = torch.exponential(torch.arange(0,d_model,2).float() * (-math.log(10000.0)/d_model))
        # Apply the sin to the even positions 
        pe[:,0::2] = torch.sin(position * div_term)
        # Apply the cosine to the odd positions
        pe[:,1::2] = torch.cos(position * div_term)

        #add an extra dimension for batch in the pe
        pe = pe.unsqueeze(0) #(1,seq_length , d_model)

        self.register_buffer("pe" , pe)

    def forward(self,x): 
        x = x+  (self.pe[:,:x.shape[1],:]).requires_grad(False)
        return self.dropout(x)
        
    
class LayerNorm(nn.Module):
    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) #multipled
        self.beta = nn.Parameter(torch.zeros(1)) #added

    def forward(self,x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)

        return self.alpha + (x-mean)/((std + self.eps)) + self.bias
    
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int , dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) #W1 and B1
        self.dropout = nn.Dropout(d_model)
        self.linear_2 = nn.Linear(d_ff , d_model) #W2 and B2
    
    def forward(self,x):
        #(Batch, seq_length , d_model) --> (Batch , seq_length , d_ff) --> (Batch, seq_length , d_model)
        return self.dropout(self.dropout(torch.relu(self.linear_1(x))))  






            




