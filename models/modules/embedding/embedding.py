import torch.nn as nn
from .token import TokenEmbedding
 
class Embedding(nn.Module):
    """
    TokenEmbedding : normal embedding matrix
    """ 

    def __init__(self, vocab_size, embed_size, dropout=0.1):
        """
        :param vocab_size: total vocab size ( num_items + 2 )  
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = TokenEmbedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size
        
    def forward(self, sequence):
        x = self.token(sequence)
        return self.dropout(x) 