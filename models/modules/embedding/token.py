import torch.nn as nn

class TokenEmbedding(nn.Module):
    
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        vocab_size = vocab_size
        self.emb = nn.Embedding(vocab_size, embed_size, padding_idx=0)

    def forward(self, x): # B x T
        return self.emb(x)  # B x T x H
