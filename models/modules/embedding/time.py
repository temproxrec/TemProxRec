import torch.nn as nn
        
class AbsoluteTimeEmbedding(nn.Module):
    # For absolute time embedding
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        vocab_size = vocab_size
        self.emb = nn.Embedding(vocab_size, embed_size, padding_idx=0)

    def forward(self, x): # B x T
        return self.emb(x)  # B x T x H
    

class RelativeTimeEmbedding(nn.Module):
    # For relataive time embedding
    def __init__(self, cilp_time, hidden_units, device):
        super().__init__()
        time_difference_clip = cilp_time 
        hidden = hidden_units 
        self.time_emb = nn.Embedding(time_difference_clip + 1, hidden)
        self.time_difference_clip = time_difference_clip

    def forward(self, time, max_len):
        # t : B x T
        # time_diff : B x T x T  (value range: -time_range ~ time_range)
        query_time, key_time = time, time
        time_diff = query_time.unsqueeze(2) - key_time.unsqueeze(1)

        time_diff.abs_().clamp_(max=self.time_difference_clip)  # B x T x T 
        time_diff_embeddings = self.time_emb(time_diff)
        #abs_ : computes the absolute value 
        #clamp_ : clipping
        return time_diff_embeddings  # B x T x T x H