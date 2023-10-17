import torch
import torch.nn as nn

class AbsolutePositionalEmbedding(nn.Module):

    def __init__(self, max_len, d_model):
        super().__init__()
        # Compute the positional encodings once in log space.
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        batch_size = x.size(0)

        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)
    
    
class RelativePositionEmbedding(nn.Module):

    def __init__(self, num_units, max_relative_position, device):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        self.device = device
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, time_x, length_q):
        
        length_k = length_q
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position) 
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat).to(self.device)
        embeddings = self.embeddings_table[final_mat].to(self.device)
        embeddings = embeddings.expand(time_x.size()[0], -1, -1, -1) #[batchsize, maxlen, maxlen, hidden]

        return embeddings



