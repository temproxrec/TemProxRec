import torch
import torch.nn as nn
from torch.nn import functional as F

class Attention(nn.Module):
    def __init__(self, args, num_relheads, num_absheads):
        super().__init__()
        d_model = args.hidden_units
        dropout = args.dropout
        
        h = num_relheads + num_absheads
        
        self.h_abs = num_absheads
        self.h_rel = num_relheads
        self.d_k = d_model // h
        self.h = h
        self.scale = 1 / (self.d_k ** 0.5)
        
        ## TODO (reference : )
        self.content_linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.abs_position_query_linear_layers = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(num_absheads)])
        self.abs_position_key_linear_layers = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(num_absheads)])
        
        self.rel_position_key_linear_layers = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(num_relheads)])
        self.rel_position_bias = nn.Parameter(torch.FloatTensor(1, self.h_rel, 1, self.d_k))
        
        ## OUTPUT
        self.output_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, query, key, value, mask, rel_kernel, abs_kernel, layer):

        # q, k, v : B x T x H
        batch_size, T = query.size(0), query.size(1)
        
        # q, k, v, kernel_q, kernel_k : B x n x T x d
        query, key, value = \
            [l(x).view(batch_size, T, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.content_linear_layers, (query, key, value))]

        scores = torch.zeros(batch_size, self.h, T, T).to(query)
        
        
        if self.h_abs > 0:
            Xq = query[:, :self.h_abs]  # B x La x T x d
            Xk = key[:, :self.h_abs]  # B x La x T x d
            Pq = torch.stack([l(x) for l, x in zip(self.abs_position_query_linear_layers, abs_kernel)], dim=1)  # B x La x T x d
            Pk = torch.stack([l(x) for l, x in zip(self.abs_position_key_linear_layers, abs_kernel)], dim=1)  # B x La x T x d
            abs_scores = torch.einsum('blid,bljd->blij', Xq + Pq, Xk + Pk)  # B x La x T x T
            scores[:, :self.h_abs] += abs_scores

        if self.h_rel > 0:
            Xq = query[:, self.h_abs:]  # B x Lr x T x d
            Xk = key[:, self.h_abs:]  # B x Lr x T x d
            R = torch.stack([l(x) for l, x in zip(self.rel_position_key_linear_layers, rel_kernel)], dim=1)  # B x Lr x T x T x d
            rel_scores = torch.einsum('blid,bljd->blij', Xq, Xk)  # B x Lr x T x T
            rel_scores += torch.einsum('blid,blijd->blij', Xq + self.rel_position_bias, R)  # B x Lr x T x T
            scores[:, self.h_abs:] += rel_scores


        scores = scores * self.scale
        
        scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)  # B x n x T x T
        p_attn = self.dropout(p_attn)
        x = torch.matmul(p_attn, value)  # B x n x T x d

        x = x.transpose(1, 2).contiguous().view(batch_size, T, self.h * self.d_k)  # B x T x H
        x = self.output_linear(x)  #  B x T x H

        return x
    # referenc code : https://github.com/SungMinCho/MEANTIME