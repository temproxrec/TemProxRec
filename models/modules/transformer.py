import torch.nn as nn
from .attention.multi_head import Attention
from .utils import SublayerConnection, PositionwiseFeedForward


class TransformerBlock(nn.Module):
    def __init__(self, args, Lr, La):
        super().__init__()

        hidden = args.hidden_units
        feed_forward_hidden = hidden * 4
        dropout = args.dropout
        self.attention = Attention(args, Lr, La)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
 
        
    def forward(self, x, mask, rel_kernel, abs_kernel, layer):
        # x : B x T x H
        # abs : La of [B x T x H]
        # rel : Lr of [B x T x T x H]
        x = self.input_sublayer(x, lambda _x: self.attention(_x, _x, _x, mask, rel_kernel, abs_kernel, layer))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)
    