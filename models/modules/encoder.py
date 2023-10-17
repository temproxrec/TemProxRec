import torch
from torch import nn as nn

from models.modules.embedding.embedding import Embedding
from models.modules.embedding.time import RelativeTimeEmbedding, AbsoluteTimeEmbedding
from models.modules.embedding.position import AbsolutePositionalEmbedding, RelativePositionEmbedding
from models.modules.transformer import TransformerBlock
from models.modules.attention.multi_head import Attention
from utils import fix_seed


class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        vocab_size = args.num_items + 2
        time_vocab_size = args.num_time_items + 1

        time_clip = args.clip_time
        position_clip = 2
        
        n_layers = args.num_blocks 
        dropout = args.dropout
        hidden = args.hidden_units
        
        self.hidden=hidden
        self.max_len = args.maxlen
        self.device = torch.device("cuda:"+args.device_idx)
        self.hidden=hidden
        
        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = Embedding(vocab_size, hidden, dropout) # item은 padding과 mask, time은 padding 
        self.dropout = nn.Dropout(p=dropout)
        
        
        #4 different embeddings
        rel_t = RelativeTimeEmbedding(time_clip, hidden, self.device)
        rel_p = RelativePositionEmbedding(hidden, position_clip, self.device)
        abs_t = AbsoluteTimeEmbedding(time_vocab_size, hidden)
        abs_p = AbsolutePositionalEmbedding(self.max_len, hidden)
        
        # relative head kernerls
        self.relative_embeddings_list = nn.ModuleList()
        self.relative_embeddings_list.append(rel_t)
        self.relative_embeddings_list.append(rel_p)
        self.num_relheads = len(self.relative_embeddings_list)
        
        # absolute head kernerls
        self.absolute_embeddings_list = nn.ModuleList()
        self.absolute_embeddings_list.append(abs_t)
        self.absolute_embeddings_list.append(abs_p)
        self.num_absheads = len(self.absolute_embeddings_list)
        
        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(args, self.num_relheads, self.num_absheads) for _ in range(n_layers)])
        
        fix_seed(args)
        self.init_weights()
        
        self.out = nn.Linear(self.hidden, args.num_items + 1)
        
    def forward(self, x, time_x):

        attn_mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)  # B x 1 x T x T
        x = self.embedding(x)

        relative_embeddings = [self.dropout(emb(time_x, self.max_len)) for emb in self.relative_embeddings_list]  # Lr of [B x T x T x H]
        absolute_embeddings = [self.dropout(emb(time_x)) for emb in self.absolute_embeddings_list]
        
        for layer, transformer in enumerate(self.transformer_blocks):
            x = transformer.forward(x, attn_mask, relative_embeddings, absolute_embeddings, layer=layer)
        pred = self.out(x)
        return pred, x
    
    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # override in each submodel if needed
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, Attention):
            for param in [module.rel_position_bias]:
                param.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()