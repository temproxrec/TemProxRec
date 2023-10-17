
import torch.nn as nn
from .modules.encoder import Encoder
from torch.nn import functional as F

class Model(nn.Module):

    def __init__(self, encoder: Encoder):
        """
        :param encoder: encoder model which should be trained
        """
        super().__init__()
        self.encoder = encoder
    def forward(self, x, abs_seq):
        out, hidden = self.encoder(x, abs_seq)
        return out, hidden
