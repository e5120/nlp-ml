import torch.nn as nn


class NMTModel(nn.Module):
    def __init__(self, enc, dec):
        super(NMTModel, self).__init__()
        self.enc = enc
        self.dec = dec

    def forward(self, src, tgt, src_len):
        tgt = tgt[:-1]

        final_state, output = self.enc(src)

        return
