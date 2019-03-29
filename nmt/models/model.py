import torch.nn as nn
import torch.nn.functional as F


class NMTModel(nn.Module):
    def __init__(self, enc, dec):
        super(NMTModel, self).__init__()
        self.enc = enc
        self.dec = dec

    def forward(self, src, tgt, src_len):
        tgt = tgt[:-1]

        final_state, output = self.enc(src)

        #self.dec.init_state(final_state)
        final_state, output = self.dec(tgt, output, final_state)

        return F.log_softmax(output.view(-1, output.size(2)))
