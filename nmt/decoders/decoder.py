import torch.nn as nn


class StdDecoder(nn.Module):
    def forward(self, tgt, memory_bank):
        raise NotImplementedError

    def init_state(self, enc_final_state):
        raise NotImplementedError
