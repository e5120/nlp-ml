import torch
import torch.nn as nn

from nmt.decoders.decoder import StdDecoder
from nmt.utils.misc import rnn_create


class RNNDecoder(StdDecoder):
    def __init__(self, embedding, rnn_type, rnn_hidden_size,
                layers, dropout, bidirectional_enc):
        super(RNNDecoder, self).__init__()
        self.bidirectional_enc = bidirectional_enc
        self.embedding = embedding
        self.rnn = rnn_create(rnn_type,
                            input_size=embedding.embedding_dim,
                            hidden_size=rnn_hidden_size,
                            num_layers=layers,
                            dropout=dropout)

        self.output = nn.Linear(rnn_hidden_size, embedding.num_embeddings)


    def forward(self, tgt, memory_bank, init_state):
        def concat_vec_state(state):
            if self.bidirectional_enc:
                state = torch.cat([state[0:state.size(0):2],
                                        state[1:state.size(0):2]], 2)
            return state

        if isinstance(init_state, tuple):
            h, c = init_state
            h = concat_vec_state(h)
            c = concat_vec_state(c)
            init_state = tuple((h, c))
        else:
            init_state = (concat_vec_state(init_state), )

        emb = self.embedding(tgt)
        output, final_state = self.rnn(emb, init_state)
        output = self.output(output)

        return final_state, output

    def init_state(self, enc_final_state):
        #if isinstance(enc_final_state, tuple):
        pass
