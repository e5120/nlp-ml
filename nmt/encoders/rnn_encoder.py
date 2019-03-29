from nmt.encoders.encoder import StdEncoder

from nmt.utils.misc import rnn_create


class RNNEncoder(StdEncoder):
    def __init__(self, embedding, rnn_type, rnn_hidden_size,
                layers, dropout, bidirectional):
        super(StdEncoder, self).__init__()

        num_directions = 2 if bidirectional else 1
        assert rnn_hidden_size // num_directions
        rnn_hidden_size = rnn_hidden_size // num_directions

        self.embedding = embedding
        self.rnn = rnn_create(rnn_type,
                            input_size=embedding.embedding_dim,
                            hidden_size=rnn_hidden_size,
                            num_layers=layers,
                            dropout=dropout,
                            bidirectional=bidirectional)

    def forward(self, src):
        emb = self.embedding(src)
        output, final_state = self.rnn(emb)
        return final_state, output
