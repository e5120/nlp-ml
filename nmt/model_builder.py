import torch.nn as nn

from nmt.encoders.rnn_encoder import RNNEncoder
from nmt.models.model import NMTModel
from nmt.loaders.data_loader import PAD
from nmt.utils.misc import load_dict


def build_embedding(vocab_size, emb_dim, padding_idx):
    return nn.Embedding(num_embeddings=vocab_size,
                        embedding_dim=emb_dim,
                        padding_idx=padding_idx)

def build_encoder(args, embedding):
    if args.enc_type in ["rnn", "brnn"]:
        return RNNEncoder(embedding=embedding,
                        rnn_type=args.rnn_type,
                        rnn_hidden_size=args.enc_vec_size,
                        layers=args.enc_layers,
                        dropout=args.dropout,
                        bidirectional=args.enc_type=="brnn")
    elif args.enc_type == "cnn":
        pass
    else:
        pass


def build_decoder(args, embedding):
    return None


def build_model(args):
    src_vocab = load_dict(args.src_dict)
    src_embedding = build_embedding(len(src_vocab), args.src_vec_size, src_vocab[PAD])
    enc = build_encoder(args, src_embedding)

    tgt_vocab = load_dict(args.tgt_dict)
    tgt_embedding = build_embedding(len(tgt_vocab), args.tgt_vec_size, tgt_vocab[PAD])
    dec = build_decoder(args, tgt_embedding)

    model = NMTModel(enc, dec).to(args.device)

    return model
