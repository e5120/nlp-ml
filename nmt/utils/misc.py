import json

import torch.nn as nn


def rnn_create(type, **kwargs):
    rnn = getattr(nn, type)(**kwargs)
    return rnn


def load_dict(filename):
    with open(filename, encoding="utf-8") as f:
        return json.load(f)
