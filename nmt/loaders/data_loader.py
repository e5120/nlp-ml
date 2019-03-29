import torch
from torch.utils.data import Dataset, DataLoader


UNK = "<unk>"
BOS = "<bos>"
EOS = "<eos>"
PAD = "<blank>"


class TextDataset(Dataset):
    def __init__(self, src_file, tgt_file, src_vocab, tgt_vocab, max_len=100):
        super(TextDataset, self).__init__()
        with open(src_file, encoding="utf-8") as f:
            self.src = f.read().split("\n")
        with open(tgt_file, encoding="utf-8") as f:
            self.tgt = f.read().split("\n")

        assert len(self.src) == len(self.tgt), \
                "length is not equal, {} and {}".format(len(self.src), len(self.tgt))

        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        src = self.src[idx].split(" ")
        tgt = self.tgt[idx].split(" ")

        # word to index and add EOS and BOS token
        src = [self.src_vocab[word] for word in src] + [self.src_vocab[EOS]]
        tgt = [self.tgt_vocab[BOS]] + [self.tgt_vocab[word] for word in tgt] + [self.tgt_vocab[EOS]]
        src_len = len(src)
        tgt_len = len(tgt)

        assert src_len <= self.max_len and tgt_len <= self.max_len, \
            "sequence length exceeds upper limit"

        # padding
        src += [self.src_vocab[PAD]] * (self.max_len - src_len)
        tgt += [self.tgt_vocab[PAD]] * (self.max_len - tgt_len)
        return src, tgt, src_len, tgt_len


class TextDataLoader(object):
    def __init__(self, dataset, batch_size, num_workers):
        self.dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    def __iter__(self):
        for src, tgt, src_len, tgt_len in self.dataloader:
            src = torch.unsqueeze(torch.stack(src[:max(src_len)]), -1)
            tgt = torch.unsqueeze(torch.stack(tgt[:max(tgt_len)]), -1)
            yield src, tgt, src_len, tgt_len


def get_data_loader(src_file, tgt_file, src_vocab, tgt_vocab, max_len, batch_size, num_workers=4):
    dataset = TextDataset(src_file, tgt_file, src_vocab, tgt_vocab, max_len)
    return TextDataLoader(dataset, batch_size, num_workers)


if __name__ == '__main__':
    import json

    with open("../../data/sample/train.problem.vocab.json", encoding="utf-8") as f:
        src_vocab = json.load(f)
    with open("../../data/sample/train.answer.vocab.json", encoding="utf-8") as f:
        tgt_vocab = json.load(f)

    src_file = "../../data/sample/train.problem"
    tgt_file = "../../data/sample/train.answer"

    dataloader = get_data_loader(src_file, tgt_file, src_vocab, tgt_vocab,
                                max_len=100, batch_size=10, num_workers=0)
    for i, (src, tgt, src_len, tgt_len) in enumerate(dataloader):
        print(src, tgt)
        print(src.size(), tgt.size())
        print(src_len, tgt_len)
        break
