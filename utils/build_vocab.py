import argparse
import collections
import json


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="build dictionary of vocabulary")

    parser.add_argument("--path", type=str, required=True,
                        help="")
    parser.add_argument("--file", type=str, required=True,
                        help="")

    args = parser.parse_args()


    vocab = collections.Counter()
    vocab["<unk>"] = 0
    vocab["<bos>"] = 1
    vocab["<eos>"] = 2
    vocab["<blank>"] = 3
    idx = 4
    with open("{}/{}".format(args.path, args.file), encoding="utf-8") as f:
        for line in f:
            line = line.strip().split(" ")
            for subword in line:
                if subword not in vocab:
                    vocab[subword] = idx
                    idx += 1
    with open("{}/{}.vocab.json".format(args.path, args.file), "w", newline="\n", encoding="utf-8") as f:
        json.dump(vocab, f, indent=4, ensure_ascii=False)
