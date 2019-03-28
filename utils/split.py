import random
import argparse

import numpy as np


def write(args, prefix, src, tgt):
    with open("{}/{}.{}".format(args.output_path, prefix, args.src), "w", encoding="utf-8") as s, \
        open("{}/{}.{}".format(args.output_path, prefix, args.tgt), "w", encoding="utf-8") as t:
        s.write("\n".join(src))
        t.write("\n".join(tgt))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="split the data to train, dev and test")

    parser.add_argument("--input-path", type=str, required=True,
                        help="")
    parser.add_argument("--output-path", type=str, default=None,
                        help="")
    parser.add_argument("--src", type=str, required=True,
                        help="")
    parser.add_argument("--tgt", type=str, required=True,
                        help="")
    parser.add_argument("--shuffle", action="store_true",
                        help="randomly shuffle each line sentence")
    parser.add_argument("--train", type=float, default=0.8,
                        help="ratio for train data")
    parser.add_argument("--dev", type=float, default=0.1,
                        help="ratio for dev data")
    parser.add_argument("--test", type=float, default=0.1,
                        help="ratio for test data")


    args = parser.parse_args()
    assert abs(1.0 - (args.train + args.dev + args.test)) < 1e-8, \
            "total of ratio is not 1.0"

    if args.output_path is None:
        args.output_path = args.input_path

    with open("{}/{}".format(args.input_path, args.src), encoding="utf-8") as f, \
        open("{}/{}".format(args.input_path, args.tgt), encoding="utf-8") as g:

        src = np.array(f.read().split("\n"))
        tgt = np.array(g.read().split("\n"))

        assert len(src) == len(tgt), "length is not equal, {} and {}".format(len(src), len(tgt))

        if args.shuffle is True:
            pem = np.random.permutation(len(src))
            src, tgt = src[pem], tgt[pem]

        idx1 = int(args.train * len(src))
        idx2 = int(args.dev * len(src) + idx1)

        write(args, "train", src[:idx1], tgt[:idx1])
        write(args, "dev", src[idx1:idx2], tgt[idx1:idx2])
        write(args, "test", src[idx2:], tgt[idx2:])
