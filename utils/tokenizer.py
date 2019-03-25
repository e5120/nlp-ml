import argparse

import sentencepiece as spm
import spacy

def word_tokenize(args):
    if args.lang == "en":
        tokenizer = spacy.load("en")
    elif args.lang == "ja":
        tokenizer = spacy.blank("ja")
    else:
        print("language error")
        exit()

    with open(args.input_path + "/" + args.input_file, encoding="utf-8") as f:
        seqs = []
        for i, line in enumerate(f, 1):
            seqs.append(" ".join([tok.text for tok in tokenizer.tokenizer(line.strip())]))
            if i % 10000 == 0:
                print("processed {} sentences".format(i))
        with open(args.output_path + "/" + args.output_file, "w", encoding="utf-8") as g:
            g.write("\n".join(seqs))

def spm_tokenize(args):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="tokenizer for text files")

    parser.add_argument("--input-file", metavar="FILE", type=str, required=True,
                        help="filename of input file")
    parser.add_argument("--output-file", metavar="FILE", type=str, default=None,
                        help="filename of output file(default:input-file + "." + type)")
    parser.add_argument("--input-path", metavar="PATH", type=str, required=True,
                        help="path of input file")
    parser.add_argument("--output-path", metavar="PATH", type=str, default=None,
                        help="path of output file(default:same as input-path)")
    parser.add_argument("--type", metavar="TYPE", choices=["word", "bpe", "unigram", "char"],
                        type=str, default="word", help="model type")
    parser.add_argument("--lang", metavar="LANG", type=str, choices=["en", "ja"], required=True,
                        help="language")

    args = parser.parse_args()
    if args.output_path is None:
        args.output_path = args.input_path

    if args.output_file is None:
        args.output_file = args.input_file + "." + args.type

    print(args)
    if args.type == "word":
        word_tokenize(args)
    else:
        spm_tokenize(args)
