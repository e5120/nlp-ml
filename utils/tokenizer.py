import os.path
import argparse

import sentencepiece as spm
import spacy


def get_word_tokenizer(args):
    if args.lang == "en":
        tokenizer = spacy.load("en")
    elif args.lang == "ja":
        tokenizer = spacy.blank("ja")
    else:
        print("language error")
        exit()

    return lambda x: " ".join([tok.text for tok in tokenizer.tokenizer(x)])

def get_spm_tokenizer(args):
    def spm_train(args):
        argument = "--input=" + args.input_path + "/" + args.input_file + \
                " --model_prefix=" + model_path + \
                " --vocab_size=" + str(args.vocab_size) + \
                " --model_type=" + args.type
        spm.SentencePieceTrainer.Train(argument)
        return

    model_path = "{}/{}.{}".format(args.input_path, args.lang, args.type)
    if os.path.exists(model_path + ".model") is False:
        print("There is not the model. Do you train the model? [y/n]")
        ans = input()
        if ans[0] != "y" and ans[0] != "Y":
            exit()
        spm_train(args)

    sp = spm.SentencePieceProcessor()
    sp.Load(model_path + ".model")
    return lambda x: " ".join(sp.EncodeAsPieces(x))

def document_tokenize(args):
    if args.type == "word":
        fn_tokenizer = get_word_tokenizer(args)
    else:
        fn_tokenizer = get_spm_tokenizer(args)

    with open("{}/{}".format(args.input_path, args.input_file), encoding="utf-8") as f:
        seqs = []
        for i, line in enumerate(f, 1):
            seqs.append(fn_tokenizer(line.strip()))
            if i % 100000 == 0:
                print("processed {} sentences".format(i))
        with open("{}/{}".format(args.output_path, args.output_file), "w", encoding="utf-8") as g:
            g.write("\n".join(seqs))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="tokenizer for text files")

    parser.add_argument("--input-file", metavar="FILE", type=str, required=True,
                        help="filename of input file")
    parser.add_argument("--output-file", metavar="FILE", type=str, default=None,
                        help="filename of output file(default:input-file + '.' + type)")
    parser.add_argument("--input-path", metavar="PATH", type=str, required=True,
                        help="path of input file")
    parser.add_argument("--output-path", metavar="PATH", type=str, default=None,
                        help="path of output file(default:same as input-path)")
    parser.add_argument("--type", metavar="TYPE", choices=["word", "bpe", "unigram", "char"],
                        type=str, default="word", help="model type[word|bpe|unigram|char](default:word)")
    parser.add_argument("--vocab-size", type=int, default=30000,
                        help="vocabulary size(default:30000). use it except 'word' type ")
    parser.add_argument("--lang", metavar="LANG", type=str, choices=["en", "ja"], required=True,
                        help="language")

    args = parser.parse_args()

    if args.output_path is None:
        args.output_path = args.input_path
    assert os.path.exists(args.output_path) == True, "Invalid output path\n"

    if args.output_file is None:
        args.output_file = args.input_file + "." + args.type


    print(args)
    document_tokenize(args)
    print("successed")
