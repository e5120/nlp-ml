import argparse

def model_opts(parser):
  # Embedding options
  group = parser.add_argument_group("Embedding")
  group.add("--src-vec-size", type=int, default=256,
            help="word embedding size for src side.")
  group.add("--tgt-vec-size", type=int, default=256,
            help="word embedding size for tgt side.")
  
  # Encoder-Decoder options
  group = parser.add_argument_group("Encoder-Decoder")
  group.add("--enc-type", type=str, default="brnn",
            choices=["rnn", "brnn"]
            help="type of encoder.[rnn|brnn]")
  group.add("--dec-type", type=str, default="rnn",
            choices=["rnn"]
            help="type of decoder.[rnn]")
  group.add("--enc-vec-size", type=int, default=256,
            help="size of encoder rnn hidden states.")
  group.add("--dec-vec-size", type=int, default=256,
            help="size of decoder rnn hidden states.")
  group.add("--enc-layers", type=int, default=2,
            help="number of layers in the encoder.")
  group.add("--dec-layers", type=int, default=2,
            help="number of layers in the decoder.")
  group.add("--rnn-type", type=str, default="LSTM",
            choices=["LSTM", "GRU"],
            help="the gate type to use in the RNNs")
  
def train_opts(parser):
  # training and saving options
  group = parser.add_argument_group("General")
  group.add("--data", type=str, required=True,
            help="path to the 'train and 'valid' file")
  group.add("--exec-name", type=str, default="test",
            help="identification name when executes the program")
  group.add("--save-model", type=str, default="model",
            help="model filename")
  group.add("--save-checkpoint-steps", type=int, default=5000,
            help="save a checkpoint every X stepts")
  group.add("--no-gpu", action="store_true",
            help="do not want to use GPU")

  # init options
  group = parser.add_argument_group("Initialization")
  group.add("--seed", type=int, default=-1,
            help="random seed used for the experiments")

  #group.add("--pre-word-vecs-enc", type="str", default=None,
  #          help="If a valid path is specified, then this will load pretrained word embeddings on the encoder side.")
  #group.add("--pre-word-vecs-dec", type="str", default=None,
  #          help="If a valid path is specified, then this will load pretrained word embeddings on the decoder side.")
  #group.add("--fix-word-vecs-enc", action="store_true",
  #          help="fix word embeddings on the encoder side.")
  #group.add("--fix-word-vecs-dec", action="store_true",
  #          help="fix word embeddings on the decoder side.")

  # optimization options
  group = parser.add_argument_group("Optimization")
  group.add("--batch-size"< type=int, default=64,
            help="maximum batch size for training")
  group.add("--valid-steps", type=int, default=10000,
            help="perform validation every X steps")
  group.add("--valid-batch-size", type=int, default=32,
            help="maximum batch size for validation")
  group.add("--train-steps", type=int, default=100000,
            help="number of training steps")
  group.add("--optim", type=str, default="adam",
            choices=["sgd", "adagrad", "adam"],
            help="optimization method[sgd|adagrad|adam]")
  group.add("--dropout", type=float, default=0.3,
            help="dropout probability. applied in RNN stacks.")
  group.add("--adam-beta1", type=float, default=0.9,
            help="the beta1 parameter used by adam")
  group.add("--adam-beta2", type=float, default=0.999,
            help="the beta2 parameter used by adam")
  group.add("--learning-rate", type=float, default=1.0,
            help="starting learning rate")

  group = parser.add_argument_group("Logging")
  group.add("--report-every", type=int, default=50,
            help="print stats at this interval.")
  group.add("--log-file", type=str, default=None,
            help="output logs to a file under this path.")
  group.add("--tensorboard", action="store_true",
            help="use tensorboardX for visualization during training.")
            
            
def translate_opts(parser):
  group = parser.add_argument_group("Model")
  group.add("--model", type=str, required=True, nargs="+",
            help="path to model .pt file(s).")
  
  group = parser.add_argument_group("Data")
  group.add("--src", type=str, required=True,
            help="source sequence to decode (one line per sequence)")
  group.add("--tgt", type=str, default=None,
            help="True target sequence (optional)")
  group.add("--output", type=str, default="pred.txt",
            help="path to output the predictions")
  
  group = parser.add_argument_group("Logging")
  group.add("--verbose", action="store_true",
            help="print scores and predictions for each sentence")
  group.add("--log-file", type=str, default=None,
            help="output logs to a file under this path.")
  group.add("--attn-debug", action="store_true",
            help="print best attn for each word")
  group.add("--n-best", type=int, default=1,
            help="If verbose is set, will output the n_best decoded sentences")
            
  group = parser.add_argument_group("Efficiency")
  group.add("--batch-size", type=int, default=30,
            help="batch size")
  group.add("--no-gpu", action="store_true",
            help="do not want to use GPU")
  

def preprocess_opts(parser):
  pass
