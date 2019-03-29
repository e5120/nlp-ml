import json

import configargparse
import torch
import torch.nn as nn

from nmt import options
from nmt.model_builder import build_model, build_optimizer
from nmt.loaders.data_loader import get_data_loader
from nmt.utils.logging import init_logger, logger


def validation():
    pass


def train(args, model, optimizer, criterion):
    steps = 0
    while steps < args.train_steps:
        dataloader = get_data_loader(args.src, args.tgt, args.src_dict, args.tgt_dict,
                                    args.max_len, args.batch_size, args.device)
        for src, tgt, src_len, tgt_len in dataloader:
            steps += 1

            optimizer.zero_grad()

            output = model(src, tgt, src_len)

            loss = criterion(output, tgt[1:].view(-1))
            loss.backward()
            optimizer.step()

            if steps % 50 == 0:
                logger.info("loss:{}".format(loss.item()))

            if steps % args.save_checkpoint_steps == 0:
                logger.info("save the model")

            if steps % args.valid_steps == 0:
                validation()

            if steps >= args.train_steps:
                break

def main(args):
    init_logger()

    model = build_model(args)
    optimizer = build_optimizer(args, model)
    criterion = nn.NLLLoss().to(args.device)
    logger.info(model)
    train(args, model, optimizer, criterion)


if __name__ == '__main__':
    parser = configargparse.ArgumentParser(description="train the model")

    options.train_opts(parser)
    options.model_opts(parser)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.no_gpu:
        device = "cpu"
    args.device = device

    main(args)
