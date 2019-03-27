import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="split the data to train, dev and test")

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
