import argparse
from utils import booltype, Struct


def argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--clf', type=str, required=True)
    parser.add_argument('--paradigm', type=str, required=True)
    parser.add_argument('--num-workers', type=int, required=True)
    parser.add_argument('--num-clusters', type=int, nargs='+', required=True)
    parser.add_argument('--uniform-clusters', type=booltype, required=False, default=True)
    parser.add_argument('--shuffle-workers', type=booltype, required=False, default=False)
    parser.add_argument('--batch-size', type=int, required=False, default=0)
    parser.add_argument('--test-batch-size', type=int, required=False, default=0)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--nesterov', type=booltype, required=False, default=False)
    parser.add_argument('--decay', type=float, required=False, default=1e-5)
    parser.add_argument('--no-cuda', type=booltype, required=False, default=False)
    parser.add_argument('--seed', type=int, required=False, default=1)
    parser.add_argument('--log-interval', type=int, required=False, default=1)
    parser.add_argument('--save-model', type=booltype, required=False, default=True)
    parser.add_argument('--stratify', type=booltype, required=False, default=True)
    parser.add_argument('--uniform-data', type=booltype, required=False, default=True)
    parser.add_argument('--shuffle-data', type=booltype, required=False, default=True)
    parser.add_argument('--non-iid', type=int, required=True)
    parser.add_argument('--repeat', type=int, required=True)
    parser.add_argument('--shuffle-worker-data', type=booltype, required=False, default=False)
    parser.add_argument('--rounds', type=int, required=False, default=0)
    parser.add_argument('--sigma-mul', type=float, required=False, default=0)
    parser.add_argument('--const-graph', type=booltype, required=False, default=True)
    parser.add_argument('--radius', type=str, required=False, default='multi')
    parser.add_argument('--d2d', type=float, required=False, default=1.0)
    parser.add_argument('--lut-int', type=int, required=False, default=0)
    parser.add_argument('--eut-int', type=int, required=False, default=1)
    parser.add_argument('--eut-gamma', type=float, required=False, default=1)
    parser.add_argument('--factor', type=int, required=False, default=2)
    parser.add_argument('--topology', type=str, required=False, default='rgg')
    parser.add_argument('--dry-run', type=booltype, required=False, default=True)

    args = vars(parser.parse_args())
    args = Struct(**args)

    return args
