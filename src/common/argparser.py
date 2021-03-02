import argparse
from common.utils import booltype, Struct


def argparser():
    parser = argparse.ArgumentParser()

    # expt config
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--clf', type=str, required=True)
    parser.add_argument('--paradigm', type=str, required=True)

    # stop criterion
    parser.add_argument('--accuracy', type=float, required=True)
    parser.add_argument('--patience', type=int, required=True)

    # clustering config
    parser.add_argument('--num-workers', type=int, required=True)
    parser.add_argument('--num-clusters', type=int, nargs='+', required=True)
    parser.add_argument('--uniform-clusters', type=booltype, required=False, default=True)
    parser.add_argument('--shuffle-workers', type=booltype, required=False, default=False)

    # worker training config
    parser.add_argument('--batch-size', type=int, required=False, default=0)
    parser.add_argument('--test-batch-size', type=int, required=False, default=0)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--decay', type=float, required=False, default=1e-5)
    parser.add_argument('--no-cuda', type=booltype, required=False, default=False)
    parser.add_argument('--seed', type=int, required=False, default=1)
    parser.add_argument('--stratify', type=booltype, required=False, default=True)
    parser.add_argument('--uniform-data', type=booltype, required=False, default=True)
    parser.add_argument('--shuffle-data', type=booltype, required=False, default=True)
    parser.add_argument('--non-iid', type=int, required=True)
    parser.add_argument('--repeat', type=int, required=True)

    # consensus config
    parser.add_argument('--factor', type=int, required=False, default=2)
    parser.add_argument('--topology', type=str, required=False, default='rgg')
    parser.add_argument('--const-graph', type=booltype, required=False, default=True)
    parser.add_argument('--eut-range', type=int, nargs='+', required=False)
    parser.add_argument('--eut-seed', type=int, required=False)
    parser.add_argument('--lut-intv', type=int, required=False)
    parser.add_argument('--rounds', type=int, required=False)

    # constants
    parser.add_argument('--beta', type=float, required=False)
    parser.add_argument('--mu', type=float, required=False)
    parser.add_argument('--delta', type=float, required=False)
    parser.add_argument('--zeta', type=float, required=False)
    parser.add_argument('--phi', type=float, required=False)
    parser.add_argument('--sigma', type=float, required=False)
    parser.add_argument('--xi', type=float, required=False)
    parser.add_argument('--tau-max', type=int, required=False)
    parser.add_argument('--e-frac', type=float, required=False)
    parser.add_argument('--d-frac', type=float, required=False)
    parser.add_argument('--cs', type=float, nargs='+', required=False)

    # logging and debug
    parser.add_argument('--log-intv', type=int, required=False, default=1)
    parser.add_argument('--save-model', type=booltype, required=False, default=True)
    parser.add_argument('--dry-run', type=booltype, required=False, default=True)

    args = vars(parser.parse_args())
    args = Struct(**args)

    return args
