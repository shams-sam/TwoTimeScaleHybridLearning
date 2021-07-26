import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
from common.utils import Struct, history_parser


matplotlib.rcParams.update({'font.size': 20})
matplotlib.rcParams['lines.linewidth'] = 2.5
matplotlib.rcParams['lines.markersize'] = 7

ap = argparse.ArgumentParser()
ap.add_argument('--dataset', type=str, required=False, default='mnist')
ap.add_argument('--num-nodes', type=int, required=False, default=125)
ap.add_argument('--epochs', type=int, required=False)
ap.add_argument('--histories', type=str, nargs='+', required=True)
ap.add_argument('--labels', type=str, nargs='+', required=True)
ap.add_argument('--name', type=str, required=True)
ap.add_argument('--ncols', type=int, required=True)
ap.add_argument('--dpi', type=int, required=True)
ap.add_argument('--colors', type=str, nargs='+', required=False, default=[])
args = vars(ap.parse_args())
args = Struct(**args)

fig = plt.figure(figsize=(10, 4))
ax1 = fig.add_subplot(121)
ax4 = fig.add_subplot(122)

colors = ['k.-', 'r.:', 'm.:', 'b.:', 'g.:', 'c.:', 'y.:', 'k.:', 'r', 'b']
if len(args.colors):
    colors = args.colors

hist1 = args.histories[:6]

for idx, history in enumerate(hist1):
    x_ax, y_ax, l_test = history_parser(args.dataset, args.num_nodes, history)
    x_ax = x_ax[:args.epochs]
    y_ax = y_ax[:args.epochs]
    l_test = [_*100 for _ in l_test[:args.epochs]]

    ax1.plot(x_ax, y_ax, colors[idx], label=args.labels[idx])
    ax4.plot(x_ax, l_test, colors[idx], label=args.labels[idx])

ax1.set_xlim(left=0, right=args.epochs)
ax4.set_xlim(left=0, right=args.epochs)
ax1.set_yticks(np.arange(0, 0.91, 0.15))

ax1.set_xlabel('t')
ax1.set_ylabel('accuracy')
ax4.set_xlabel('t')
ax4.set_ylabel('loss ($x 10^{-2}$)')
ax1.grid()
ax4.grid()

ax4.legend(loc='upper right', ncol=args.ncols,
           bbox_to_anchor=(-1.65, 1.1, 2.7, .24),
           mode='expand', frameon=False)

# ax1.set_title('(a) 10 labels/node', y=-0.45)
# ax4.set_title('(d) 10 labels/node', y=-0.45)

print('Saving: ', args.name)
fig.subplots_adjust(wspace=0.35, hspace=0.5)
plt.savefig('../ckpts/{}_{}/plots/{}'.format(
    args.dataset, args.num_nodes, args.name),
    bbox_inches='tight', dpi=args.dpi)
