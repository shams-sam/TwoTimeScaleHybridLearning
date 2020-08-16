import argparse
# import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
from utils import Struct


# matplotlib.rcParams.update({'font.size': 37})
# matplotlib.rcParams['lines.linewidth'] = 2.0
# matplotlib.rcParams['lines.markersize'] = 8

ap = argparse.ArgumentParser()
ap.add_argument('--dataset', type=str, required=False, default='mnist')
ap.add_argument('--num-nodes', type=int, required=False, default=125)
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
ax2 = fig.add_subplot(122)

colors = ['k.-', 'r.:', 'm.:', 'b.:', 'g.:', 'c.:', 'y.:', 'k.:']
if len(args.colors):
    colors = args.colors

for idx, history in enumerate(args.histories):
    x_ax, y_ax, l_test = pkl.load(
        open('../ckpts/{}_{}/history/{}'.format(
            args.dataset, args.num_nodes, history), 'rb'))
    ax1.plot(x_ax, y_ax, colors[idx], label=args.labels[idx])
    ax2.plot(x_ax, l_test, colors[idx], label=args.labels[idx])

ax1.set_xlabel('epochs')
ax1.set_ylabel('accuracy')
ax2.set_xlabel('epochs')
ax2.set_ylabel('loss')
ax1.grid(ls='-.', lw=0.25)
ax2.grid(ls='-.', lw=0.25)
ax2.legend(loc='upper right', ncol=args.ncols,
           bbox_to_anchor=(-1.33, 1.1, 2.35, .1),
           mode='expand', frameon=False)
print('Saving: ', args.name)
fig.subplots_adjust(wspace=0.3)
plt.savefig('../ckpts/{}_{}/plots/{}'.format(
    args.dataset, args.num_nodes, args.name),
            bbox_inches='tight', dpi=args.dpi)
