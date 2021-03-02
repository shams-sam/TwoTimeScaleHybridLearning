import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
from common.utils import Struct


matplotlib.rcParams.update({'font.size': 10})
matplotlib.rcParams['lines.linewidth'] = 1.5
matplotlib.rcParams['lines.markersize'] = 4

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
ax1 = fig.add_subplot(231)
ax2 = fig.add_subplot(232)

colors = ['k.-', 'r.:', 'm.:', 'b.:', 'g.:', 'c.:', 'y.:', 'k.:', 'r', 'b']
if len(args.colors):
    colors = args.colors

def parser(dataset, num_nodes, history):
    h = pkl.load(
        open('../ckpts/{}_{}/history/{}'.format(
            dataset, num_nodes, history), 'rb'))
    if len(h) == 8:
        x_ax, y_ax, l_test, rounds, eps, eta_phi, beta, mu = h
    else:
        x_ax, y_ax, l_test, rounds, eps, eta_phi = h

    return x_ax, y_ax, l_test


k = 100
for idx, history in enumerate(args.histories):
    x_ax, y_ax, l_test = parser(args.dataset, args.num_nodes, history)
    x_ax = x_ax[:args.epochs]
    y_ax = y_ax[:args.epochs]
    l_test = l_test[:args.epochs]

    ax1.plot(x_ax, y_ax, colors[idx], label=args.labels[idx])
    ax2.plot(x_ax, np.array(l_test)*k, colors[idx], label=args.labels[idx])

ax1.set_xlabel('t')
ax1.set_ylabel('accuracy')
ax2.set_xlabel('t')
ax2.set_ylabel('loss ($x 10^{-2}$)')
ax1.grid(ls='-.', lw=0.25)
ax2.grid(ls='-.', lw=0.25)

ax2.legend(loc='upper right', ncol=args.ncols,
           bbox_to_anchor=(-1.33, 1.1, 2.35, .3),
           mode='expand', frameon=False)

ax1.set_title('(a)', y=-0.5)
ax2.set_title('(b)', y=-0.5)

print('Saving: ', args.name)
fig.subplots_adjust(wspace=0.35)
plt.savefig('../ckpts/{}_{}/plots/{}'.format(
    args.dataset, args.num_nodes, args.name),
            bbox_inches='tight', dpi=args.dpi)
