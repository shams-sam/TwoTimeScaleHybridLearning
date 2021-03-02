import argparse
# import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
from common.utils import Struct


# matplotlib.rcParams.update({'font.size': 37})
# matplotlib.rcParams['lines.linewidth'] = 2.0
# matplotlib.rcParams['lines.markersize'] = 8

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

fig = plt.figure(figsize=(15, 8))
ax1 = fig.add_subplot(231)
ax2 = fig.add_subplot(232)
ax3 = fig.add_subplot(233)
ax4 = fig.add_subplot(234)
ax5 = fig.add_subplot(235)
ax6 = fig.add_subplot(236)

colors = ['k.-', 'r.:', 'm.:', 'b.:', 'g.:', 'c.:', 'y.:', 'k.:', 'r', 'b']
if len(args.colors):
    colors = args.colors

for idx, history in enumerate(args.histories):
    x_ax, y_ax, l_test, rounds, eps, eta_phi, beta, mu = pkl.load(
        open('../ckpts/{}_{}/history/{}'.format(
            args.dataset, args.num_nodes, history), 'rb'))
    x_ax = x_ax[:args.epochs]
    y_ax = y_ax[:args.epochs]
    l_test = l_test[:args.epochs]
    rounds = rounds[:args.epochs]
    eps = eps[:args.epochs]
    eta_phi = eta_phi[:args.epochs]
    beta = beta[:args.epochs]
    mu = mu[:args.epochs]

    ax1.plot(x_ax, y_ax, colors[idx], label=args.labels[idx])
    ax2.plot(x_ax, l_test, colors[idx], label=args.labels[idx])
    if sum(rounds) >= 0:
        ax3.plot(x_ax, rounds, colors[idx], label=args.labels[idx])
    if sum(eps) >= 0:
        ax4.plot(x_ax, eps, colors[idx], label=args.labels[idx])
    if sum(eta_phi) >= 0:
        ax4.plot(x_ax, eta_phi, colors[idx][0], linestyle='dashed', label=args.labels[idx])
    if sum(beta) >= 0:
        ax5.plot(x_ax, beta, colors[idx], label=args.labels[idx])
    if sum(mu) != 0:
        ax6.plot(x_ax, mu, colors[idx], label=args.labels[idx])

ax1.set_xlabel('t')
ax1.set_ylabel('accuracy')
ax2.set_xlabel('t')
ax2.set_ylabel('loss')
ax3.set_xlabel('t')
ax3.set_ylabel('rounds')
ax4.set_xlabel('t')
ax4.set_ylabel('epsilon')
ax5.set_xlabel('t')
ax5.set_ylabel(r'$\beta$')
ax6.set_xlabel('t')
ax6.set_ylabel(r'$\mu$')
ax1.grid(ls='-.', lw=0.25)
ax2.grid(ls='-.', lw=0.25)
ax3.grid(ls='-.', lw=0.25)
ax4.grid(ls='-.', lw=0.25)
ax5.grid(ls='-.', lw=0.25)
ax6.grid(ls='-.', lw=0.25)

ax2.legend(loc='upper right', ncol=args.ncols,
           bbox_to_anchor=(-1.33, 1.1, 2.35, .1),
           mode='expand', frameon=False)
print('Saving: ', args.name)
fig.subplots_adjust(wspace=0.3)
plt.savefig('../ckpts/{}_{}/plots/{}'.format(
    args.dataset, args.num_nodes, args.name),
            bbox_inches='tight', dpi=args.dpi)
