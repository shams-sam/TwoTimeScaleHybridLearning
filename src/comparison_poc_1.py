import argparse
import matplotlib
import matplotlib.pyplot as plt
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

hist1, hist2, hist3 = args.histories[:6], args.histories[6:12], args.histories[12:]

for idx, history in enumerate(hist1):
    x_ax, y_ax, l_test = history_parser(args.dataset, args.num_nodes, history)
    x_ax = x_ax[:args.epochs]
    y_ax = y_ax[:args.epochs]
    l_test = [_*100 for _ in l_test[:args.epochs]]

    ax1.plot(x_ax, y_ax, colors[idx], label=args.labels[idx])
    ax4.plot(x_ax, l_test, colors[idx], label=args.labels[idx])

ax1.set_xlabel('t')
ax1.set_ylabel('accuracy')
ax4.set_xlabel('t')
ax4.set_ylabel('loss ($x 10^{-2}$)')
ax1.grid()
ax4.grid()


for idx, history in enumerate(hist2):
    x_ax, y_ax, l_test = history_parser(args.dataset, args.num_nodes, history)
    x_ax = x_ax[:args.epochs]
    y_ax = y_ax[:args.epochs]
    l_test = [_*100 for _ in l_test[:args.epochs]]

    ax2.plot(x_ax, y_ax, colors[idx], label=args.labels[idx])
    ax5.plot(x_ax, l_test, colors[idx], label=args.labels[idx])

ax2.set_xlabel('t')
ax2.set_ylabel('accuracy')
ax5.set_xlabel('t')
ax5.set_ylabel('loss ($x 10^{-2}$)')
ax2.grid()
ax5.grid()


for idx, history in enumerate(hist3):
    x_ax, y_ax, l_test = history_parser(args.dataset, args.num_nodes, history)
    x_ax = x_ax[:args.epochs]
    y_ax = y_ax[:args.epochs]
    l_test = [_*100 for _ in l_test[:args.epochs]]

    ax3.plot(x_ax, y_ax, colors[idx], label=args.labels[idx])
    ax6.plot(x_ax, l_test, colors[idx], label=args.labels[idx])

ax3.set_xlabel('t')
ax3.set_ylabel('accuracy')
ax6.set_xlabel('t')
ax6.set_ylabel('loss ($x 10^{-2}$)')
ax3.grid()
ax6.grid()


ax3.legend(loc='upper right', ncol=args.ncols,
           bbox_to_anchor=(-2.7, 1.1, 3.7, .32),
           mode='expand', frameon=False)

ax1.set_title('(a) 10 labels/node', y=-0.45)
ax2.set_title('(b) 3 labels/node', y=-0.45)
ax3.set_title('(c) 1 label/node', y=-0.45)
ax4.set_title('(d) 10 labels/node', y=-0.45)
ax5.set_title('(e) 3 labels/node', y=-0.45)
ax6.set_title('(f) 1 label/node', y=-0.45)

print('Saving: ', args.name)
fig.subplots_adjust(wspace=0.35, hspace=0.5)
plt.savefig('../ckpts/{}_{}/plots/{}'.format(
    args.dataset, args.num_nodes, args.name),
            bbox_inches='tight', dpi=args.dpi)
