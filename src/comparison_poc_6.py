import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import common.config as cfg
from common.utils import Struct


matplotlib.rcParams.update({'font.size': 20})
matplotlib.rcParams['lines.linewidth'] = 2.5
matplotlib.rcParams['lines.markersize'] = 8

ap = argparse.ArgumentParser()
ap.add_argument('--dataset', type=str, required=False, default='mnist')
ap.add_argument('--num-nodes', type=int, required=False, default=125)
ap.add_argument('--t1', type=int, required=False, default=125)
ap.add_argument('--histories', type=str, nargs='+', required=True)
ap.add_argument('--defaults', type=float, nargs='+', required=True)
ap.add_argument('--c1', type=float, nargs='+', required=True)
ap.add_argument('--c2', type=float, nargs='+', required=True)
ap.add_argument('--c3', type=float, nargs='+', required=True)
ap.add_argument('--name', type=str, required=True)
ap.add_argument('--ncols', type=int, required=True)
ap.add_argument('--dpi', type=int, required=True)
args = vars(ap.parse_args())
args = Struct(**args)

fig = plt.figure(figsize=(15, 4))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

colors = ['k.-', 'r.:', 'm.:', 'b.:', 'g.:', 'c.:', 'y.:', 'k.:', 'r', 'b']

c1, c2, c3 = args.defaults
xticks = []
for idx, c in enumerate(args.c1, 1):
    history = args.histories[0].format('_'.join(map(str,[c, c2, c3])))
    aux = args.histories[1].format('_'.join(map(str, [c, c2, c3])))
    x_ax, y_ax, l_test, rounds, eps, eta_phi = pkl.load(
        open('../ckpts/{}_{}/history/{}'.format(
            args.dataset, args.num_nodes, history), 'rb'))
    train_args, eut_schedule = pkl.load(
        open('../ckpts/{}_{}/history/{}'.format(
            args.dataset, args.num_nodes, aux), 'rb'))
    ax1.bar(idx, max(eut_schedule)-args.t1, color='b')
    xticks.append(idx)
ax1.set_xticks(xticks)
ax1.set_xticklabels(['$5.0x10^{-5}$', '$7.5x10^{-5}$', '$1.0x10^{-4}$', '$2.5x10^{-4}$', '$5.0x10^{-4}$', '$7.5x10^{-4}$', '$1.0x10^{-3}$', '$2.5x10^{-3}$', '$5.0x10^{-3}$', '$7.5x10^{-3}$', '$1.0x10^{-2}$'], rotation=90)
ax1.set_xlabel('$c_1$')
ax1.set_ylabel(r'$\tau_2$')

for idx, c in enumerate(args.c2, 1):
    history = args.histories[0].format('_'.join(map(str,[c1, c, c3])))
    aux = args.histories[1].format('_'.join(map(str, [c1, c, c3])))
    x_ax, y_ax, l_test, rounds, eps, eta_phi = pkl.load(
        open('../ckpts/{}_{}/history/{}'.format(
            args.dataset, args.num_nodes, history), 'rb'))
    train_args, eut_schedule = pkl.load(
        open('../ckpts/{}_{}/history/{}'.format(
            args.dataset, args.num_nodes, aux), 'rb'))
    ax2.bar(idx, max(eut_schedule)-args.t1, color='r')
ax2.set_xticks(xticks)
ax2.set_xticklabels(['$5.0x10^{0}$', '$7.5x10^{0}$', '$1.0x10^{1}$', '$2.5x10^{1}$', '$5.0x10^{1}$', '$7.5x10^{1}$', '$1.0x10^{2}$', '$2.5x10^{2}$', '$5.0x10^{2}$', '$7.5x10^{2}$', '$1.0x10^{3}$'], rotation=90)
ax2.set_xlabel('$c_2$')
ax2.set_ylabel(r'$\tau_2$')


for idx, c in enumerate(args.c3, 1):
    history = args.histories[0].format('_'.join(map(str,[c1, c2, c])))
    aux = args.histories[1].format('_'.join(map(str, [c1, c2, c])))
    x_ax, y_ax, l_test, rounds, eps, eta_phi = pkl.load(
        open('../ckpts/{}_{}/history/{}'.format(
            args.dataset, args.num_nodes, history), 'rb'))
    train_args, eut_schedule = pkl.load(
        open('../ckpts/{}_{}/history/{}'.format(
            args.dataset, args.num_nodes, aux), 'rb'))
    ax3.bar(idx, max(eut_schedule)-args.t1, color='g')
ax3.set_xticks(xticks)
ax3.set_xticklabels(['$7.5x10^{2}$', '$1.0x10^{3}$', '$2.5x10^{3}$', '$5.0x10^{3}$', '$7.5x10^{3}$', '$1.0x10^{4}$', '$2.5x10^{4}$', '$5.0x10^{4}$', '$7.5x10^{4}$', '$1.0x10^{5}$', '$2.5x10^{5}$'], rotation=90)
ax3.set_xlabel('$c_3$')
ax3.set_ylabel(r'$\tau_2$')


ax1.grid()
ax2.grid()
ax3.grid()

ax1.set_title('(a)', y=-0.7)
ax2.set_title('(b)', y=-0.7)
ax3.set_title('(c)', y=-0.7)

print('Saving: ', args.name)
fig.subplots_adjust(wspace=0.25)
plt.savefig('../ckpts/{}_{}/plots/{}'.format(
    args.dataset, args.num_nodes, args.name),
            bbox_inches='tight',
            dpi=args.dpi)
