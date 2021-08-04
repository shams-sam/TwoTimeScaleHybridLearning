import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import common.config as cfg
from common.utils import Struct, history_parser


matplotlib.rcParams.update({'font.size': 20})
matplotlib.rcParams['lines.linewidth'] = 2.5
matplotlib.rcParams['lines.markersize'] = 8

ap = argparse.ArgumentParser()
ap.add_argument('--dataset', type=str, required=False, default='mnist')
ap.add_argument('--num-nodes', type=int, required=False, default=125)
ap.add_argument('--epochs', type=int, required=False)
ap.add_argument('--histories', type=str, nargs='+', required=True)
ap.add_argument('--baselines', type=str, nargs='+', required=True)
ap.add_argument('--labels', type=str, nargs='+', required=True)
ap.add_argument('--name', type=str, required=True)
ap.add_argument('--ncols', type=int, required=True)
ap.add_argument('--dpi', type=int, required=True)
ap.add_argument('--colors', type=str, nargs='+', required=False, default=[])
ap.add_argument('--fracs', type=float, nargs='+', required=False, default=[])
ap.add_argument('--accuracy', type=float, required=False)
args = vars(ap.parse_args())
args = Struct(**args)

fig = plt.figure(figsize=(15, 4))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

colors = ['k.-', 'r.:', 'm.:', 'b.:', 'g.:', 'c.:', 'y.:', 'k.:', 'r', 'b']
if len(args.colors):
    colors = args.colors


def get_milestone_epoch(mile_list, milestone):
    for idx, mile in enumerate(mile_list, 1):
        if mile > milestone:
            return idx


def calculate_num_euts(eut_schedule, mile):
    eut_schedule = [_ for _ in eut_schedule if _ <= mile]
    return len(eut_schedule), eut_schedule


milestones = {}
power = {}
delay = {}
cost = {}
c1, c2, c3 = 10**(-4), 10**(2), 0.5*10**(4)
for idx, history in enumerate(args.histories):
    aux = history[:-4] + '_aux.pkl'
    x_ax, y_ax, l_test, rounds, eps, eta_phi = pkl.load(
        open('../ckpts/{}_{}/history/{}'.format(
            args.dataset, args.num_nodes, history), 'rb'))
    train_args, eut_schedule = pkl.load(
        open('../ckpts/{}_{}/history/{}'.format(
            args.dataset, args.num_nodes, aux), 'rb'))

    nc = train_args.num_clusters[0]
    nw = train_args.num_workers
    cs = nw//nc
    e_glob, e_d2d = cfg.E_glob, cfg.E_glob*train_args.e_frac
    d_glob, d_d2d = cfg.D_glob, cfg.D_glob*train_args.d_frac
    alpha = 1600

    miles = get_milestone_epoch(y_ax, args.accuracy)
    tag = 'E_{}_D_{}'.format(train_args.e_frac, train_args.d_frac)
    milestones[tag] = miles
    num_eut, eut_schedule = calculate_num_euts(eut_schedule, miles)
    cost[tag] = sum([
        c1*(
            nc*e_glob + nw*sum(rounds[eut_schedule[i-1]:eut_schedule[i]])*e_d2d
        )/eut_schedule[i] +
        c2*(
            nc*d_glob + nw*sum(rounds[eut_schedule[i-1]:eut_schedule[i]])*d_d2d
        )/eut_schedule[i] +
        c3*(1-(eut_schedule[i-1]+alpha)/(
            eut_schedule[i-1]+eut_schedule[i]+alpha)
            ) for i in range(1, len(eut_schedule))
    ])
    power[tag] = (num_eut*nc*e_glob*d_glob) + \
        (nw*sum(rounds[:miles])*e_d2d*d_d2d)
    delay[tag] = (num_eut*d_glob) + (sum(rounds[:miles])*d_d2d)

x_ticks = []
k1, k2, k3 = 10**4, 10**4, 10**1
for i, e in enumerate(args.fracs):
    x_tick = 0.7 + i
    colors = ['m', 'b', 'g', 'c']
    for j, d in enumerate(args.fracs):
        if i == 0:
            ax1.bar(x_tick+0.2*j, cost['E_{}_D_{}'.format(e, d)]/k1,
                    width=0.2, color=colors[j],
                    label='$\Delta_{D2D}/\Delta_{Glob}$='+'{:.2}'.format(d),
                    )
        else:
            ax1.bar(x_tick+0.2*j, cost['E_{}_D_{}'.format(e, d)]/k1,
                    width=0.2, color=colors[j])
        ax2.bar(x_tick+0.2*j, power['E_{}_D_{}'.format(e, d)]/k2,
                width=0.2, color=colors[j])
        ax3.bar(x_tick+0.2*j, delay['E_{}_D_{}'.format(e, d)]/k3,
                width=0.2, color=colors[j])

for (idx, history), n in zip(enumerate(args.baselines), ['c', 'f']):
    x_ax, y_ax, l_test = history_parser(args.dataset, args.num_nodes, history)
    miles = get_milestone_epoch(y_ax, args.accuracy)
    milestones[n] = miles
    eut_schedule = list(range(1, 100))
    num_eut = len(eut_schedule)
    cost[n] = sum([
        c1*(nw*e_glob)/eut_schedule[i] +
        c2*(nw*d_glob)/eut_schedule[i] +
        c3*(1-(eut_schedule[i-1]+alpha)/(
            eut_schedule[i-1]+eut_schedule[i]+alpha)
            ) for i in range(1, len(eut_schedule))
    ])
    power[n] = miles*nw*e_glob*d_glob
    delay[n] = miles*d_glob
    break
ax1.hlines(y=cost['c']/k1, xmin=0.5, xmax=4.5,
           color='k', label=r'FL,$\tau$=1 (full)')
ax2.hlines(y=power['c']/k2, xmin=0.5, xmax=4.5, color='k')
ax3.hlines(y=delay['c']/k3, xmin=0.5, xmax=4.5, color='k')

if len(args.baselines) > 1:
    for (idx, history), n in zip(enumerate(args.baselines), ['c', 'f']):
        if idx == 0:
            continue
        x_ax, y_ax, l_test, rounds, eps, eta_phi = pkl.load(
            open('../ckpts/{}_{}/history/{}'.format(
                args.dataset, args.num_nodes, history), 'rb'))
        miles = get_milestone_epoch(y_ax, args.accuracy)
        milestones[n] = miles
        eut_schedule = list(range(0, 341, 20)) + [351]
        num_eut = len(eut_schedule)
        cost[n] = sum([
            c1*(nc*e_glob)/eut_schedule[i] +
            c2*(nc*d_glob)/eut_schedule[i] +
            c3*(1-(eut_schedule[i-1]+alpha)/(
                eut_schedule[i-1]+eut_schedule[i]+alpha)
                ) for i in range(1, len(eut_schedule))
        ])
        power[n] = num_eut*nc*e_glob*d_glob
        delay[n] = num_eut*d_glob

    ax1.hlines(y=cost['f']/k1, xmin=0.5, xmax=4.5,
               color='r', label=r'FL,$\tau$=20 (sampled)')
    ax2.hlines(y=power['f']/k2, xmin=0.5, xmax=4.5, color='r')
    ax3.hlines(y=delay['f']/k3, xmin=0.5, xmax=4.5, color='r')

print(milestones)
print(power)
print(delay)

ax1.xaxis.set_ticks([1, 2, 3, 4])
ax1.xaxis.set_ticklabels(args.fracs)
ax2.xaxis.set_ticks([1, 2, 3, 4])
ax2.xaxis.set_ticklabels(args.fracs)
ax3.xaxis.set_ticks([1, 2, 3, 4])
ax3.xaxis.set_ticklabels(args.fracs)

ax1.set_xlabel('$E_{D2D}/E_{Glob}$')
ax1.set_ylabel('total cost ($x 10^4$)')
ax1.set_yscale('log', basey=2)

ax2.set_xlabel('$E_{D2D}/E_{Glob}$')
ax2.set_ylabel('total power ($x 10^4$ J)')
ax2.set_yscale('log', basey=2)

ax3.set_xlabel('$E_{D2D}/E_{Glob}$')
ax3.set_ylabel('total delay ($x 10^1$ s)')
# ax3.set_yscale('log')

ax1.grid(b=True)
ax2.grid()
ax3.grid()

ax1.set_title('(a)', y=-0.4)
ax2.set_title('(b)', y=-0.4)
ax3.set_title('(c)', y=-0.4)

ax1.legend(loc='upper right', ncol=args.ncols,
           bbox_to_anchor=(-0.05, 1.1, 3.9, .25),
           mode='expand', frameon=False,
           # handlelength=0.7
           )

args.name = args.name.format(args.accuracy)
print('Saving: ', args.name)
fig.subplots_adjust(wspace=0.4)
plt.savefig('../ckpts/{}_{}/plots/{}'.format(
    args.dataset, args.num_nodes, args.name),
    bbox_inches='tight',
    dpi=args.dpi)
