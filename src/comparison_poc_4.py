import argparse
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pickle as pkl
import common.config as cfg
from common.utils import Struct


matplotlib.rcParams.update({'font.size': 24})
matplotlib.rcParams['lines.linewidth'] = 2.5
matplotlib.rcParams['lines.markersize'] = 4

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

fig = plt.figure(figsize=(30, 7.5))
ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133, projection='3d')

colors = ['k.-', 'r.:', 'm.:', 'b.:', 'g.:', 'c.:', 'y.:', 'k.:', 'r', 'b']
if len(args.colors):
    colors = args.colors

def get_milestone_epoch(mile_list, milestone):
    for idx, mile in enumerate(mile_list, 1):
        if mile > milestone:
            return idx

def calculate_num_euts(eut_schedule, mile):
    return len([_ for _ in eut_schedule if _ <= mile])

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
    e_glob, e_d2d = cfg.E_glob, cfg.E_glob*train_args.e_frac
    d_glob, d_d2d = cfg.D_glob, cfg.D_glob*train_args.d_frac
    alpha = 1600

    miles = get_milestone_epoch(y_ax, args.accuracy)
    tag = 'E_{}_D_{}'.format(train_args.e_frac, train_args.d_frac)
    milestones[tag] = miles
    rounds = sum(rounds[:miles])*train_args.num_clusters[0]
    num_eut = calculate_num_euts(eut_schedule, miles)
    cost[tag] = c1*(num_eut*nc*e_glob + nw*rounds*e_d2d) + \
                c2*(num_eut*d_glob + rounds*d_d2d) + \
                sum([
                    c3*(1-(eut_schedule[i-1]+alpha)/(
                        eut_schedule[i-1]+eut_schedule[i]+alpha)
                    ) for i in range(1, len(eut_schedule))
                ])
    power[tag] = (num_eut*nc*e_glob*d_glob) + (nw*rounds*e_d2d*d_d2d)
    delay[tag] = (num_eut*d_glob) + (rounds*d_d2d)

for (idx, history), n in zip(enumerate(args.baselines), ('central')):
    x_ax, y_ax, l_test, rounds, eps, eta_phi, beta, mu = pkl.load(
        open('../ckpts/{}_{}/history/{}'.format(
            args.dataset, args.num_nodes, history), 'rb'))
    miles = get_milestone_epoch(y_ax, args.accuracy)
    milestones[n] = miles
    eut_schedule = list(range(1, 100))
    num_eut = len(eut_schedule)
    cost[n] = c1*(num_eut*nw*e_glob) + c2*(num_eut*d_glob) + sum([
        c3*(1-(eut_schedule[i-1]+alpha)/(
            eut_schedule[i-1]+eut_schedule[i]+alpha)
        ) for i in range(1, len(eut_schedule))
    ])
    power[n] = miles*nw*e_glob*d_glob
    delay[n] = miles*d_glob

print(milestones)
print(cost)
print(power)
# exit()
    
fracs = args.fracs
n = len(fracs)
power_mat = np.zeros((n, n))
delay_mat = np.zeros((n, n))
costs_mat = np.zeros((n, n))
for i, ie in enumerate(fracs):
    for j, jd in enumerate(fracs):
        tag = 'E_{}_D_{}'.format(ie, jd)
        power_mat[i,n-j-1] = (power[tag]-power['c'])*100/power['c']
        delay_mat[i,n-j-1] = (delay[tag]-delay['c'])*100/delay['c']
        costs_mat[i,n-j-1] = (cost[tag]-cost['c'])*100/cost['c']

column_names = list(map(str, fracs[::-1]))
row_names = list(map(str, fracs))
r, c = len(fracs), len(fracs)
xpos = np.arange(0, r, 1)
ypos = np.arange(0, c, 1)
xpos, ypos = np.meshgrid(xpos+0.25, ypos+0.25)
x, y = np.meshgrid(np.arange(0, r+1, 1),
                   np.arange(0, c+1, 1))

xpos = xpos.flatten()
ypos = ypos.flatten()
zpos = np.zeros(r*c)

dx = 0.5 * np.ones_like(zpos)
dy = dx.copy()
dz = costs_mat.flatten()/(10**4)
flat = np.ones((r+1, c+1))*milestones['c']
cs = ['m', 'b', 'g', 'c'] * c
ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color=cs)
ax1.w_xaxis.set_ticks([0.25, 1.25, 2.25, 3.25])
ax1.w_xaxis.set_ticklabels(column_names)
ax1.w_yaxis.set_ticks([0.25, 1.25, 2.25, 3.25])
ax1.w_yaxis.set_ticklabels(row_names)
ax1.set_xlabel('delay fraction', labelpad=25)
ax1.set_ylabel('energy fraction', labelpad=25)
ax1.set_zlabel('% increase in cost', labelpad=10)

k=(1)
dz = power_mat.flatten()/k
flat = np.ones((r+1, c+1))*power['c']/k
ax2.bar3d(xpos, ypos, zpos, dx, dy, dz, color=cs)
ax2.w_xaxis.set_ticks([0.25, 1.25, 2.25, 3.25])
ax2.w_xaxis.set_ticklabels(column_names)
ax2.w_yaxis.set_ticks([0.25, 1.25, 2.25, 3.25])
ax2.w_yaxis.set_ticklabels(row_names)
ax2.set_xlabel('delay fraction', labelpad=25)
ax2.set_ylabel('energy fraction', labelpad=25)
ax2.set_zlabel('% increase in power', labelpad=10)

k=1
dz = delay_mat.flatten()/k
flat = np.ones((r+1,c+1))*delay['c']/k
ax3.bar3d(xpos, ypos, zpos, dx, dy, dz, color=cs)
ax3.w_xaxis.set_ticks([0.25, 1.25, 2.25, 3.25])
ax3.w_xaxis.set_ticklabels(column_names)
ax3.w_yaxis.set_ticks([0.25, 1.25, 2.25, 3.25])
ax3.w_yaxis.set_ticklabels(row_names)
ax3.set_xlabel('delay fraction', labelpad=25)
ax3.set_ylabel('energy fraction', labelpad=25)
ax3.set_zlabel('% increase in delay', labelpad=10)


ax1.set_title('(a)', y=-0.2)
ax2.set_title('(b)', y=-0.2)
ax3.set_title('(c)', y=-0.2)

args.name = args.name.format(args.accuracy)
print('Saving: ', args.name)
fig.subplots_adjust(wspace=0.025)
plt.savefig('../ckpts/{}_{}/plots/{}'.format(
    args.dataset, args.num_nodes, args.name),
            bbox_inches='tight',
            pad_inches=0.5, dpi=args.dpi)
