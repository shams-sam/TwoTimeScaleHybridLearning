from argparser import argparser
from arguments import Arguments
from fcn import FCN
from distributor import get_fog_graph
import matplotlib.pyplot as plt
import numpy as np
from train import fog_test, fog_train as train, test
import os
import pickle as pkl
from svm import SVM
import syft as sy
import sys
import torch
from utils import get_testloader


# Setups
args = Arguments(argparser())
hook = sy.TorchHook(torch)
USE_CUDA = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
device = torch.device("cuda" if USE_CUDA else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if USE_CUDA else {}
kwargs = {}

ckpt_path = '../ckpts'
folder = '{}_{}'.format(args.dataset, args.num_workers)
model_name = 'clf_{}_paradigm_{}_uniform_{}_non_iid_{}' \
                '_num_workers_{}_lr_{}_decay_{}_batch_{}'.format(
                    args.clf, args.paradigm, args.uniform_data, args.non_iid,
                    args.num_workers, args.lr, args.decay,
                    args.batch_size)

if args.paradigm == 'fog':
    model_name += '_rounds_{}_radius_{}_d2d_{}_factor_{}' \
                     '_eut_{}_lut_{}'.format(
                         args.rounds, args.radius, args.d2d, args.factor,
                         args.eut_int, args.lut_int)

file_ = '{}/{}/logs/{}.log'.format(ckpt_path, folder, model_name)
print("Logging: ", file_)
if not args.dry_run:
    log_file = open(file_, 'w')
    std_out = sys.stdout
    sys.stdout = log_file

print('+'*80)
print(model_name)
print('+'*80)

print(args.__dict__)
print('+'*80)

init_path = '{}/{}/{}_{}.init'.format(ckpt_path, 'init',
                                      args.dataset, args.clf)
best_path = os.path.join(ckpt_path, folder, 'models',  model_name + '.best')
stop_path = os.path.join(ckpt_path, folder, 'models',  model_name + '.stop')

# prepare graph and data
fog_graph, workers = get_fog_graph(hook, args.num_workers,
                                   args.num_clusters,
                                   args.shuffle_workers,
                                   args.uniform_clusters)

data_file = '../ckpts/{}_{}/data/n_classes_per_node_{}_stratify_{}' \
            '_uniform_{}_repeat_{}.pkl'.format(
                args.dataset, args.num_workers, args.non_iid,
                args.stratify, args.uniform_data, args.repeat)

print('Loading data: {}'.format(data_file))
X_trains, X_tests, y_trains, y_tests, meta = pkl.load(
    open(data_file, 'rb'))

test_loader = get_testloader(args.dataset, args.test_batch_size)

print(fog_graph)
print('+'*80)

# Fire the engines
if args.clf == 'fcn':
    print('Initializing FCN...')
    model_class = FCN
elif args.clf == 'svm':
    print('Initializing SVM...')
    model_class = SVM

model = model_class(args.input_size, args.output_size).to(device)

model.load_state_dict(torch.load(init_path))
print('Load init: {}'.format(init_path))

loss_type = 'hinge' if args.clf == 'svm' else 'nll'
agg_type = 'laplacian' if args.paradigm == 'fog' else 'averaging'

print("Loss: {}\nAggregation: {}".format(loss_type, agg_type))

if args.batch_size == 0:
    args.batch_size = int(meta['batch_size'])
    print("Resetting batch size: {}...".format(args.batch_size))

print('+'*80)

best = 0

x_ax = []
y_ax = []
l_test = []
l_mean = []
l_std = []
y_mean = []
y_std = []

# non_zero_grad for getting a true_grad in fog_train
# grad is not used in fcn case
print('Pre-Training')
test(args, model, device, test_loader, best, 1, loss_type)

worker_models = {}
for epoch in range(1, args.epochs + 1):
    train(
        args, model, fog_graph, workers, X_trains, y_trains, device, epoch,
        loss_type, agg_type, args.rounds, args.graphs, args.d2d,
        args.factor, args.shuffle_worker_data, worker_models)
    acc, loss = test(args, model, device, test_loader, best, epoch, loss_type)
    loss_mean, loss_std, acc_mean, acc_std = fog_test(
        args, workers, X_tests, y_tests,
        device, epoch, loss_type, worker_models)
    y_ax.append(acc)
    x_ax.append(epoch)
    l_test.append(loss)
    l_mean.append(loss_mean)
    l_std.append(loss_std)
    y_mean.append(acc_mean)
    y_std.append(acc_std)

    if args.save_model and acc > best:
        best = acc
        torch.save(model.state_dict(), best_path)
        print('Model best  @ {}, acc {}: {}\n'.format(
            epoch, acc, best_path))
if (args.save_model):
    torch.save(model.state_dict(), stop_path)
    print('Model stop: {}'.format(stop_path))

hist_file = '{}/{}/history/{}.pkl'.format(ckpt_path, folder, model_name)
pkl.dump((x_ax, y_ax, l_test), open(hist_file, 'wb'))
print('Saved: ', hist_file)

l_mean = np.array(l_mean)
l_std = np.array(l_std)
y_mean = np.array(y_mean)
y_std = np.array(y_std)

fig = plt.figure(figsize=(5, 4))
ax1 = fig.add_subplot(111)
ax2 = plt.twinx()
l1 = ax1.plot(x_ax, y_ax, 'b.-.', label='accuracy')
l1_ = ax1.plot(x_ax, y_mean, 'b', label='acc mean')
ax1.plot(x_ax, y_mean+y_std, 'b:')
ax1.plot(x_ax, y_mean-y_std, 'b:')
ax1.set_ylabel('accuracy')
l2 = ax2.plot(x_ax, l_test, 'r.-.', label='{} loss'.format(loss_type))
l2_ = ax2.plot(x_ax, l_mean, 'r', label='{} mean'.format(loss_type))
ax2.plot(x_ax, l_mean+l_std, 'r:')
ax2.plot(x_ax, l_mean-l_std, 'r:')
ax2.set_ylabel('{} loss'.format(loss_type))
ax2.set_xlabel('epochs')
ls = l1+l1_+l2+l2_
lab = [_.get_label() for _ in ls]
ax1.legend(ls, lab, loc=7)
ax1.grid()
plt.xlim(left=0, right=args.epochs)
plot_file = '{}/{}/plots/{}.jpg'.format(ckpt_path, folder, model_name)
plt.savefig(plot_file, bbox_inches='tight', dpi=300)
print('Saved: ', plot_file)
if args.dry_run:
    print("Remove: ", plot_file)
    os.remove(plot_file)
    print("Remove: ", hist_file)
    os.remove(hist_file)

    
if not args.dry_run:
    log_file.close()
    sys.stdout = std_out
