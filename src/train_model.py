from common.argparser import argparser
from common.arguments import Arguments
from common.utils import get_device, get_eut_schedule, get_model, \
    get_paths, get_testloader, init_logger
from data.distributor import get_fog_graph
from models.train import tthl_train, fl_train, test
import os
import pickle as pkl
import syft as sy
import sys
import torch
from viz.train_plots import train_plots


# Setups
args = Arguments(argparser())
device = get_device(args)
hook = sy.TorchHook(torch)
paths = get_paths(args)
log_file, std_out = init_logger(paths.log_file, args.dry_run)

print('+'*80)
print(paths.model_name)
print('+'*80)

print(args.__dict__)
print('+'*80)

# prepare graph and data
fog_graph, workers = get_fog_graph(hook, args.num_workers,
                                   args.num_clusters,
                                   args.shuffle_workers,
                                   args.uniform_clusters)

print('Loading data: {}'.format(paths.data_path))
X_trains, X_tests, y_trains, y_tests, meta = pkl.load(
    open(paths.data_path, 'rb'))

test_loader = get_testloader(args.dataset, args.test_batch_size)

print(fog_graph)
print('+'*80)

# Fire the engines
model, loss_type, agg_type = get_model(args)

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
best, _ = test(args, model, device, test_loader, best, 1, loss_type)
best_iter = -1
print('Acc: {:.3f}'.format(best))

if args.paradigm == 'hl':
    print('EUT Schedule')
    eut_schedule = get_eut_schedule(args)
    print(eut_schedule)

print('+'*80)
print('Training')
print('epoch \t tr loss (acc) (mean+-std) \t test loss (acc)')
worker_models = {}

gamma = 2/args.mu
Gamma = 2*np.pow()
alpha = max(args.beta*gamma/args.kappa,
            args.beta*gamma*(
                1-(args.kappa/4) + \
                np.pow(((1+(args.kappa/4))**2) + 2*args.omega, 0.5))
kwargs = {
    'alpha': max()
    'Gamma': 
}
for epoch in range(1, args.epochs + 1):
    if args.paradigm == 'fl':
        acc_mean, acc_std, loss_mean, loss_std = fl_train(
            args, model, fog_graph, workers, X_trains, y_trains,
            device, epoch, loss_type)
    elif args.paradigm == 'hl':
        acc_mean, acc_std, loss_mean, loss_std = tthl_train(
            args, model, fog_graph, workers, X_trains, y_trains, device, epoch,
            loss_type, agg_type, eut_schedule, worker_models)
    acc, loss = test(args, model, device, test_loader, best, epoch, loss_type)
    y_ax.append(acc)
    x_ax.append(epoch)
    l_test.append(loss)
    l_mean.append(loss_mean)
    l_std.append(loss_std)
    y_mean.append(acc_mean)
    y_std.append(acc_std)

    if epoch % args.log_intv == 0:
        print('{} \t {:.2f}+-{:.2f} ({:.2f}+-{:.2f}) \t {:.5f} ({:.3f})'.format(
            epoch, loss_mean, loss_std, acc_mean, acc_std, loss, acc))

    if args.save_model and acc > best:
        best = acc
        best_iter = epoch
        torch.save(model.state_dict(), paths.best_path)

if (args.save_model):
    print('\nModel best  @ {}, acc {}: {}'.format(
        best_iter, best, paths.best_path))
    torch.save(model.state_dict(), paths.stop_path)
    print('Model stop: {}'.format(paths.stop_path))
    

pkl.dump((x_ax, y_ax, l_test), open(paths.hist_path, 'wb'))
print('Saved: ', paths.hist_path)

plot_file = train_plots(args, x_ax, y_ax, l_test, l_mean, l_std,
                        y_mean, y_std, loss_type, paths.plot_path)

if args.dry_run:
    print("Remove: ", plot_file)
    os.remove(plot_file)
    print("Remove: ", paths.hist_path)
    os.remove(paths.hist_path)

    
if not args.dry_run:
    log_file.close()
    sys.stdout = std_out
