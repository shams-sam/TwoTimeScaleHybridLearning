from collections import defaultdict
from common.argparser import argparser
from common.arguments import Arguments
from common.consensus import approx_eps, approx_rounds, estimate_alpha, \
    estimate_nu, estimate_omega_max, estimate_phi, optimize_tau
from common.utils import get_device, get_eut_schedule, get_lut_schedule, \
    get_model, get_paths, get_testloader, init_logger, Struct
from data.distributor import get_fog_graph
from models.model_op import get_num_params
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
h_rounds = []
h_eps = []
h_eta_phi = []

# non_zero_grad for getting a true_grad in fog_train
# grad is not used in fcn case
print('Pre-Training')
best, _ = test(args, model, device, test_loader, best, 1, loss_type)
best_iter = -1
print('Acc: {:.3f}'.format(best))

print('EUT Schedule')
eut_schedule = get_eut_schedule(args)
lut_schedule = get_lut_schedule(args)
print('EUT: ', eut_schedule)
print('LUT: ', lut_schedule)
print('Rounds: ', args.rounds)


print('+'*80)
print('Training')
print('epoch \t tr loss (acc) (mean+-std) \t test loss (acc) \t EUT')
worker_models = {}

# not calculated if lut given manually (used for proof of concept)
kwargs = {}
if args.paradigm == 'hl' and not args.lut_intv:
    kwargs['gamma'] = 2/args.mu
    kwargs['alpha'] = estimate_alpha(args, Struct(**kwargs))
    kwargs['omega_max'] = estimate_omega_max(args, Struct(**kwargs))
    kwargs['omega'] = args.omega
    kwargs['nu'] = estimate_nu(args, Struct(**kwargs))
    kwargs['phi'] = args.phi if args.phi \
        else estimate_phi(kwargs['alpha'], args, Struct(**kwargs))
    kwargs['M'] = get_num_params(model)
kwargs = Struct(**kwargs)
print('+'*80)
print(kwargs.__dict__)
print('+'*80)

aggregate_eps = defaultdict(list)
aggregate_rounds = {}
aggregate_sc = {}
aggregate_lamda = {}
extension = 0

epoch = 1
# for epoch in range(1, args.epochs + 1):
while epoch <= args.epochs:
    if args.paradigm == 'fl':
        worker_models, acc_mean, acc_std, \
            loss_mean, loss_std = fl_train(
                args, model, fog_graph, workers, X_trains, y_trains,
                device, epoch, eut_schedule, loss_type,
                worker_models)
        eut = 'N/A'
        avg_rounds = -1
        avg_eps = -2
        avg_eta_phi = -3
    elif args.paradigm == 'hl':
        worker_models, acc_mean, acc_std, loss_mean, loss_std, \
            avg_rounds, avg_eps, avg_eta_phi, \
            aggregate_eps, aggregate_rounds, aggregate_sc, aggregate_lamda, \
            eut = tthl_train(
                args, model, fog_graph, workers, X_trains, y_trains, device,
                epoch, loss_type, agg_type, eut_schedule, lut_schedule,
                worker_models, aggregate_eps, aggregate_rounds,
                aggregate_sc, aggregate_lamda, kwargs)

        if eut and args.tau_max:
            kwargs.alpha = estimate_alpha(args, kwargs)
            kwargs.phi = args.phi if args.phi \
                else estimate_phi(kwargs.alpha, args, kwargs)
            print('+'*80)
            print('delta: {:.4f}\nsigma: {:.4f}\nalpha: {:.4f}\nphi: {:.4f}'.format(
                args.delta, args.sigma, kwargs.alpha, kwargs.phi))
            print('+'*80)

            # calculate the next Tau
            predicted_eps = {}
            predicted_rounds = {}
            aggregators = [_ for _ in fog_graph.keys() if 'L1' in _]
            for a in aggregators:
                agg_eps = '{}_c'.format(a) \
                          if aggregate_rounds[a] else '{}_nc'.format(a)
                agg_eps = aggregate_eps[agg_eps]
                predicted_eps[a] = approx_eps(
                    agg_eps, args.tau_max)
            predicted_rounds = approx_rounds(
                predicted_eps, aggregate_sc, aggregate_lamda, kwargs.phi,
                kwargs, epoch+1, epoch+args.tau_max)
            # print(predicted_eps, predicted_rounds)
            optimal_tau = optimize_tau(
                predicted_rounds, args, kwargs.alpha, epoch)
            eut_schedule.append(int(optimal_tau))
            eut_schedule = sorted(eut_schedule)
            print(eut_schedule)
        else:
            assert args.eut_range

    acc, loss = test(args, model, device, test_loader, best, epoch, loss_type)
    y_ax.append(acc)
    x_ax.append(epoch)
    l_test.append(loss)
    l_mean.append(loss_mean)
    l_std.append(loss_std)
    y_mean.append(acc_mean)
    y_std.append(acc_std)
    h_rounds.append(avg_rounds)
    h_eps.append(avg_eps)
    h_eta_phi.append(avg_eta_phi)

    if epoch % args.log_intv == 0:
        print(
            '{} \t {:.2f}+-{:.2f} ({:.2f}+-{:.2f}) \t {:.5f} ({:.4f}) \t {}'
            .format(
                  epoch, loss_mean, loss_std,
                  acc_mean, acc_std, loss, acc, eut))

    if args.save_model and acc > best:
        best = acc
        best_iter = epoch
        torch.save(model.state_dict(), paths.best_path)

    if epoch == args.epochs and not args.dry_run and best < args.accuracy:
        args.epochs += 10
        extension += 1
        print('+'*80)
        print('best: {}\nneeded: {} \npatience: {} \next: {}'.format(
            best, args.accuracy, args.patience, extension))
        print('+'*80)
    epoch += 1

    if extension > args.patience:
        break

if (args.save_model):
    print('\nModel best  @ {}, acc {:.4f}: {}'.format(
        best_iter, best, paths.best_path))
    torch.save(model.state_dict(), paths.stop_path)
    print('Model stop: {}'.format(paths.stop_path))

pkl.dump((x_ax, y_ax, l_test, h_rounds, h_eps, h_eta_phi),
         open(paths.hist_path, 'wb'))
print('Saved: ', paths.hist_path)
if args.paradigm == 'hl':
    pkl.dump((args, eut_schedule), open(paths.aux_path, 'wb'))

plot_file = train_plots(args, x_ax, y_ax, l_test, l_mean, l_std,
                        y_mean, y_std, loss_type, paths.plot_path)

if args.dry_run:
    print("Remove: ", plot_file)
    os.remove(plot_file)
    print("Remove: ", paths.hist_path)
    os.remove(paths.hist_path)
    if args.paradigm == 'hl':
        print("Remove: ", paths.aux_path)
        os.remove(paths.aux_path)

if not args.dry_run:
    log_file.close()
    sys.stdout = std_out
