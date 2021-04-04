import common.config as cfg
from math import factorial as f
from models.fcn import FCN
from models.svm import SVM
import networkx as nx
import numpy as np
import os
import pickle as pkl
from random import random
import sys
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms


def booltype(arg):
    return bool(int(arg))


def decimal_format(num, places=4):
    return round(num, places)


def eut_add(eut_range):
    return eut_range[0] \
        if len(eut_range)==1 \
        else np.random.randint(
                eut_range[0], eut_range[-1])


def flip(p):
    return True if random() < p else False


def get_average_degree(graph):
    return sum(dict(graph.degree()).values())/len(graph)


def get_dataloader(data, targets, batchsize, shuffle=True):
    dataset = TensorDataset(data, targets)

    return DataLoader(dataset, batch_size=batchsize,
                      shuffle=shuffle, num_workers=1)


def get_device(args):
    USE_CUDA = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    return torch.device("cuda" if USE_CUDA else "cpu")


def get_laplacian(graph):
    return nx.laplacian_matrix(graph).toarray()


def get_max_degree(graph):
    return max(dict(graph.degree()).values())


def get_model(args):
    if args.clf == 'fcn':
        print('Initializing FCN...')
        model_class = FCN
    elif args.clf == 'svm':
        print('Initializing SVM...')
        model_class = SVM

    device = get_device(args)
    model = model_class(args.input_size, args.output_size).to(device)

    paths = get_paths(args)
    model.load_state_dict(torch.load(paths.init_path))
    print('Load init: {}'.format(paths.init_path))

    loss_type = 'hinge' if args.clf == 'svm' else 'nll'
    agg_type = 'laplacian' if args.paradigm == 'hl' else 'averaging'
    print("Loss: {}\nAggregation: {}".format(loss_type, agg_type))

    return model, loss_type, agg_type


def get_data_path(ckpt_path, args):
    return '{}/{}_{}/data/n_classes_per_node_{}_stratify_{}' \
            '_uniform_{}_repeat_{}.pkl'.format(
                ckpt_path, args.dataset, args.num_workers, args.non_iid,
                args.stratify, args.uniform_data, args.repeat)


def get_eut_schedule(args):
    if not args.eut_range:
        return list(range(1, args.epochs+1))

    if args.tau_max:
        return [min(args.eut_range), args.epochs]

    eut_schedule = [0]
    np.random.seed(args.eut_seed)
    add = eut_add(args.eut_range)

    while eut_schedule[-1] + add < args.epochs:
        eut_schedule.append(eut_schedule[-1] + add)
        add = eut_add(args.eut_range)

    return eut_schedule[1:] + [args.epochs]


def get_lut_schedule(args):
    if not args.lut_intv:
        return []
    
    lut_schedule = [0]
    while lut_schedule[-1] + args.lut_intv < args.epochs:
        lut_schedule.append(lut_schedule[-1] + args.lut_intv)

    return lut_schedule[1:]

def get_paths(args):
    ckpt_path = cfg.ckpt_path
    folder = '{}_{}'.format(args.dataset, args.num_workers)
    if args.dry_run:
        model_name = 'debug'
    else:
        model_name = 'clf_{}_paradigm_{}_uniform_{}_non_iid_{}' \
                     '_num_workers_{}_lr_{}_decay_{}_batch_{}'.format(
                         args.clf, args.paradigm, args.uniform_data, args.non_iid,
                         args.num_workers, args.lr, args.decay,
                         args.batch_size)

    if args.paradigm == 'hl':
        if args.lut_intv:
            model_name += '_eut_{}_lut_{}_rounds_{}'.format(
                args.eut_range[0], args.lut_intv, args.rounds)
        else:
            model_name += '_delta_{}_zeta_{}_beta_{}_mu_{}_phi_{}_factor_{}'.format(
                             args.delta, args.zeta, args.beta, args.mu,
                             args.phi, args.factor)
    if args.tau_max:
        model_name += '_T1_{}_Tmax_{}_E_{}_D_{}'.format(
            min(args.eut_range), args.tau_max, args.e_frac, args.d_frac)
    elif args.eut_range and not args.lut_intv:
        model_name += '_eut_range_{}'.format('_'.join(map(str, args.eut_range)))

    if args.cs:
        model_name += '_cs_{}'.format('_'.join(map(str, args.cs)))

    paths = {}
    paths['model_name'] = model_name
    paths['log_file'] = '{}/{}/logs/{}.log'.format(
        ckpt_path, folder, model_name)
    paths['init_path'] = '{}/{}/{}_{}.init'.format(
        ckpt_path, 'init', args.dataset, args.clf)
    paths['best_path'] = os.path.join(
        ckpt_path, folder, 'models',  model_name + '.best')
    paths['stop_path'] = os.path.join(
        ckpt_path, folder, 'models',  model_name + '.stop')
    paths['data_path'] = get_data_path(ckpt_path, args)
    paths['plot_path'] = '{}/{}/plots/{}.jpg'.format(
        ckpt_path, folder, model_name)
    paths['hist_path'] = '{}/{}/history/{}.pkl'.format(
        ckpt_path, folder, model_name)
    paths['aux_path'] = '{}/{}/history/{}_aux.pkl'.format(
        ckpt_path, folder, model_name)

    return Struct(**paths)
    

def get_rho(graph, num_nodes, factor):
    max_d = get_max_degree(graph)
    d = 1/(factor*max_d)
    L = get_laplacian(graph)
    V = np.eye(num_nodes) - d*L
    Z = V-(1/num_nodes)
    return get_spectral_radius(Z)


def get_spectral_radius(matrix):
    eig, _ = np.linalg.eig(matrix)

    return max(eig)


def get_testloader(dataset, batch_size, shuffle=True):
    kwargs = {}
    if dataset == 'mnist':
        return torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=batch_size, shuffle=shuffle, **kwargs)
    elif dataset == 'cifar':
        return torch.utils.data.DataLoader(
            datasets.CIFAR10('../data', train=False,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5),
                                                      (0.5, 0.5, 0.5))])),
            batch_size=batch_size, shuffle=shuffle, **kwargs)
    elif dataset == 'fmnist':
        return torch.utils.data.DataLoader(
            datasets.FashionMNIST('../data', train=False,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.2861,),
                                                           (0.3530,))])),
            batch_size=batch_size, shuffle=shuffle, **kwargs)


def get_trainloader(dataset, batch_size, shuffle=True):
    kwargs = {}
    if dataset == 'mnist':
        return torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=batch_size, shuffle=shuffle, **kwargs)
    elif dataset == 'cifar':
        return torch.utils.data.DataLoader(
            datasets.CIFAR10('../data', train=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5),
                                                      (0.5, 0.5, 0.5))])),
            batch_size=batch_size, shuffle=shuffle, **kwargs)
    elif dataset == 'fmnist':
        return torch.utils.data.DataLoader(
            datasets.FashionMNIST('../data', train=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.2861,),
                                                           (0.3530,))])),
            batch_size=batch_size, shuffle=shuffle, **kwargs)


def history_parser(dataset, num_nodes, history):
    h = pkl.load(
        open('../ckpts/{}_{}/history/{}'.format(
            dataset, num_nodes, history), 'rb'))
    if len(h) == 8:
        x_ax, y_ax, l_test, rounds, eps, eta_phi, beta, mu = h
    else:
        x_ax, y_ax, l_test, rounds, eps, eta_phi = h

    return x_ax, y_ax, l_test

def in_range(elem, upper, lower):
    return (elem >= lower) and (elem <= upper)


def init_logger(log_file, dry_run=False):
    print("Logging: ", log_file)
    std_out = sys.stdout
    if not dry_run:
        log_file = open(log_file, 'w')
        sys.stdout = log_file

    return log_file, std_out


def nCr(n, r):
    return f(n)//f(r)//f(n-r)


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)
