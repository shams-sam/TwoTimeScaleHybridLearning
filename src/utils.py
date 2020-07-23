from math import factorial as f
import networkx as nx
import numpy as np
from random import random
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms


def booltype(arg):
    return bool(int(arg))


def decimal_format(num, places=4):
    return round(num, places)


def flip(p):
    return True if random() < p else False


def get_average_degree(graph):
    return sum(dict(graph.degree()).values())/len(graph)


def get_dataloader(data, targets, batchsize, shuffle=False):
    dataset = TensorDataset(data, targets)

    return DataLoader(dataset, batch_size=batchsize,
                      shuffle=shuffle, num_workers=1)


def get_laplacian(graph):
    return nx.laplacian_matrix(graph).toarray()


def get_max_degree(graph):
    return max(dict(graph.degree()).values())


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


def in_range(elem, upper, lower):
    return (elem >= lower) and (elem <= upper)


def nCr(n, r):
    return f(n)//f(r)//f(n-r)


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)
