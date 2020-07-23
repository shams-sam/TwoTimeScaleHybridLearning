from collections import OrderedDict
from distributor import get_connected_graph
from model_op import add_model_weights, get_model_weights, scale_model_weights
from networkx import laplacian_matrix
import numpy as np
import pickle as pkl
import torch


def averaging_consensus(cluster, models, weights):
    with torch.no_grad():
        weighted_models = [get_model_weights(models[_], weights[_])
                           for _ in cluster]
        model_sum = weighted_models[0]
        for m in weighted_models[1:]:
            model_sum = add_model_weights(model_sum, m)

    return model_sum


def laplacian_average(models, V, num_nodes, rounds):
    model = OrderedDict()
    idx = np.random.randint(0, num_nodes)
    for key, val in models[0].items():
        size = val.size()
        initial = torch.stack([_[key] for _ in models])
        final = torch.matmul(torch.matrix_power(V, rounds),
                             initial.reshape(num_nodes, -1))*num_nodes
        model[key] = final[idx].reshape(size)
        for id_ in range(len(models)):
            models[id_][key] = final[id_].reshape(size)

    return model, models


def consensus_matrix(num_nodes, radius, factor, topology):
    if type(radius) == str:
        graph = pkl.load(open('../graphs/{}'.format(radius), 'rb'))
    else:
        graph = get_connected_graph(num_nodes, radius, topology)
    max_deg = max(dict(graph.degree()).values())
    d = 1/(factor*max_deg)
    L = laplacian_matrix(graph).toarray()
    V = torch.Tensor(np.eye(num_nodes) - d*L)

    return V


# when consensus is done using d2d
# this gives closed form expression of such communication
def laplacian_consensus(cluster, nodes, models, weights, V, rounds):
    num_nodes = len(cluster)
    with torch.no_grad():
        weighted_models, ws = zip(*[
            [get_model_weights(models[_].get(), weights[_]), weights[_]]
            for _ in cluster])
        model_sum, laplace_models = laplacian_average(weighted_models, V,
                                                      num_nodes, rounds)
    ws = sum(ws)
    for _, model in zip(cluster, laplace_models):
        model = scale_model_weights(model, 1/ws)
        models[_].load_state_dict(model)
        models[_].send(nodes[_])

    return model_sum


def do_aggregation(epoch, interval):
    if interval == 0:
        return False
    if interval == 1:
        return True
    
    return (epoch % interval == 0)
