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


def estimate_rounds(sigma, num_nodes_in_cluster, eps, lamda):
    return (np.log2(sigma)-2*np.log2(
        num_nodes_in_cluster*eps))/(2*np.log2(lamda))


def estimate_delta(rho, fog_graph, grads):
    clusters = [_ for _ in fog_graph if 'L1' in _]

    num = 0
    den = 0

    for cluster in clusters:
        cluster_grad = 0
        rho_c = 0
        for i in cluster:
            rho_c += rho[i]
            cluster_grad += rho[i]*grad[i].view(-1)
        num += rho_c * (torch.norm(cluster_grad).item()**2)
        den += rho_c * cluster_grad
    den = torch.norm(den).item()**2

    return num/den


def get_cluster_eps(cluster, models, weights,
                    nodes, param='weight'):
    cluster_norms = []
    for _ in cluster:
        model = models[_].get()
        weights = [val.flatten()
                   for full_name, val in model.state_dict().items()]
        weight = torch.cat(weights, dim=0)
        norm = torch.norm(weight).item()
        cluster_norms.append(norm)
        models[_] = model.copy().send(nodes[_])
    cluster_norms = np.array(cluster_norms)
    cluster_norms = cluster_norms
    eps = cluster_norms.max()-cluster_norms.min()

    return eps


def get_sigma(num_nodes, eps, factor):
    return factor*(num_nodes**2)*(eps**2)


def get_spectral_radius(matrix):
    eig, _ = torch.eig(matrix)
    return torch.max(eig).item()


# when consensus is done using d2d
# this gives closed form expression of such communication
def laplacian_consensus(cluster, nodes, models, weights, V, rounds, sigma_mul):
    num_nodes = len(cluster)

    assert rounds != sigma_mul
    if not rounds:
        eps = get_cluster_eps(cluster, models, weights, nodes)
        sigma = get_sigma(num_nodes, eps, sigma_mul)
        lamda = get_spectral_radius(V - (1/num_nodes))
        rounds = int(np.ceil(estimate_rounds(sigma, num_nodes, eps, lamda)))
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

    if sigma_mul:
        return model_sum, [eps, sigma, lamda, rounds]

    return model_sum, False


def do_aggregation(epoch, interval):
    if interval == 0:
        return False
    if interval == 1:
        return True
    
    return (epoch % interval == 0)
