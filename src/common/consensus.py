from collections import OrderedDict
import common.config as cfg
from models.model_op import get_tensor_sum, get_model_weights
from networkx import laplacian_matrix
import numpy as np
import pickle as pkl
from sklearn.linear_model import LinearRegression
from terminaltables import AsciiTable
import torch


def approx_eps(data, x_max):
    xs, ys = zip(*data)
    xs, ys = np.array(xs).reshape(-1, 1), np.array(ys)
    x_new = np.array(range(1, x_max+1)).reshape(-1, 1)
    reg = LinearRegression().fit(xs, ys)
    y_new = reg.predict(x_new)
    y_new = y_new.clip(min=cfg.F['eps_min'])

    return y_new


def approx_rounds(eps, sc, lamda, phi, kwargs, t_min, t_max):
    eta = kwargs.gamma/(np.array(range(t_min, t_max+1))+kwargs.alpha)
    rounds = {}
    for a, eps_a in eps.items():
        eps_a = eps_a.clip(min=cfg.F['eps_min'])
        r = np.ceil(np.log(eta*phi/(sc[a]*eps_a))/np.log(lamda[a]))
        r[r < 0] = 0
        rounds[a] = np.ceil(r*cfg.F['gamma_pred'])

    return rounds


def averaging_consensus(cluster, models, weights):
    with torch.no_grad():
        weighted_models = [get_model_weights(models[_], weights[idx])
                           for idx, _ in enumerate(cluster)]
        model_sum = get_tensor_sum(weighted_models)

    return model_sum


def consensus_matrix(num_nodes, graph, factor, topology):
    graph = pkl.load(open('../graphs/{}'.format(graph), 'rb'))
    max_deg = max(dict(graph.degree()).values())
    d = 1/(factor*max_deg)
    L = laplacian_matrix(graph).toarray()
    V = torch.Tensor(np.eye(num_nodes) - d*L)

    return V


def do_sync(epoch, schedule):
    return epoch in schedule


def estimate_alpha(args, kwargs):
    mu_over_beta = args.mu/(4*args.beta)
    # calculate
    alpha = 2 * max(
        1,
        (args.beta**2)*kwargs.gamma/args.mu,
        args.beta*kwargs.gamma*(
            mu_over_beta-1+np.sqrt(
                ((1+mu_over_beta)**2)+2*args.omega
            )
        )
    )

    while True:
        Z1 = estimate_Z1(alpha, args, kwargs)
        lhs = (1/(args.beta*kwargs.gamma))*np.sqrt(
            alpha*(
                (args.mu*kwargs.gamma-1+(1/(1+alpha)))/Z1
            )
        )
        if lhs >= args.omega:
            return alpha
        else:
            alpha = 2*alpha


def estimate_delta(zeta, grads, agg_grad, weight):
    delta = 0
    count = 0
    for g, ag, w in zip(grads, agg_grad, weight):
        delta += torch.abs(torch.norm(g-ag)-zeta*torch.norm(w.flatten()))
        count += 1

    return delta.item()/count


def estimate_nu(args, kwargs):
    while True:
        nu = args.xi*(args.epochs + kwargs.alpha)
        Z1 = estimate_Z1(kwargs.alpha, args, kwargs)
        Z2 = estimate_Z2(kwargs.alpha, args, kwargs)
        omega_max = estimate_omega_max(args, kwargs)
        lhs = Z2*max(
            ((args.beta**2)*(kwargs.gamma**2)/(args.mu*kwargs.gamma-1)),
            kwargs.alpha/(Z1*((omega_max**2)-(args.omega**2)))
            # grad at F(w^(0))
        )
        # print(args.xi, lhs, nu, Z1, Z2)
        if lhs <= nu:
            return nu
        args.xi = 2*args.xi


def estimate_omega_max(args, kwargs):
    Z1 = estimate_Z1(kwargs.alpha, args, kwargs)
    return (1/(args.beta*kwargs.gamma))*np.sqrt(
        kwargs.alpha*(
            (args.mu*kwargs.gamma-1+(1/(1+kwargs.alpha)))/Z1
        )
    )


def estimate_phi(sigma, args, kwargs):
    Z1 = estimate_Z1(kwargs.alpha, args, kwargs)
    Z2 = estimate_Z2(sigma, args, kwargs)
    omega_max = estimate_omega_max(args, kwargs)

    numerator = args.beta*kwargs.nu/max(
        ((args.beta**2)*(kwargs.gamma**2)/(args.mu*kwargs.gamma-1)),
        kwargs.alpha/(Z1*((omega_max**2)-(args.omega**2)))
    ) - args.beta*Z2
    denominator = 1+(50*args.beta*kwargs.gamma)*(args.tau_max-1)*(
        1+(args.tau_max-2)/(kwargs.alpha+1)
    )*((
        1+((args.tau_max-1)/(kwargs.alpha-1))
    )**(6*args.beta*kwargs.gamma))

    return np.sqrt(numerator/denominator)*cfg.F['phi']


def estimate_Z1(alpha, args, kwargs):
    return 32*(
        args.beta**2*kwargs.gamma/args.mu
    )*(args.tau_max-1)*((
        1 + (args.tau_max/(alpha-1))
    )**2)*((1+(args.tau_max-1)/(alpha-1))**(6*args.beta*kwargs.gamma))


def estimate_Z2(alpha, args, kwargs):
    return ((args.sigma**2)/(2*args.beta))+50*args.beta*kwargs.gamma*(
        args.tau_max-1
    )*(1+((args.tau_max-2)/(alpha+1)))*((
        1+((args.tau_max-1)/(alpha-1))
    )**(6*args.beta*kwargs.gamma))*(
        ((args.sigma**2)/args.beta)+((args.delta**2)/args.beta)
    )


def get_cluster_eps(cluster, models, nodes):
    if not len(models):
        return 0

    cluster_norms = []  # norm infinity: max(abs(w))
    for _ in cluster:
        model = models[_].get()
        weights = [val.flatten()
                   for full_name, val in model.state_dict().items()]
        weight = torch.cat(weights, dim=0)
        norm = torch.norm(weight, float('inf')).item()
        cluster_norms.append(norm)
        models[_] = model.copy().send(nodes[_])
    cluster_norms = np.array(cluster_norms)
    cluster_norms = cluster_norms
    eps = cluster_norms.max()-cluster_norms.min()

    return eps


def get_node_weights(graph, nodes, aggregators, num_workers):
    node_weights = {}
    for a in aggregators:
        children = graph[a]
        for child in children:
            node_weights[child] = len(children)/num_workers

    return node_weights


def get_sigma(num_nodes, eps, factor):
    return factor*(num_nodes**2)*(eps**2)


def get_spectral_radius(matrix):
    eig, _ = torch.eig(matrix)
    return torch.max(eig).item()


def laplacian_average(models, V, num_nodes, rounds):
    model = OrderedDict()
    idx = np.random.randint(0, num_nodes)
    for key, val in models[0].items():
        size = val.size()
        initial = torch.stack([_[key] for _ in models])
        final = torch.matmul(torch.matrix_power(V, rounds),
                             initial.reshape(num_nodes, -1))# *num_nodes
        model[key] = final[idx].reshape(size)
        for id_ in range(len(models)):
            models[id_][key] = final[id_].reshape(size)

    return model, models


# when consensus is done using d2d
# this gives closed form expression of such communication
def laplacian_consensus(cluster, nodes, models, V, rounds):
    num_nodes = len(cluster)
    with torch.no_grad():
        weighted_models, ws = zip(*[
            [get_model_weights(models[_].get(), 1), 1]
            for _ in cluster])
        _, laplace_models = laplacian_average(weighted_models, V,
                                              num_nodes, rounds)
    for idx, _ in enumerate(cluster):
        models[_].load_state_dict(laplace_models[idx])


def optimize_tau(rounds, args, alpha, t):
    e_glob, d_glob = cfg.E_glob, cfg.D_glob
    e_d2d, d_d2d = args.e_frac*e_glob, args.d_frac*d_glob
    nc = args.num_clusters[0]  # num cluster
    cs = args.num_workers/nc  # cluster size

    # importance factors
    if args.cs:
        c1, c2, c3 = args.cs
    else:
        # c1, c2, c3 = 10**(-4), 10**(2), 0.5*10**(4)  # for svms
        c1, c2, c3 = 10**(-8), 10**(2), 0.5*10**(1)

    optimal_tau = 0
    optimal_cost = float('inf')
    data = [[
        'rounds', 'e_glob', 'e_d2d', 'a', 'b', 'c', 'cost', 'tau', 't' 
    ]]
    for tau in range(1, args.tau_max+1):
        gamma_t = sum([sum(r[:tau])
                       for _, r in rounds.items()])
        print((nc*e_glob + nc*cs*gamma_t*e_d2d), tau, (d_glob + gamma_t*d_d2d))
        a = c1 * (nc*e_glob + nc*cs*gamma_t*e_d2d)/tau
        b = c2 * (d_glob + gamma_t*d_d2d)/tau
        c = c3 * (1-((t+alpha)/(t+tau+alpha)))
        cost = a + b + c
        data.append([gamma_t, nc*e_glob, nc*cs*gamma_t*e_d2d, '{:.4f}'.format(a),
                     '{:.4f}'.format(b), '{:.4f}'.format(c),
                     '{:.4f}'.format(cost), tau, t])
        if cost < optimal_cost:
            optimal_tau = tau
            optimal_cost = cost
    table = AsciiTable(data)
    table.title = 'tau-optim'
    print(table.table)

    return t+optimal_tau
