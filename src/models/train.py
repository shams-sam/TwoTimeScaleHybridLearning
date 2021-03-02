import common.config as cfg
from common.consensus import averaging_consensus, consensus_matrix, \
    estimate_delta, get_cluster_eps, get_node_weights, get_spectral_radius, \
    do_sync, laplacian_consensus
from common.utils import get_dataloader
from models.model_op import get_tensor_sum, \
    calc_beta, calc_mu, calc_sigma, \
    get_flattened_grads, get_flattened_weights, get_model_weights
from models.multi_class_hinge_loss import multiClassHingeLoss
import numpy as np
from terminaltables import AsciiTable
import torch
import torch.nn.functional as F
import torch.optim as optim


def train_step(model, data, target, loss_fn, args, device):
    worker_optim = optim.SGD(
        params=model.parameters(),
        lr=args.lr, weight_decay=args.decay)
    dataloader = get_dataloader(data, target, args.batch_size)

    w_correct = 0
    w_loss = 0
    grads = []
    for idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        worker_optim.zero_grad()
        output = model(data)
        pred = output.argmax(1, keepdim=True)
        w_correct += pred.eq(target.view_as(pred)).sum().item()
        loss = loss_fn(output, target)
        w_loss += loss.item()
        loss.backward()
        # train for only 2 minibatch
        # there is no concept of global epoch
        # but only local update cycle during each minibatch
        # 2 minibatches are used for gradient estimation
        grads.append(get_flattened_grads(model))
        if idx == 1:
            assert len(grads) == 2
            break
    sigma = calc_sigma(grads)
    worker_optim.step()

    # if removing break in minibatch
    # normalize loss with the number of len(dataloader)
    # normalize correct with len(dataloader.dataset)
    return w_loss, w_correct/((idx+1)*args.batch_size), sigma, grads[-1]


def test_step(model, data, target, loss_fn, args, device):
    dataloader = get_dataloader(data, target, args.test_batch_size)

    w_correct = 0
    w_loss = 0
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(1, keepdim=True)
        w_correct += pred.eq(target.view_as(pred)).sum().item()
        loss = loss_fn(output, target)
        w_loss += loss.item()
        loss.backward()

    return w_loss/len(dataloader), w_correct/len(dataloader.dataset)


def get_loss_fn(loss_fn):
    if loss_fn == 'nll':
        loss_fn_ = F.nll_loss
    elif loss_fn == 'hinge':
        loss_fn_ = multiClassHingeLoss()

    return loss_fn_


def send_to_worker(nodes, X_trains, y_trains):
    workers = [_ for _ in nodes.keys() if 'L0' in _]
    worker_data = {}
    worker_targets = {}
    worker_num_samples = {}

    for w, x, y in zip(workers, X_trains, y_trains):
        worker_data[w] = x.send(nodes[w])
        worker_targets[w] = y.send(nodes[w])
        worker_num_samples[w] = x.shape[0]

    return workers, worker_data, worker_targets, \
        worker_num_samples


def tthl_train(args, model, fog_graph, nodes, X_trains, y_trains,
               device, epoch, loss_fn, consensus, eut_schedule, lut_schedule,
               worker_models, aggregate_eps, aggregate_rounds,
               aggregate_sc, aggregate_lamda, kwargs):

    model.train()
    loss_fn_ = get_loss_fn(loss_fn)
    worker_losses, worker_accs, worker_sigmas, worker_grads = {}, {}, {}, {}

    # send data, model to workers
    # setup optimizer for each worker
    workers, worker_data, worker_targets, \
        _ = send_to_worker(
            nodes, X_trains, y_trains)
    aggregators = [_ for _ in nodes.keys() if 'L1' in _]
    weight_nodes = get_node_weights(fog_graph, nodes, aggregators, len(workers))

    eta = args.lr if args.lr else kwargs.gamma/(epoch+kwargs.alpha)
    args.lr = eta
    downlink = do_sync(epoch-1, eut_schedule) or epoch == 1

    # local descent
    for w in workers:
        if downlink:
            worker_models[w] = model.copy().send(nodes[w])
        node_model = worker_models[w].get()
        data = worker_data[w].get()
        target = worker_targets[w].get()
        w_loss, w_acc, w_sigma, w_grad = train_step(
            node_model, data, target, loss_fn_, args, device)
        worker_models[w] = node_model.send(nodes[w])
        worker_losses[w] = w_loss
        worker_accs[w] = w_acc
        worker_sigmas[w] = w_sigma
        worker_grads[w] = w_grad

    log_list = []
    log_head = ['W', 'lamda', 'eps', 'eta', 'rounds']
    log_list.append(log_head)

    selected_nodes = []
    eut = do_sync(epoch, eut_schedule)
    lut = do_sync(epoch, lut_schedule)
    avg_rounds = 0
    avg_eps = 0
    avg_eta_phi = 0
    for a in aggregators:
        if eut:
            worker_models[a] = model.copy().send(nodes[a])
            children = fog_graph[a]
            selected = children[np.random.randint(0, len(children))]
            selected_nodes.append(selected)
        else:
            children = fog_graph[a]
            for child in children:
                worker_models[child].move(nodes[a])
            num_nodes_in_cluster = len(children)
            V = consensus_matrix(num_nodes_in_cluster,
                                 args.graphs[0] if args.const_graph
                                 else args.graphs[
                                         np.random.randint(
                                             0, len(args.graphs)+1)],
                                 args.factor, args.topology)
            eps = get_cluster_eps(children, worker_models, nodes)
            lamda = get_spectral_radius(V - (1/num_nodes_in_cluster))
            if args.lut_intv:  # if lut_intv and rounds given manually 
                rounds = args.rounds if lut else 0
            else:  # else calculate using tthl algorithm
                eps = eps.clip(min=cfg.F['eps_min'])
                rounds = int(np.ceil(np.log(
                    (eta*kwargs.phi)/(num_nodes_in_cluster*eps))/np.log(lamda)))
                rounds = int(max(0, rounds))  # *cfg.F['gamma'])

                if rounds: # c: consensus rounds data, nc: no consensus rounds data
                    aggregate_eps['{}_c'.format(a)].append((epoch, eps))
                else:
                    aggregate_eps['{}_nc'.format(a)].append((epoch, eps))
            aggregate_rounds[a] = rounds
            aggregate_lamda[a] = lamda
            aggregate_sc[a] = num_nodes_in_cluster

            avg_rounds += rounds
            avg_eps += eps
            log_list.append([a, '{:.2f}'.format(lamda), '{:.6f}'.format(eps),
                             '{:.6f}'.format(eta), rounds])

            if rounds > 0:
                laplacian_consensus(children, nodes, worker_models,
                                    V.to(device), rounds)
                for child in children:
                    worker_models[child] = worker_models[child].send(nodes[child])
            else:
                for child in children:
                    worker_models[child] = worker_models[child].get().send(nodes[child])
        if not args.lut_intv:
            avg_eta_phi += eta*kwargs.phi
    avg_rounds /= len(aggregators)
    avg_eps /= len(aggregators)
    if not args.lut_intv:
        avg_eta_phi /= len(aggregators)

    if eut:
        weights = []
        agg_grads = []
        selected_grads = []
        selected_sigmas = []

        # model sum
        for _ in selected_nodes:
            worker_models[_] = worker_models[_].get()
            weights.append(weight_nodes[_])
            agg_grads.append([weight_nodes[w]*_ for _ in worker_grads[w]])
            selected_grads.append(worker_grads[w])
            selected_sigmas.append(worker_sigmas[w])
        agg_model = averaging_consensus(
            selected_nodes, worker_models, weights)
        model.load_state_dict(agg_model)

        # sigma and delta estimations for tau optimization
        if args.tau_max:
            agg_grads = get_tensor_sum(agg_grads)
            weights = get_flattened_weights(model)
            args.delta = max(
                [estimate_delta(
                    args.zeta, _, agg_grads, weights)
                 for _ in selected_grads]
            )
            args.sigma = max(selected_sigmas)

    if len(log_list) > 1:
        table = AsciiTable(log_list)
        table.title = 'worker-train'
        print(table.table)

    loss = np.array([_ for dump, _ in worker_losses.items()])
    acc = np.array([_ for dump, _ in worker_accs.items()])

    return worker_models, acc.mean(), acc.std(), \
        loss.mean(), loss.std(), avg_rounds, avg_eps, avg_eta_phi, \
        aggregate_eps, aggregate_rounds, aggregate_sc, \
        aggregate_lamda, eut


def fl_train(args, model, fog_graph, nodes, X_trains, y_trains,
             device, epoch, eut_schedule, loss_fn,
             worker_models):
    # federated learning with model averaging
    loss_fn_ = get_loss_fn(loss_fn)
    model.train()

    worker_losses = {}
    worker_accs = {}

    # send data, model to workers
    # setup optimizer for each worker
    workers, worker_data, worker_targets, \
        worker_num_samples = send_to_worker(nodes, X_trains, y_trains)
    downlink = do_sync(epoch-1, eut_schedule) or epoch == 1

    for w in workers:
        if downlink:
            worker_models[w] = model.copy().send(nodes[w])
        node_model = worker_models[w].get()
        data = worker_data[w].get()
        target = worker_targets[w].get()
        w_loss, w_acc, _, _ = train_step(node_model, data, target,
                                         loss_fn_, args, device)
        worker_models[w] = node_model.send(nodes[w])
        worker_losses[w] = w_loss
        worker_accs[w] = w_acc

    if epoch in eut_schedule:
        agg = 'L1_W0'
        worker_models[agg] = model.copy().send(nodes[agg])
        for w in workers:
            worker_models[w].move(nodes[agg])

        with torch.no_grad():
            weighted_models = [get_model_weights(
                worker_models[w],
                worker_num_samples[w]/args.num_train) for w in workers]
            model_sum = get_tensor_sum(weighted_models)
            worker_models[agg].load_state_dict(model_sum)

        master = get_model_weights(worker_models[agg].get())
        model.load_state_dict(master)

    loss = np.array([_ for dump, _ in worker_losses.items()])
    acc = np.array([_ for dump, _ in worker_accs.items()])

    return worker_models, acc.mean(), acc.std(), loss.mean(), loss.std()


def test(args, model, device, test_loader, best, epoch=0, loss_fn='nll'):

    if loss_fn == 'nll':
        loss_fn_ = F.nll_loss
    elif loss_fn == 'hinge':
        loss_fn_ = multiClassHingeLoss()

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            if loss_fn == 'nll':
                test_loss += loss_fn_(output, target, reduction='sum').item()
            elif loss_fn == 'hinge':
                test_loss += loss_fn_(output, target).item()
            pred = output.argmax(1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)

    return accuracy, test_loss
