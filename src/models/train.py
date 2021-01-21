from common.consensus import averaging_consensus, consensus_matrix, \
    do_aggregation, laplacian_consensus
from common.utils import flip, get_dataloader
from models.model_op import add_model_weights, get_model_weights
from models.multi_class_hinge_loss import multiClassHingeLoss
import numpy as np
from random import shuffle
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
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        worker_optim.zero_grad()
        output = model(data)
        pred = output.argmax(1, keepdim=True)
        w_correct += pred.eq(target.view_as(pred)).sum().item()
        loss = loss_fn(output, target)
        w_loss += loss.item()
        loss.backward()
        worker_optim.step()

    return w_loss/len(dataloader), w_correct/len(dataloader.dataset)


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
    cluster_weights = {head: len(children)/len(workers)
                     for head, children in nodes.items()
                     if 'L1' in head}

    for w, x, y in zip(workers, X_trains, y_trains):
        worker_data[w] = x.send(nodes[w])
        worker_targets[w] = y.send(nodes[w])
        worker_num_samples[w] = x.shape[0]

    return workers, worker_data, worker_targets, \
        worker_num_samples, cluster_weights


def tthl_train(args, model, fog_graph, nodes, X_trains, y_trains,
               device, epoch, loss_fn, consensus,
               eut_schedule, worker_models):

    model.train()
    loss_fn_ = get_loss_fn(loss_fn)
    worker_losses, worker_accs, worker_models = {}, {}, {}

    # send data, model to workers
    # setup optimizer for each worker
    workers, worker_data, worker_targets, \
        _, cluster_weights = send_to_worker(nodes, X_trains, y_trains)
    
    for w in workers:
        if do_aggregation(epoch-1, args.eut_int) or epoch == 1:
            worker_models[w] = model.copy().send(nodes[w])
        node_model = worker_models[w].get()
        data = worker_data[w].get()
        target = worker_targets[w].get()
        w_loss, w_acc = train_step(
            node_model, data, target, loss_fn_, args, device)
        worker_models[w] = node_model.send(nodes[w])
        worker_losses[w] = w_loss
        worker_accs[w] = w_acc

    eut = epoch in eut_schedule

    log_list = []
    log_head = ['div', 'sigma', 'lamda', 'rounds']
    log_list.append(log_head)

    aggregators = [_ for _ in nodes.keys() if 'L1' in _]
    for a in aggregators:
        if eut:
            worker_models[a] = model.copy().send(nodes[a])
            children_a = fog_graph[a]
            selected = children_a[np.random.randint(0, len(children_a))]
            worker_models[selected].move(nodes[a])
        else:
            theta = calc_theta(epoch, args.lut_int)
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
            model_sum, log = laplacian_consensus(
                children, nodes,
                worker_models,
                worker_num_samples,
                V.to(device), rounds, args.sigma_mul)
            
            for child in children:
                worker_models[child] = worker_models[child].send(
                    nodes[child])
            if log:
                log_list.append(log)
            else:
                for child in children:
                    worker_models[child] = worker_models[child].send(
                        nodes[child])
            if eut:
                agg_model = worker_models[a].get()
                agg_model.load_state_dict(model_sum)
                worker_models[a] = agg_model.send(nodes[a])

    if len(log_list) > 1:
        table = AsciiTable(log_list)
        print(table.table)

    if eut:
        assert len(aggregators) == 1
        master = get_model_weights(worker_models[aggregators[0]].get(),
                                   1/args.num_train)

        model.load_state_dict(master)

    loss = np.array([_ for dump, _ in worker_losses.items()])
    acc = np.array([_ for dump, _ in worker_accs.items()])

    return acc.mean(), acc.std(), loss.mean(), loss.std()

def fl_train(args, model, fog_graph, nodes, X_trains, y_trains,
             device, epoch, loss_fn='nll'):
    # federated learning with model averaging
    loss_fn_ = get_loss_fn(loss_fn)
    model.train()

    worker_models = {}
    worker_losses = {}
    worker_accs = {}

    # send data, model to workers
    # setup optimizer for each worker
    workers, worker_data, worker_targets, \
        worker_num_samples, _ = send_to_worker(nodes, X_trains, y_trains)

    for w in workers:
        worker_models[w] = model.copy().send(nodes[w])
        node_model = worker_models[w].get()
        data = worker_data[w].get()
        target = worker_targets[w].get()
        w_loss, w_acc = train_step(node_model, data, target,
                                   loss_fn_, args, device)
        worker_models[w] = node_model.send(nodes[w])
        worker_losses[w] = w_loss
        worker_accs[w] = w_acc

    agg = 'L1_W0'
    worker_models[agg] = model.copy().send(nodes[agg])
    for w in workers:
        worker_models[w].move(nodes[agg])

    with torch.no_grad():
        weighted_models = [get_model_weights(
            worker_models[w],
            worker_num_samples[w]/args.num_train) for w in workers]
        model_sum = weighted_models[0]
        for m in weighted_models[1:]:
            model_sum = add_model_weights(model_sum, m)
        worker_models[agg].load_state_dict(model_sum)

    master = get_model_weights(worker_models[agg].get())
    model.load_state_dict(master)

    loss = np.array([_ for dump, _ in worker_losses.items()])
    acc = np.array([_ for dump, _ in worker_accs.items()])

    return acc.mean(), acc.std(), loss.mean(), loss.std()

# Test
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
