from consensus import averaging_consensus, consensus_matrix, \
    do_aggregation, laplacian_consensus
from model_op import add_model_weights, get_model_weights
from multi_class_hinge_loss import multiClassHingeLoss
import numpy as np
from random import shuffle
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import flip, get_dataloader


def fog_train(args, model, fog_graph, nodes, X_trains, y_trains,
              device, epoch, loss_fn, consensus, rounds, radius,
              d2d, factor, shuffle_worker_data, worker_models):
    # fog learning with model averaging
    if loss_fn == 'nll':
        loss_fn_ = F.nll_loss
    elif loss_fn == 'hinge':
        loss_fn_ = multiClassHingeLoss()

    model.train()

    worker_data = {}
    worker_targets = {}
    worker_num_samples = {}
    worker_optims = {}
    worker_losses = {}

    # send data, model to workers
    # setup optimizer for each worker
    if shuffle_worker_data:
        data = list(zip(X_trains, y_trains))
        shuffle(data)
        X_trains, y_trains = zip(*data)

    workers = [_ for _ in nodes.keys() if 'L0' in _]
    for w, x, y in zip(workers, X_trains, y_trains):
        worker_data[w] = x.send(nodes[w])
        worker_targets[w] = y.send(nodes[w])
        worker_num_samples[w] = x.shape[0]

    for w in workers:
        if do_aggregation(epoch-1, args.eut_int) or epoch == 1:
            worker_models[w] = model.copy().send(nodes[w])
        node_model = worker_models[w].get()
        worker_optims[w] = optim.SGD(
            params=node_model.parameters(),
            lr=args.lr*np.exp(-0.01*epoch) if args.nesterov else args.lr,
            weight_decay=args.decay)
        data = worker_data[w].get()
        target = worker_targets[w].get()
        dataloader = get_dataloader(data, target, args.batch_size)

        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            worker_optims[w].zero_grad()
            output = node_model(data)
            loss = loss_fn_(output, target)
            loss.backward()
            worker_optims[w].step()
        worker_models[w] = node_model.send(nodes[w])
        worker_losses[w] = loss.item()

    var_radius = type(radius) == list
    eut = do_aggregation(epoch, args.eut_int)
    lut = do_aggregation(epoch, args.lut_int)
    for layer_num in range(1, len(args.num_clusters)+1):
        aggregators = [_ for _ in nodes.keys() if 'L{}'.format(layer_num) in _]
        if eut or (layer_num == 1 and lut):
            for a in aggregators:
                if eut:
                    worker_models[a] = model.copy().send(nodes[a])
                worker_num_samples[a] = 1
                children = fog_graph[a]

                for child in children:
                    worker_models[child].move(nodes[a])

                if consensus == 'averaging' or \
                   flip(1-d2d) or (eut and not lut):
                    model_sum = averaging_consensus(children, worker_models,
                                                    worker_num_samples)
                    worker_models[a].load_state_dict(model_sum)
                elif consensus == 'laplacian':
                    if lut:
                        num_nodes_in_cluster = len(children)
                        V = consensus_matrix(num_nodes_in_cluster,
                                             radius if not var_radius
                                             else radius[layer_num-1],
                                             factor, args.topology)
                        model_sum = laplacian_consensus(
                            children, nodes,
                            worker_models,
                            worker_num_samples,
                            V.to(device), rounds)
                    else:
                        for child in children:
                            worker_models[child] = worker_models[child].send(
                                nodes[child])
                    if eut:
                        agg_model = worker_models[a].get()
                        agg_model.load_state_dict(model_sum)
                        worker_models[a] = agg_model.send(nodes[a])

    if eut:
        assert len(aggregators) == 1
        master = get_model_weights(worker_models[aggregators[0]].get(),
                                   1/args.num_train)

        model.load_state_dict(master)

    if epoch % args.log_interval == 0:
        loss = np.array([_ for dump, _ in worker_losses.items()])
        print('\n[EUT:{}-LUT:{}] '
              'Train Epoch: {}({}) \tLoss: {:.6f} +- {:.6f}'.format(
                  eut, lut,
                  epoch, len(dataloader), loss.mean(), loss.std()))


def fl_train(args, model, fog_graph, nodes, X_trains, y_trains,
             device, epoch, loss_fn='nll'):
    # federated learning with model averaging

    if loss_fn == 'nll':
        loss_fn_ = F.nll_loss
    elif loss_fn == 'hinge':
        loss_fn_ = multiClassHingeLoss()

    model.train()

    worker_data = {}
    worker_targets = {}
    worker_num_samples = {}
    worker_models = {}
    worker_optims = {}
    worker_losses = {}

    # send data, model to workers
    # setup optimizer for each worker

    workers = [_ for _ in nodes.keys() if 'L0' in _]
    for w, x, y in zip(workers, X_trains, y_trains):
        worker_data[w] = x.send(nodes[w])
        worker_targets[w] = y.send(nodes[w])
        worker_num_samples[w] = x.shape[0]

    for w in workers:
        worker_models[w] = model.copy().send(nodes[w])
        node_model = worker_models[w].get()
        worker_optims[w] = optim.SGD(
            params=worker_models[w].parameters(), lr=args.lr)

        data = worker_data[w].get()
        target = worker_targets[w].get()
        dataloader = get_dataloader(data, target, args.batch_size)

        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            worker_optims[w].zero_grad()
            output = node_model(data)
            loss = loss_fn_(output, target)
            loss.backward()
            worker_optims[w].step()
        worker_models[w] = node_model.send(nodes[w])
        worker_losses[w] = loss.item()

    agg = 'L1_W0'
    worker_models[agg] = model.copy().send(nodes[agg])
    children = fog_graph[agg]

    for child in children:
        worker_models[child].move(nodes[agg])

    with torch.no_grad():
        weighted_models = [get_model_weights(
            worker_models[_],
            worker_num_samples[_]/args.num_train) for _ in children]
        model_sum = weighted_models[0]
        for m in weighted_models[1:]:
            model_sum = add_model_weights(model_sum, m)
        worker_models[agg].load_state_dict(model_sum)

    master = get_model_weights(worker_models[agg].get())
    model.load_state_dict(master)

    if epoch % args.log_interval == 0:
        loss = np.array([_ for dump, _ in worker_losses.items()])
        print('Train Epoch: {} \tLoss: {:.6f} +- {:.6f}'.format(
            epoch, loss.mean(), loss.std(), dict(grad).values()
        ))


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

    if epoch % args.log_interval == 0:
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%) ==> '
              '{:.2f}%'.format(
                  test_loss, correct, len(test_loader.dataset),
                  100.*accuracy, 100.*best))

    return accuracy, test_loss


def fog_test(args, nodes, X_tests, y_tests,
             device, epoch, loss_fn, worker_models):
    # fog learning with model averaging
    if loss_fn == 'nll':
        loss_fn_ = F.nll_loss
    elif loss_fn == 'hinge':
        loss_fn_ = multiClassHingeLoss()

    worker_data = {}
    worker_targets = {}
    worker_num_samples = {}
    worker_losses = {}
    worker_accs = {}

    workers = [_ for _ in nodes.keys() if 'L0' in _]
    for w, x, y in zip(workers, X_tests, y_tests):
        worker_data[w] = x.send(nodes[w])
        worker_targets[w] = y.send(nodes[w])
        worker_num_samples[w] = x.shape[0]

    for w in workers:
        node_model = worker_models[w].get()
        data = worker_data[w].get()
        target = worker_targets[w].get()
        dataloader = get_dataloader(data, target, worker_num_samples[w])

        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = node_model(data)
            loss = loss_fn_(output, target)
            pred = output.argmax(1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
        worker_accs[w] = correct / len(dataloader.dataset)
        worker_models[w] = node_model.send(nodes[w])
        worker_losses[w] = loss.item()

    if epoch % args.log_interval == 0:
        loss = np.array([_ for dump, _ in worker_losses.items()])
        acc = np.array([_ for dump, _ in worker_accs.items()])
        print('Test fog: {}({}) Accuracy: {:.4f} += {:.4f} '
              '\tLoss: {:.6f} +- {:.6f}\n'.format(
                  epoch, len(dataloader), acc.mean(), acc.std(),
                  loss.mean(), loss.std()))

    return loss.mean(), loss.std(), acc.mean(), acc.std()
