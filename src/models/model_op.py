from collections import defaultdict
import torch



def add_tensor_dicts(weights1, weights2):
    for key, val in weights2.items():
        weights1[key] += val

    return weights1


def add_tensor_lists(grad1, grad2):
    grad = []
    for g1, g2 in zip(grad1, grad2):
        grad.append(g1 + g2)

    return grad


def calc_beta(global_grads, global_model, local_model):
    local_grads = get_flattened_grads(local_model)
    global_weights = get_flattened_weights(global_model)
    local_weights = get_flattened_weights(local_model)

    beta = 0
    count = 0.0
    for lg, gg, lw, gw in zip(
            local_grads, global_grads, local_weights, global_weights):
        beta += torch.norm(lg-gg)/torch.norm(lw-gw)
        count += 1.0

    return beta.item()/count


def calc_mu(global_grads, global_model, local_model):
    local_grads = get_flattened_grads(local_model)
    global_weights = get_flattened_weights(global_model)
    local_weights = get_flattened_weights(local_model)

    mu = 0
    count = 0.0
    for lg, gg, lw, gw in zip(
            local_grads, global_grads, local_weights, global_weights):
        mu += torch.dot(lg-gg, lw-gw)/torch.norm(lw-gw)**2
        count += 1.0

    return mu.item()/count


def calc_sigma(grads):
    grad1, grad2 = grads
    sigma = 0
    count = 0
    for g1, g2 in zip(grad1, grad2):
        sigma += torch.norm(g1-g2).item()/1.4142  # sqrt(2)
        count += 1

    return sigma/count


def get_flattened_grads(model):
    return [_.grad.clone().flatten() for _ in model.parameters()]


def get_flattened_weights(model):
    return [_.clone().flatten() for _ in model.parameters()]


def get_model_grads(model, scaling_factor=1):
    grads = []
    for param in model.parameters():
        grads.append(param.grad.clone()*scaling_factor)
    return grads


def get_model_weights(model, scaling_factor=1):
    if scaling_factor == 1:
        return model.state_dict()

    else:
        weights = model.state_dict()
        for key, val in weights.items():
            weights[key] = val*scaling_factor
        return weights


def get_tensor_sum(tensor_list):
    tensor_sum = tensor_list[0]
    adder_fn = add_tensor_lists \
               if type(tensor_sum) == list else add_tensor_dicts
    for t in tensor_list[1:]:
        tensor_sum = adder_fn(tensor_sum, t)

    return tensor_sum


def scale_model_weights(weights, factor):
    for key, val in weights.items():
        weights[key] = val*factor

    return weights


def weight_gradient(w1, w2, lr):
    return torch.norm((w1.flatten()-w2.flatten())/lr).item()


def model_gradient(model1, model2, lr):
    grads = defaultdict(list)
    for key, val in model1.items():
        grads[key.split('.')[-1]] = weight_gradient(
            model1[key], model2[key], lr)

    return grads


def get_num_params(model):
    return sum([_.flatten().size()[0] for _ in model.parameters()])
