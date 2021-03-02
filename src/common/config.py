ckpt_path = '../ckpts'

num_trains = {
    'mnist': 60000,
    'cifar': 50000,
    'fmnist': 60000
}

num_tests = {
    'mnist': 10000,
    'cifar': 10000,
    'fmnist': 10000
}

input_sizes = {
    'mnist': 28*28,
    'cifar': 3*32*32,
    'fmnist': 28*28
}

output_sizes = {
    'mnist': 10,
    'cifar': 10,
    'fmnist': 10
}

E_glob = 250  # \times 10^4 J
D_glob = 0.25  # seconds
F = {
    'eps_min': 10**(-10),
    'gamma': 1/4.0,
    'gamma_pred': 1/1.0,
    'phi': 1/10.0
}
