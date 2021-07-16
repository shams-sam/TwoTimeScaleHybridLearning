import argparse
import common.config as cfg
from common.utils import Struct
from models.cnn import CNN
from models.fcn import FCN
from models.svm import SVM
import torch


ap = argparse.ArgumentParser()
ap.add_argument('--models', required=True, type=str, nargs="+")
ap.add_argument('--dataset', required=True, type=str)
args = vars(ap.parse_args())
args = Struct(**args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if 'cnn' in args.models:
    print("Initializing CNN...")
    model = CNN(cfg.input_sizes[args.dataset],
                cfg.output_sizes[args.dataset]).to(device)
    print('NA', model.output_size)
    init_path = '../ckpts/init/{}_cnn.init'.format(args.dataset)
    torch.save(model.state_dict(), init_path)
    print('Save init: {}'.format(init_path))

if 'fcn' in args.models:
    print("Initializing FCN...")
    model = FCN(cfg.input_sizes[args.dataset],
                cfg.output_sizes[args.dataset]).to(device)
    print(model.input_size, model.output_size)
    init_path = '../ckpts/init/{}_fcn.init'.format(args.dataset)
    torch.save(model.state_dict(), init_path)
    print('Save init: {}'.format(init_path))

if 'svm' in args.models:
    print("Initializing SVM...")
    model = SVM(cfg.input_sizes[args.dataset],
                cfg.output_sizes[args.dataset]).to(device)
    print(model.n_feature, model.n_class)
    init_path = '../ckpts/init/{}_svm.init'.format(args.dataset)
    torch.save(model.state_dict(), init_path)
    print('Save init: {}'.format(init_path))
