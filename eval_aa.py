"""
Evaluation with AutoAttack.

python eval-aa.py --fname_input xxx --eps_eval xxx --batch_size_for_eval xxx
"""

import json
import time
import argparse
from model_zoo import *
import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from autoattack import AutoAttack

from core.data import get_data_info
from core.data import load_data
from core.models import create_model

from core.utils import Logger
from core.utils import seed


# Setup

def parser_eval():
    """
    Parse input arguments (eval-adv.py, eval-corr.py, eval-aa.py).
    """
    parser = argparse.ArgumentParser(description='Robustness evaluation.')
    ATTACKS = ['fgsm', 'linf-pgd', 'linf-df', 'linf-apgd', 'fgm', 'l2-pgd', 'l2-df', 'l2-apgd']
    parser.add_argument('--norm_attack', type=str, default='Linf', choices = ['Linf', 'L2'])
    parser.add_argument('--eps_eval', type=float, default=8, help='Random seed.') # 8 for Linf, 0.5 for L2
    parser.add_argument('--fname_input', type=str, default='...')
    parser.add_argument('-a', '--attack', type=str, choices=ATTACKS, default='linf-pgd', help='Type of attack.')
    parser.add_argument('--depth', default=18, type=int, help="Number of layers.")
    parser.add_argument('--batch_size_for_eval', type=int, default=1024)
    parser.add_argument('--data_dir', default='./data/', type=str, help="The folder where you store your dataset")
    parser.add_argument('--act_fn', default="relu", choices=["relu", "softplus", "swish"],
                        help="choose the activation function for your model")
    parser.add_argument('--train', action='store_true', default=False, help='Evaluate on training set.')
    parser.add_argument('-v', '--version', type=str, default='custom', choices=['custom', 'plus', 'standard'], 
                        help='Version of AA.')
    parser.add_argument('--model_type', default='PreActResNet', choices=['WideResNet', 'ResNet', 'PreActResNet'])
    parser.add_argument('--seed', type=int, default=1, help='Random seed.')
    return parser

if __name__ == "__main__":

    parse = parser_eval()
    args = parse.parse_args()

    if args.norm_attack == 'Linf':
        eps_eval = args.eps_eval/255. # will use the eps specified by the parser_eval
    else:
        eps_eval = args.eps_eval

    # accessing and appending the args for training the model
    # with open(args.fname_input+'/args.txt', 'r') as f:
    #     old = json.load(f)
    #     args.__dict__ = dict(vars(args), **old) # new args = args from parser_eval and training args

    DATA_DIR = args.data_dir
    WEIGHTS = args.fname_input + '/val_best.pt'

    # log_path = args.fname_input + '/log-aa.log'
    # logger = Logger(log_path)
    # logger.log('\n\n')

    info = get_data_info(DATA_DIR)
    BATCH_SIZE = 256
    BATCH_SIZE_VALIDATION = args.batch_size_for_eval
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data

    seed(args.seed)
    _, _, train_dataloader, test_dataloader = load_data(DATA_DIR, BATCH_SIZE, BATCH_SIZE_VALIDATION, use_augmentation=False,
                                                        shuffle_train=False)

    if args.train:
        print('Evaluating on training set.')
        l = [x for (x, y) in train_dataloader]
        x_test = torch.cat(l, 0)
        l = [y for (x, y) in train_dataloader]
        y_test = torch.cat(l, 0)
    else:
        l = [x for (x, y) in test_dataloader]
        x_test = torch.cat(l, 0)
        l = [y for (x, y) in test_dataloader]
        y_test = torch.cat(l, 0)

    print('evaluation data size:{}'.format(y_test.size(0)))
    # Model

    # +
    conv1_size = 3
    if args.act_fn == "relu":
        activation_fn = nn.ReLU
    elif args.act_fn == "softplus":
        activation_fn = nn.Softplus
    elif args.act_fn == "swish":
        activation_fn = Swish
    else:
        raise NotImplementedError("Unsupported activation function!")


    class NormalizeByChannelMeanStd(torch.nn.Module):
        def __init__(self, mean, std):
            super(NormalizeByChannelMeanStd, self).__init__()
            if not isinstance(mean, torch.Tensor):
                mean = torch.tensor(mean)
            if not isinstance(std, torch.Tensor):
                std = torch.tensor(std)
            self.register_buffer("mean", mean)
            self.register_buffer("std", std)

        def forward(self, tensor):
            return self.normalize_fn(tensor, self.mean, self.std)

        def extra_repr(self):
            return 'mean={}, std={}'.format(self.mean, self.std)

        def normalize_fn(self, tensor, mean, std):
            """Differentiable version of torchvision.functional.normalize"""
            # here we assume the color channel is in at dim=1
            mean = mean[None, :, None, None]
            std = std[None, :, None, None]
            return tensor.sub(mean).div(std)

    num_classes=200
    norm_layer= NormalizeByChannelMeanStd(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if args.model_type == "WideResNet":
        if args.depth == 16:
            model = WRN_16_8(num_classes=num_classes, dropout=args.dropout,
                             activation_fn=activation_fn)
        elif args.depth == 28:
            model = WRN_28_10(num_classes=num_classes, dropout=args.dropout,
                              activation_fn=activation_fn)
        elif args.depth == 34:
            model = WRN_34_10(num_classes=num_classes, dropout=args.dropout,
                              activation_fn=args.act_fn)
        elif args.depth == 70:
            model = WRN_70_16(num_classes=num_classes, dropout=args.dropout,
                              activation_fn=activation_fn)
        else:
            raise NotImplementedError("Unsupported WideResNet!")
    elif args.model_type == "PreActResNet":
        if args.depth == 18:
            model = PreActResNet18(num_classes=num_classes, activation_fn=activation_fn)
        elif args.depth == 34:
            model = PreActResNet34(num_classes=num_classes, activation_fn=activation_fn)
        else:
            model = PreActResNet50(num_classes=num_classes, activation_fn=activation_fn)
    elif args.model_type == "ResNet":
        if args.depth == 18:
            model = ResNet18(num_classes=num_classes, activation_fn=activation_fn)
        elif args.depth == 34:
            model = ResNet34(num_classes=num_classes, activation_fn=activation_fn)
        else:
            model = ResNet50(num_classes=num_classes, activation_fn=activation_fn)
    else:
        raise NotImplementedError("Unsupported Model Type!")
    model.normalize = norm_layer

    model.load_state_dict(torch.load(args.fname_input, map_location=torch.device(device)))
    model = model.to(device)
    model_name = ".".join(args.fname_input.split('/')[-1].split('.')[:-1])

    # model = create_model(args.model, args.normalize, info, device,GroupNorm=args.GroupNorm) ## dataParallel
    # # checkpoint = torch.load(WEIGHTS)
    # if 'tau' in args and args.tau:
    #     logger.log('Using WA model.')
    # else:
    #     raise ValueError('Why not using WA model? Check again?')
    #
    # try:
    #     model.load_state_dict(checkpoint['wa_model'])
    # except:
    #     model.module.load_state_dict(checkpoint['wa_model']) # when checkpt is not dataParallel
    #
    model.eval()
    # -

    # AA Evaluation

    # +
    seed(args.seed)
    if args.norm_attack == 'Linf':
        assert args.attack in ['fgsm', 'linf-pgd', 'linf-df', 'linf-apgd']
    elif args.norm_attack == 'L2':
        assert args.attack in ['fgm', 'l2-pgd', 'l2-df', 'l2-apgd']
    else:
        raise ValueError('Invalid norm_attack for evaluation')

    adversary = AutoAttack(model, norm=args.norm_attack, eps=eps_eval, version=args.version, seed=args.seed)
    #

    print('eps:{:.4f} batch size:{}\n'.format(eps_eval,BATCH_SIZE_VALIDATION))

    if args.version == 'custom':
        adversary.attacks_to_run = ['apgd-ce', 'apgd-t', 'fab-t']
        adversary.apgd.n_restarts = 1
        adversary.apgd_targeted.n_restarts = 1

    with torch.no_grad():
        x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=BATCH_SIZE_VALIDATION)

    print ('Script Completed.')
