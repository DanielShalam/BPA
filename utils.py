import argparse

import numpy as np
import torch
from torch import optim
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment as linear_assignment
from torch.utils.data import DataLoader

from datasets.samplers import CategoriesSampler
from models.wrn_mixup_model import wrn28_10
from datasets.mini_imagenet import MiniImageNet
from methods.PT_MAP import PT_MAP

models = dict(wrn=wrn28_10)
datasets = dict(miniimagenet=MiniImageNet)
methods = dict(pt_map=PT_MAP)


def get_model(model_name: str):
    """
    Get the backbone model.
    """
    if model_name.lower() in models.keys():
        return models[model_name.lower()]()
    else:
        raise ValueError(f'Model not implemented. available models are: {list(models.keys())}')


def get_dataloader(set_name: str, args):
    """
    Get dataloader with categorical sampler for few-shot classification.
    """
    num_episodes = args.train_episodes if set_name == 'train' else args.eval_episodes
    # define datasets and loaders
    data_set = datasets[args.dataset.lower()](data_path=args.data_path, setname=set_name,
                                              backbone=args.backbone, augment=set_name == 'train')
    data_sampler = CategoriesSampler(set_name=set_name, labels=data_set.label, num_episodes=num_episodes,
                                     num_way=args.num_way, num_shot=args.num_shot, num_query=args.num_query)
    data_loader = DataLoader(dataset=data_set, batch_sampler=data_sampler, num_workers=2, pin_memory=True)
    return data_loader


def get_optimizer(params, optimizer: str, lr: float):
    """
    Get optimizer.
    """
    if optimizer == 'adam':
        return optim.Adam(params, lr=lr)
    elif optimizer == 'sgd':
        return optim.SGD(params, lr=lr)
    else:
        raise ValueError


def get_method(args, sot):
    """
    Get the few-shot classification method (e.g. pt_map).
    """
    if args.method.lower() in methods.keys():
        return methods[args.method.lower()](args=vars(args), sot=sot)
    else:
        raise ValueError(f'Not implemented method. available methods are: {list(methods.keys())}')


def get_fs_labels(num_way, num_shot, num_query):
    """
    Prepare few-shot labels. For example for 5-way, 1-shot, 2-query: [0, 1, 2, 3, 4, 0, 0, 1, 1, ...]
    """
    support_label = torch.arange(num_way).reshape((num_way, 1)).expand((num_way, num_shot)).reshape(-1)
    query_label = torch.arange(num_way).reshape((num_way, 1)).expand((num_way, num_query)).reshape(-1)
    return query_label, torch.cat((support_label, query_label))


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")