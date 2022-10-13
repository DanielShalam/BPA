import os

import wandb
import argparse
import random
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment as linear_assignment

from datasets.samplers import CategoriesSampler
from models.wrn_mixup_model import wrn28_10
from datasets.mini_imagenet import MiniImageNet
from methods.PT_MAP.train.PT_MAP import PT_MAP
from self_optimal_transport import SOT

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


def get_dataloader(set_name: str, args: argparse):
    """
    Get dataloader with categorical sampler for few-shot classification.
    """
    num_episodes = args.set_episodes[set_name]
    num_way = args.train_way if set_name == 'train' else args.val_way
    # define datasets and loaders
    data_set = datasets[args.dataset.lower()](data_path=args.data_path, setname=set_name,
                                              backbone=args.backbone, augment=set_name == 'train' and args.augment)
    data_sampler = CategoriesSampler(set_name=set_name, labels=data_set.label, num_episodes=num_episodes,
                                     num_way=num_way, num_shot=args.num_shot, num_query=args.num_query)
    data_loader = DataLoader(dataset=data_set, batch_sampler=data_sampler, num_workers=args.num_workers,
                             pin_memory=set_name != 'test')
    return data_loader


def get_optimizer(args: argparse, params):
    """
    Get optimizer.
    """
    if args.optimizer == 'adam':
        return optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        return optim.SGD(params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError


def get_scheduler(args: argparse, optimizer: torch.optim):
    """
    Get optimizer.
    """
    if args.scheduler == '':
        return None
    elif args.scheduler == 'step':
        return optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=args.step_size, gamma=args.gamma)
    else:
        raise ValueError


def get_method(args: argparse, sot: SOT):
    """
    Get the few-shot classification method (e.g. pt_map).
    """
    if args.method.lower() in methods.keys():
        return methods[args.method.lower()](args=vars(args), sot=sot)
    else:
        raise ValueError(f'Not implemented method. available methods are: {list(methods.keys())}')


def get_criterion_by_method(method: str):
    """
    Get loss function based on the method.
    """
    if 'pt_map' in method:
        return torch.nn.NLLLoss()
    else:
        raise ValueError(f'Not implemented criterion for this method. available methods are: {list(methods.keys())}')


def get_logger(args: argparse):
    """
    Initialize and returns wandb logger if args.wandb is True.
    """
    if args.wandb:
        exp_name = f'model_{args.backbone}_method_{args.method}_sot_{args.sot}_diag_mask_{args.mask_diag}'
        logger = wandb.init(project=args.project, entity=args.entity, name=exp_name, config=vars(args))

        # define which metrics will be plotted against it
        logger.define_metric("train_loss", step_metric="epoch")
        logger.define_metric("train_accuracy", step_metric="epoch")
        logger.define_metric("val_loss", step_metric="epoch")
        logger.define_metric("val_accuracy", step_metric="epoch")
        return logger

    return None


def log_step(results: dict, logger: wandb):
    """
    Log step to the logger without print.
    """
    if logger is not None:
        logger.log(results)


def get_output_dir(args: argparse):
    """
    Initialize the output dir.
    """
    out_dir = f'./checkpoints/{args.backbone.lower()}/{args.dataset.lower()}/' \
              f'way{args.train_way}_shot{args.num_shot}_mask_diag{args.mask_diag}_lr{args.lr}'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print(f'Model will be saved in {out_dir}')
    return out_dir


def print_and_log(results: dict, n: int = 0, logger: wandb = None, epoch: int = None):
    """
    Print and log current results.
    """
    for key in results.keys():
        # average by n if needed (n > 0)
        if n > 0:
            results[key] = results[key] / n

        # print and log
        print(f'{key}: {results[key]:.4f}')

    if logger is not None:
        if epoch is not None:
            results['epoch'] = epoch
        logger.log(results)


def load_weights(model: torch.nn.Module, path: str):
    """
    Load pretrained weights from given path.
    """
    if path is None or path == '':
        print(f"Can't find model in {path}, resuming without loading ")
        return model

    print(f'Loading weights from {path}')
    state_dict = torch.load(path)
    if 'state' in state_dict.keys():
        model.load_state_dict(state_dict['state'])
    else:
        model.load_state_dict(state_dict)

    print("Weights loaded successfully !")
    return model


def get_fs_labels(num_way: int, num_shot: int, num_query: int, method: str, to_cuda=True):
    """
    Prepare few-shot labels. For example for 5-way, 1-shot, 2-query: [0, 1, 2, 3, 4, 0, 0, 1, 1, ...]
    """
    if 'pt_map' in method.lower():
        labels = torch.arange(num_way).view(1, 1, num_way).expand(1, num_shot + num_query, num_way).clone().view(1, -1)
        labels = labels[0]
    else:
        support_label = torch.arange(num_way).reshape((num_way, 1)).expand((num_way, num_shot)).reshape(-1)
        query_label = torch.arange(num_way).reshape((num_way, 1)).expand((num_way, num_query)).reshape(-1)
        labels = torch.cat((support_label, query_label))

    if to_cuda:
        return labels.cuda()
    else:
        return labels


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


def set_seed(seed: int):
    """
    seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def clustering_accuracy(true_row_labels, predicted_row_labels):
    """
    The :mod:`coclust.evaluation.external` module provides functions
    to evaluate clustering or co-clustering results with external information
    such as the true labeling of the clusters.
    """

    """Get the best accuracy.
    Parameters
    ----------
    true_row_labels: array-like
        The true row labels, given as external information
    predicted_row_labels: array-like
        The row labels predicted by the model
    Returns
    -------
    float
        Best value of accuracy
    """

    cm = confusion_matrix(true_row_labels, predicted_row_labels)
    rows, cols = linear_assignment(_make_cost_m(cm))
    total = 0
    for row, column in zip(rows, cols):
        value = cm[row][column]
        total += value

    return (total * 1. / np.sum(cm)), cols


def _make_cost_m(cm):
    s = np.max(cm)
    return - cm + s
