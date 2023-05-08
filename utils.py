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

from models.wrn_mixup_model import wrn28_10
from models.resnet12 import Res12
from datasets import MiniImageNet, CIFAR, CUB
from datasets.samplers import CategoriesSampler
from methods import PTMAPLoss, ProtoLoss
from self_optimal_transport import SOT


models = dict(wrn=wrn28_10, resnet12=Res12)
datasets = dict(miniimagenet=MiniImageNet, cifar=CIFAR)
methods = dict(pt_map=PTMAPLoss, pt_map_sot=PTMAPLoss, proto=ProtoLoss, proto_sot=ProtoLoss, )


def get_model(model_name: str, args):
    """
    Get the backbone model.
    """
    arch = model_name.lower()
    if arch in models.keys():
        if 'vit' in arch:
            model = models[arch](img_size=args.img_size, patch_size=16)
        else:
            model = models[arch](dropout=args.dropout)

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        return model
    else:
        raise ValueError(f'Model {model_name} not implemented. available models are: {list(models.keys())}')


def get_dataloader(set_name: str, args: argparse, constant: bool = False):
    """
    Get dataloader with categorical sampler for few-shot classification.
    """
    num_episodes = args.set_episodes[set_name]
    num_way = args.train_way if set_name == 'train' else args.val_way

    # define dataset sampler and data loader
    data_set = datasets[args.dataset.lower()](
        args.data_path, set_name, args.backbone, augment=set_name == 'train' and args.augment
    )
    args.img_size = data_set.image_size

    data_sampler = CategoriesSampler(
        set_name, data_set.label, num_episodes, const_loader=constant,
        num_way=num_way, num_shot=args.num_shot, num_query=args.num_query
    )
    return DataLoader(
        data_set, batch_sampler=data_sampler, num_workers=args.num_workers, pin_memory=not constant
    )


def get_optimizer(args: argparse, params):
    """
    Get optimizer.
    """
    if args.optimizer == 'adam':
        return optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        return optim.SGD(params, lr=args.lr, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay)
    else:
        raise ValueError(f'Optimizer {args.optimizer} not available.')


def get_scheduler(args: argparse, optimizer: torch.optim):
    """
    Get optimizer.
    """
    if args.scheduler == '':
        return None
    elif args.scheduler == 'step':
        return optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=args.step_size, gamma=args.gamma)
    else:
        raise ValueError(f'Scheduler {args.scheduler} not available.')


def get_method(args: argparse, sot: SOT):
    """
    Get the few-shot classification method (e.g. pt_map).
    """

    if args.method.lower() in methods.keys():
        return methods[args.method.lower()](args=vars(args), sot=sot)
    else:
        raise ValueError(f'Not implemented method. available methods are: {methods.keys()}')


def get_criterion_by_method(method: str):
    """
    Get loss function based on the method.
    """

    if 'pt_map' in method:
        return torch.nn.NLLLoss()
    elif 'proto' in method:
        return torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f'Not implemented criterion for this method. available methods are: {list(methods.keys())}')


def get_logger(exp_name: str, args: argparse):
    """
    Initialize and returns wandb logger if args.wandb is True.
    """
    if args.wandb:
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

    for key, value in results.items():
        if 'acc' in key:
            print(f"{key}: {100 * value:.2f}%")
        else:
            print(f"{key}: {value:.4f}")


def get_output_dir(args: argparse):
    """
    Initialize the output dir.
    """

    out_dir = f'./checkpoints/{args.backbone.lower()}/{args.dataset.lower()}/' \
              f'way{args.train_way}_shot{args.num_shot}_mask_diag{args.mask_diag}' \
              f'_lr{args.lr}_sched{args.scheduler}_step{args.step_size}_drop{args.dropout}'

    if args.eval:
        return out_dir

    while os.path.exists(out_dir):
        out_dir += f'_{np.random.randint(100)}'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # write args to a file
    with open(os.path.join(out_dir, "args.txt"), 'w') as f:
        for key, value in vars(args).items():
            f.write('%s:%s\n' % (key, value))

    print("Model checkpoints will be saved at:", out_dir)
    return out_dir


def load_weights(model: torch.nn.Module, pretrained_path: str):
    """
    Load pretrained weights from given path.
    """
    if not pretrained_path:
        return model

    print(f'Loading weights from {pretrained_path}')
    state_dict = torch.load(pretrained_path)
    sd_keys = list(state_dict.keys())
    if 'state' in sd_keys:
        state_dict = state_dict['state']
        for k in list(state_dict.keys()):
            if k.startswith('module.'):
                state_dict["{}".format(k[len('module.'):])] = state_dict[k]
                del state_dict[k]

        model.load_state_dict(state_dict, strict=False)

    elif 'params' in sd_keys:
        state_dict = state_dict['params']
        for k in list(state_dict.keys()):
            if k.startswith('encoder.'):
                state_dict["{}".format(k[len('encoder.'):])] = state_dict[k]

            del state_dict[k]

        model.load_state_dict(state_dict, strict=True)
    else:
        model.load_state_dict(state_dict)

    print("Weights loaded successfully ")
    return model


def get_fs_labels(method: str, num_way: int, num_query: int, num_shot: int):
    """
    Prepare few-shot labels. For example for 5-way, 1-shot, 2-query: [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, ...]
    """
    n_samples = num_shot + num_query if 'map' in method else num_query
    labels = torch.arange(num_way, dtype=torch.int16).repeat(n_samples).type(torch.LongTensor)

    if torch.cuda.is_available():
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


def print_and_log(results: dict, n: int = 0, logger: wandb = None):
    """
    Print and log current results.
    """
    for key in results.keys():
        # average by n if needed (n > 0)
        if n > 0 and 'time' not in key and '/epoch' not in key:
            results[key] = results[key] / n

        # print and log
        print(f'{key}: {results[key]:.4f}')

    if logger is not None:
        logger.log(results)


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


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
