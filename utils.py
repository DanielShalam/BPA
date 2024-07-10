import datetime
import os
import argparse
import random
import time
from collections import defaultdict, deque

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

from models.wrn_mixup_model import wrn28_10
from models.resnet12 import Res12
from datasets import MiniImageNet, CIFAR, CUB
from datasets.samplers import CategoriesSampler
from methods import PTMAPLoss, ProtoLoss

try:
    import wandb
    HAS_WANDB = True
except Exception as e:
    HAS_WANDB = False


MODELS = dict(
    wrn=wrn28_10, resnet12=Res12
)
DATASETS = dict(
    miniimagenet=MiniImageNet, cifar=CIFAR
)
METHODS = dict(
    pt_map=PTMAPLoss, pt_map_bpa=PTMAPLoss, proto=ProtoLoss, proto_bpa=ProtoLoss,
)


def get_model(model_name: str, args):
    """
    Get the backbone model.
    """
    arch = model_name.lower()
    if arch in MODELS.keys():
        model = MODELS[arch](dropout=args.dropout)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        return model
    else:
        raise ValueError(f'Model {model_name} not implemented. available models are: {list(MODELS.keys())}')


def get_dataloader(set_name: str, args: argparse, constant: bool = False):
    """
    Get dataloader with categorical sampler for few-shot classification.
    """
    num_episodes = args.set_episodes[set_name]
    num_way = args.train_way if set_name == 'train' else args.val_way

    # define dataset sampler and data loader
    data_set = DATASETS[args.dataset.lower()](
        args.data_path, set_name, args.backbone,
        augment=set_name == 'train' and args.augment
    )
    args.img_size = data_set.image_size

    data_sampler = CategoriesSampler(
        set_name, data_set.label, num_episodes, const_loader=constant,
        num_way=num_way, num_shot=args.num_shot, num_query=args.num_query,
        replace=set_name == 'train',
    )
    return DataLoader(
        data_set, batch_sampler=data_sampler, num_workers=args.num_workers, pin_memory=not constant
    )


def get_optimizer_and_lr_scheduler(args, params):
    optimizer = get_optimizer(args, params)
    lr_scheduler = get_scheduler(args, optimizer)
    return optimizer, lr_scheduler


def get_optimizer(args, params):
    """
    Get optimizer.
    """
    if args.optimizer == 'adam':
        return optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        return optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        return optim.SGD(params, lr=args.lr, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay)
    else:
        raise ValueError(f'Optimizer {args.optimizer} not available.')


def get_scheduler(args, optimizer: torch.optim):
    """
    Get optimizer.
    """
    if not args.scheduler or args.scheduler == '':
        return None
    elif args.scheduler == 'step':
        return optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=args.step_size, gamma=args.gamma)
    else:
        raise ValueError(f'Error: LR-scheduler {args.scheduler} is not available.')


def get_method(args, bpa=None):
    """
    Get the few-shot classification method (e.g. pt_map).
    """

    if args.method.lower() in METHODS.keys():
        return METHODS[args.method.lower()](args=vars(args), bpa=bpa)
    else:
        raise ValueError(f'Not implemented method. available methods are: {METHODS.keys()}')


def get_criterion_by_method(method: str):
    """
    Get loss function based on the method.
    """

    if 'pt_map' in method:
        return torch.nn.NLLLoss()
    elif 'proto' in method:
        return torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f'Not implemented criterion for this method. available methods are: {list(METHODS.keys())}')


def init_wandb(exp_name: str, args):
    """
    Initialize and returns wandb logger if args.wandb is True.
    """
    if not args.wandb:
        return None
    assert HAS_WANDB, "Install wandb via - 'pip install wandb' in order to use wandb logging. "
    logger = wandb.init(project=args.project, entity=args.entity, name=exp_name, config=vars(args))
    # define which metrics will be plotted against it
    logger.define_metric("train_loss", step_metric="epoch")
    logger.define_metric("train_accuracy", step_metric="epoch")
    logger.define_metric("val_loss", step_metric="epoch")
    logger.define_metric("val_accuracy", step_metric="epoch")
    return logger


def wandb_log(results: dict):
    """
    Log step to the logger without print.
    """
    if HAS_WANDB and wandb.run is not None:
        wandb.log(results)


def get_output_dir(args: argparse):
    """
    Initialize the output dir.
    """

    if args.checkpoint_dir is None:
        checkpoint_dir = os.path.join(args.root_path, 'checkpoints', args.dataset.lower(), args.backbone.lower(), args.method.lower())

        name_str = f'-n_way={args.train_way}' \
                   f'-n_shot={args.num_shot}' \
                   f'-lr={args.lr}' \
                   f'-scheduler={args.scheduler}' \
                   f'-dropout={args.dropout}'

        checkpoint_dir = os.path.join(checkpoint_dir, name_str)
    else:
        checkpoint_dir = args.checkpoint_dir

    if args.eval:
        return checkpoint_dir

    while os.path.exists(checkpoint_dir):
        checkpoint_dir += f'-{np.random.randint(100)}'

    os.makedirs(checkpoint_dir, exist_ok=True)

    # write args to a file
    with open(os.path.join(checkpoint_dir, "args.txt"), 'w') as f:
        for key, value in vars(args).items():
            f.write('%s:%s\n' % (key, value))

    print("=> Checkpoints will be saved at:\n", checkpoint_dir)

    return checkpoint_dir


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


def print_and_log(results: dict, n: int = 0):
    """
    Print and log current results.
    """
    for key in results.keys():
        # average by n if needed (n > 0)
        if n > 0 and 'time' not in key and '/epoch' not in key:
            results[key] = results[key] / n

        # print and log
        print(f'{key}: {results[key]:.4f}')

    if wandb.run is not None:
        wandb.log(results)


def set_seed(seed: int):
    """
    seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)


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


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.6f} ({global_avg:.6f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''

        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.6f}')
        data_time = SmoothedValue(fmt='{avg:.6f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'mem: {memory:.0f} '
                'mem reserved: {memory_res:.0f} '
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            len_iterable = len(iterable)
            if i % print_freq == 0 or i == len_iterable - 1:
                eta_seconds = iter_time.global_avg * (len_iterable - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len_iterable, eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.memory_allocated() / MB,
                        memory_res=torch.cuda.memory_reserved() / MB))
                else:
                    print(log_msg.format(
                        i, len_iterable, eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.6f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))
