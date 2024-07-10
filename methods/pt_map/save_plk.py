from __future__ import print_function
import os
import collections
import pickle
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from data.datamgr import SimpleDataManager
from io_utils import parse_args
import sys

from models import wrn_mixup_model, res_mixup_model

use_gpu = torch.cuda.is_available()


class WrappedModel(nn.Module):
    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module

    def forward(self, x):
        return self.module(x)


def save_pickle(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


def extract_feature(val_loader, model, checkpoint_dir, tag='last'):
    save_dir = '{}/{}'.format(checkpoint_dir, tag)
    if os.path.isfile(save_dir + '/output.plk'):
        data = load_pickle(save_dir + '/output.plk')
        return data
    else:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

    model.eval()
    with torch.no_grad():

        output_dict = collections.defaultdict(list)

        for i, (inputs, labels) in enumerate(val_loader):
            print(f"{i}/{len(val_loader)}")
            # compute output
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs, return_logits=False).cpu().data.numpy()
            for out, label in zip(outputs, labels):
                output_dict[label.item()].append(out)

        all_info = output_dict
        save_pickle(save_dir + '/output.plk', all_info)
        return


def main():
    args = parse_args('test')
    loadfile = args.json_dir + f'{args.dataset.lower()}_novel.json'

    if args.dataset.lower() == 'miniimagenet' or args.dataset.lower() == 'cub':
        datamgr = SimpleDataManager(84, batch_size=64)
        novel_loader = datamgr.get_data_loader(loadfile, aug=False)
    else:
        raise ValueError

    model_file = os.path.join(args.model_dir, args.file_name)
    if args.model == 'WideResNet28_10':
        model = wrn_mixup_model.wrn28_10(num_classes=args.num_classes)
    elif args.model == 'ResNet18':
        model = res_mixup_model.resnet18(num_classes=args.num_classes)
    else:
        raise ValueError

    model = model.cuda()
    cudnn.benchmark = True

    if model_file.endswith('.tar'):
        checkpoint = torch.load(model_file)
        state = checkpoint['state']
        state_keys = list(state.keys())
        callwrap = False
        if 'module' in state_keys[0]:
            callwrap = True
        if callwrap:
            model = WrappedModel(model)
        model_dict_load = model.state_dict()
        model_dict_load.update(state)
        model.load_state_dict(model_dict_load)
    else:
        model.load_state_dict(torch.load(model_file))

    model.eval()
    extract_feature(novel_loader, model, args.model_dir, tag='last')
    print("features saved!")


if __name__ == '__main__':
    main()
