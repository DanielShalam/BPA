import argparse
import math
import os
from time import time

import torch

import utils
from bpa import BPA


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=1,
                        help="""Random seed.""")

    parser.add_argument('--root_path', type=str, default='./',
                        help=""" Path to project root directory. """)
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help=""" Where to save model checkpoints. If None, it will automatically created. """)
    parser.add_argument('--dataset', type=str, default='miniimagenet',
                        choices=['miniimagenet', 'cifar'])
    parser.add_argument('--data_path', type=str, default='./datasets/few_shot/miniimagenet',
                        help="""Path to dataset root directory.""")

    parser.add_argument('--backbone', type=str, default='wrn',
                        help="""Define which backbone network to use. """)
    parser.add_argument('--pretrained_path', type=str, default=False,
                        help=""" Path to pretrained model, used for testing/fine-tuning. """)

    parser.add_argument('--eval', type=utils.bool_flag, default=False,
                        help=""" If true, make evaluation on the *test set*. 
                             The amount of test episodes controlled by --test_episodes=<>""")
    parser.add_argument('--eval_freq', type=int, default=1,
                        help=""" Evaluate training every n epochs. """)
    parser.add_argument('--eval_first', type=utils.bool_flag, default=False,
                        help=""" Set to true to evaluate the model before training. Useful for fine-tuning. """)
    parser.add_argument('--num_workers', type=int, default=8)

    # wandb specific arguments
    parser.add_argument('--wandb', type=utils.bool_flag, default=False,
                        help=""" Log data into wandb. """)
    parser.add_argument('--project', type=str, default='',
                        help=""" Project name in wandb. """)
    parser.add_argument('--entity', type=str, default='',
                        help=""" Your wandb entity name. """)

    # few-shot specific arguments
    parser.add_argument('--method', type=str, default='pt_map_bpa',
                        choices=['proto', 'proto_bpa', 'pt_map', 'pt_map_bpa'],
                        help="""Specify which few-shot classifier to use.""")
    parser.add_argument('--train_way', type=int, default=5,
                        help=""" Number of classes for each training task. """)
    parser.add_argument('--val_way', type=int, default=5,
                        help=""" Number of classes for each validation/test task. """)
    parser.add_argument('--num_shot', type=int, default=5,
                        help=""" Number of (labeled) support samples for each class. """)
    parser.add_argument('--num_query', type=int, default=15,
                        help=""" Number of (un-labeled) query samples for each class. """)
    parser.add_argument('--train_episodes', type=int, default=200,
                        help=""" Number of few-shot tasks for each epoch. """)
    parser.add_argument('--eval_episodes', type=int, default=400,
                        help=""" Number of tasks to evaluate. """)
    parser.add_argument('--test_episodes', type=int, default=10000,
                        help=""" Number of tasks to evaluate. """)
    parser.add_argument('--temperature', type=float, default=0.1,
                        help=""" Temperature for ProtoNet. """)

    # training specific arguments
    parser.add_argument('--max_epochs', type=int, default=25,
                        help="""Number of training/finetuning epochs. """)
    parser.add_argument('--optimizer', type=str, default='adam',
                        help="""Optimizer""", choices=['adam', 'adamw', 'sgd'])
    parser.add_argument('--lr', type=float, default=5e-5,
                        help="""Learning rate. """)
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help="""Weight decay. """)
    parser.add_argument('--dropout', type=float, default=0.,
                        help=""" Dropout probability. """)
    parser.add_argument('--momentum', type=float, default=0.9,
                        help="""Momentum of SGD optimizer. """)
    parser.add_argument('--scheduler', type=str, default='step',
                        help="""Learning rate scheduler. To disable the scheduler, use scheduler=''. """)
    parser.add_argument('--step_size', type=int, default=5,
                        help="""Step size (in epochs) of StepLR scheduler. """)
    parser.add_argument('--gamma', type=float, default=0.5,
                        help="""Gamma of StepLR scheduler. """)
    parser.add_argument('--augment', type=utils.bool_flag, default=False,
                        help=""" Apply data augmentation. """)

    # BPA specific arguments
    parser.add_argument('--ot_reg', type=float, default=0.1,
                        help=""" Sinkhorn entropy regularization. 
                                 For few-shot methods, 0.1-0.2 seems to work best. 
                                 For larger tasks (~10,000) samples, try to increase this value. """)
    parser.add_argument('--sink_iters', type=int, default=20,
                        help=""" Number of Sinkhorn iterations. 
                                 Usually small number (~ 5-10) is sufficient. """)
    parser.add_argument('--distance_metric', type=str, default='cosine',
                        help=""" Distance metric for the OT cost matrix. """,
                        choices=['cosine', 'euclidean'])
    parser.add_argument('--mask_diag', type=utils.bool_flag, default=True,
                        help=""" If true, mask diagonal (self) values before and after the OT. """)
    parser.add_argument('--max_scale', type=utils.bool_flag, default=True,
                        help=""" Scaling range of the BPA values to [0,1]. 
                             This should always be True. """)

    return parser.parse_args()


def main():
    args = get_args()
    utils.set_seed(seed=args.seed)
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    output_dir = utils.get_output_dir(args=args)

    # define datasets and loaders
    args.set_episodes = dict(train=args.train_episodes, val=args.eval_episodes, test=args.test_episodes)
    if not args.eval:
        train_dataloader = utils.get_dataloader(set_name='train', args=args, constant=False)
        val_dataloader = utils.get_dataloader(set_name='val', args=args, constant=True)
    else:
        val_dataloader = utils.get_dataloader(set_name='test', args=args, constant=False)
        train_dataloader = None

    # define model and load pretrained weights if available
    model = utils.get_model(args.backbone, args)
    model = model.to(device)
    utils.load_weights(model, args.pretrained_path)

    # BPA and few-shot classification method (e.g. proto, pt-map...)
    bpa = None
    if 'bpa' in args.method.lower():
        bpa = BPA(
            distance_metric=args.distance_metric,
            ot_reg=args.ot_reg,
            mask_diag=args.mask_diag,
            sinkhorn_iterations=args.sink_iters,
            max_scale=args.max_scale
        )
    fewshot_method = utils.get_method(args=args, bpa=bpa)

    # few-shot labels
    train_labels = utils.get_fs_labels(args.method, args.train_way, args.num_query, args.num_shot)
    val_labels = utils.get_fs_labels(args.method, args.val_way, args.num_query, args.num_shot)

    # initialized wandb
    if args.wandb:
        utils.init_wandb(exp_name=output_dir.split('/')[-1], args=args)

    # define loss
    criterion = utils.get_criterion_by_method(method=args.method)

    # Test-set evaluation
    if args.eval:
        print(f"Evaluate model for {args.test_episodes} episodes... ")
        eval_one_epoch(model, val_dataloader, fewshot_method, criterion, val_labels, 0, args, set_name='test')
        exit(1)

    # define optimizer and scheduler
    optimizer, lr_scheduler = utils.get_optimizer_and_lr_scheduler(args=args, params=model.parameters())

    # evaluate model before training
    if args.eval_first:
        print("Evaluate model before training... ")
        eval_one_epoch(model, val_dataloader, fewshot_method, criterion, val_labels, -1, args, set_name='val')

    # train
    print("Start training...")
    best_acc = 0.
    best_loss = math.inf
    for epoch in range(args.max_epochs):
        print("[Epoch {}/{}]...".format(epoch, args.max_epochs))

        # train
        train_one_epoch(model, train_dataloader, optimizer, fewshot_method, criterion, train_labels, epoch, args)
        if lr_scheduler is not None:
            lr_scheduler.step()

        # evaluate
        if epoch % args.eval_freq == 0:
            eval_loss, eval_acc = eval_one_epoch(model, val_dataloader, fewshot_method, criterion, val_labels,
                                                 epoch, args, set_name='val')
            # save best model
            if eval_loss < best_loss:
                best_loss = eval_loss
                torch.save(model.state_dict(), os.path.join(output_dir, 'min_loss.pth'))
            elif eval_acc > best_acc:
                best_acc = eval_acc
                torch.save(model.state_dict(), os.path.join(output_dir, 'max_acc.pth'))

        # save last checkpoint
        torch.save(model.state_dict(), os.path.join(output_dir, 'last.pth'))


def train_one_epoch(model, dataloader, optimizer, fewshot_method, criterion, labels, epoch, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Train Epoch: [{}/{}]'.format(epoch, args.max_epochs)
    log_freq = 50
    n_batches = len(dataloader)

    model.train()
    for batch_idx, (images, _) in enumerate(metric_logger.log_every(dataloader, log_freq, header=header)):
        images = images.to(device)
        # extract features
        features = model(images)
        # few-shot classifier
        probas, accuracy = fewshot_method(features, labels=labels, mode='train')
        q_labels = labels if len(labels) == len(probas) else labels[-len(probas):]
        # loss
        loss = criterion(probas, q_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss.detach().item(),
                             accuracy=accuracy)

        if batch_idx % log_freq == 0:
            utils.wandb_log(
                {
                    'train/loss_step': loss.item(),
                    'train/accuracy_step': accuracy,
                    'train/step': batch_idx + (epoch * n_batches)
                }
            )

    print("Averaged stats:", metric_logger)
    utils.wandb_log(
        {
            'lr': optimizer.param_groups[0]['lr'],
            'train/epoch': epoch,
            'train/loss': metric_logger.loss.global_avg,
            'train/accuracy': metric_logger.accuracy.global_avg,
        }
    )
    return metric_logger


@torch.no_grad()
def eval_one_epoch(model, dataloader, fewshot_method, criterion, labels, epoch, args, set_name):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Validation:' if set_name == "val" else 'Test:'
    log_freq = 50

    n_batches = len(dataloader)
    model.eval()
    for batch_idx, (images, _) in enumerate(metric_logger.log_every(dataloader, log_freq, header=header)):
        images = images.to(device)
        # extract features
        features = model(images)
        # few-shot classifier
        probas, accuracy = fewshot_method(X=features, labels=labels, mode='val')
        q_labels = labels if len(labels) == len(probas) else labels[-len(probas):]
        # loss
        loss = criterion(probas, q_labels)
        metric_logger.update(loss=loss.detach().item(),
                             accuracy=accuracy)

    print("Averaged stats:", metric_logger)
    utils.wandb_log(
        {
            '{}/epoch'.format(set_name): epoch,
            '{}/loss'.format(set_name): metric_logger.loss.global_avg,
            '{}/accuracy'.format(set_name): metric_logger.accuracy.global_avg,
        }
    )
    return metric_logger.loss.global_avg, metric_logger.accuracy.global_avg


if __name__ == '__main__':
    main()
