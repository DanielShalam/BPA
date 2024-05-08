import argparse
import os
from time import time
import torch

from self_optimal_transport import SOT
import utils


def get_args():
    parser = argparse.ArgumentParser()

    # global args
    parser.add_argument('--root_path', type=str, default='./',
                        help=""" Path to project root directory. """)

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='miniimagenet', choices=['miniimagenet', 'cifar'])
    parser.add_argument('--data_path', type=str, default='./datasets/few_shot/miniimagenet')
    parser.add_argument('--method', type=str, default='pt_map_sot',
                        choices=['pt_map', 'pt_map_sot', 'proto', 'proto_sot'],
                        help="Specify the few shot method to use. ")
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--eval', type=utils.bool_flag, default=False,
                        help=""" Set to true if you want to evaluate trained model on test set. """)
    parser.add_argument('--eval_freq', type=int, default=1,
                        help=""" Evaluate training every n epochs. """)
    parser.add_argument('--eval_first', type=utils.bool_flag, default=False,
                        help=""" Set to true to evaluate the model before training. Useful for fine-tuning. """)

    # wandb args
    parser.add_argument('--wandb', type=utils.bool_flag, default=False, help=""" Log data into wandb. """)
    parser.add_argument('--project', type=str, default='', help=""" Project name in wandb. """)
    parser.add_argument('--entity', type=str, default='', help=""" Your wandb entity name. """)
    parser.add_argument('--log_step', type=utils.bool_flag, default=False, help=""" Log training steps. """)
    parser.add_argument('--log_epoch', type=utils.bool_flag, default=True, help=""" Log epoch. """)

    # few-shot args
    parser.add_argument('--train_way', type=int, default=5, help=""" Number of classes in training batches. """)
    parser.add_argument('--val_way', type=int, default=5, help=""" Number of classes in validation/testing batches. """)
    parser.add_argument('--num_shot', type=int, default=5, help=""" Support size. """)
    parser.add_argument('--num_query', type=int, default=15, help=""" Query size. """)
    parser.add_argument('--train_episodes', type=int, default=200, help=""" Number of episodes for each epoch. """)
    parser.add_argument('--eval_episodes', type=int, default=400, help=""" Number of tasks to evaluate. """)
    parser.add_argument('--test_episodes', type=int, default=10000, help=""" Number of tasks to evaluate. """)

    # training args
    parser.add_argument('--max_epochs', type=int, default=25)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--scheduler', type=str, default='step')
    parser.add_argument('--step_size', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--augment', type=utils.bool_flag, default=True, help=""" Apply data augmentation. """)

    # model args
    parser.add_argument('--backbone', type=str, default='WRN')
    parser.add_argument('--pretrained_path', type=str, default=False,
                        help=""" Path to pretrained model. For testing/fine-tuning. """)
    parser.add_argument('--temperature', type=float, default=0.1, help=""" Temperature for ProtoNet. """)
    parser.add_argument('--dropout', type=float, default=0., help=""" Dropout probability. """)

    # SOT args
    parser.add_argument('--ot_reg', type=float, default=0.1,
                        help=""" Sinkhorn entropy regularization. 
                                 For few-shot methods, 0.1-0.2 seems to work best. 
                                 For larger tasks (~10,000) samples, try to increase this value. """)
    parser.add_argument('--sink_iters', type=int, default=20,
                        help=""" Number of Sinkhorn iterations. 
                                 Usually small number (~ 5-10) is sufficient. """)
    parser.add_argument('--distance_metric', type=str, default='cosine',
                        help=""" Build the cost matrix. """, choices=['cosine', 'euclidean'])
    parser.add_argument('--mask_diag', type=utils.bool_flag, default=True,
                        help=""" If true, apply mask diagonal values before and after the OT. """)
    parser.add_argument('--max_scale', type=utils.bool_flag, default=True,
                        help=""" Scaling range of the SOT values to [0,1]. """)
    return parser.parse_args()


def main():
    args = get_args()
    print(vars(args))

    utils.set_seed(seed=args.seed)
    out_dir = utils.get_output_dir(args=args)

    # define datasets and loaders
    args.set_episodes = dict(train=args.train_episodes, val=args.eval_episodes, test=args.test_episodes)
    if not args.eval:
        train_loader = utils.get_dataloader(set_name='train', args=args, constant=False)
        val_loader = utils.get_dataloader(set_name='val', args=args, constant=True)
    else:
        train_loader = None
        val_loader = utils.get_dataloader(set_name='test', args=args, constant=False)

    # define model and load pretrained weights if available
    model = utils.get_model(args.backbone, args)
    model = model.cuda()
    utils.load_weights(model, args.pretrained_path)

    # define optimizer and scheduler
    optimizer = utils.get_optimizer(args=args, params=model.parameters())
    scheduler = utils.get_scheduler(args=args, optimizer=optimizer)

    # SOT and few-shot classification method (e.g. pt-map...)
    sot = None
    if 'sot' in args.method.lower():
        sot = SOT(distance_metric=args.distance_metric, ot_reg=args.ot_reg, mask_diag=args.mask_diag,
                  sinkhorn_iterations=args.sink_iters, max_scale=args.max_scale)

    method = utils.get_method(args=args, sot=sot)

    # few-shot labels
    train_labels = utils.get_fs_labels(args.method, args.train_way, args.num_query, args.num_shot)
    val_labels = utils.get_fs_labels(args.method, args.val_way, args.num_query, args.num_shot)

    # set logger and criterion
    criterion = utils.get_criterion_by_method(method=args.method)
    logger = utils.get_logger(exp_name=out_dir.split('/')[-1], args=args)

    # only evaluate
    if args.eval:
        print(f"Evaluate model for {args.test_episodes} episodes... ")
        eval_one_epoch(model, val_loader, method, criterion, val_labels, logger, 0, set_name='test')
        exit(1)

    # evaluate model before training
    if args.eval_first:
        print("Evaluate model before training... ")
        eval_one_epoch(model, val_loader, method, criterion, val_labels, logger, 0, set_name='val')

    # main loop
    print("Start training...")
    best_loss = 1000
    best_acc = 0
    for epoch in range(1, args.max_epochs + 1):
        print(f"Epoch {epoch}/{args.max_epochs}: ")
        # train
        train_one_epoch(model, train_loader, optimizer, method, criterion, train_labels, logger, args.log_step, epoch)
        if scheduler is not None:
            scheduler.step()

        # eval
        if epoch % args.eval_freq == 0:
            result = eval_one_epoch(model, val_loader, method, criterion, val_labels, logger, epoch)

            # save best model
            if result['val/loss'] < best_loss:
                best_loss = result['val/loss']
                torch.save(model.state_dict(), os.path.join(out_dir, f'{epoch}_min_loss.pth'))
            elif result['val/accuracy'] > best_acc:
                best_acc = result['val/accuracy']
                torch.save(model.state_dict(), os.path.join(out_dir, f'{epoch}_max_acc.pth'))

        torch.save(model.state_dict(), os.path.join(out_dir, 'checkpoint_last.pth'))


def train_one_epoch(model, loader, optimizer, method, criterion, labels, logger, log_step, epoch):
    model.train()
    results = {'train/accuracy': 0, 'train/loss': 0}
    start = time()
    for batch_idx, batch in enumerate(loader):
        images = batch[0].cuda()

        features = model(images)
        # apply few_shot method
        probas, accuracy = method(features, labels=labels, mode='train')
        q_labels = labels if len(labels) == len(probas) else labels[-len(probas):]

        loss = criterion(probas, q_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        results["train/loss"] += loss.item()
        results["train/accuracy"] += accuracy

        if log_step and (batch_idx + 1) % 50 == 0:
            step = batch_idx+((epoch-1) * len(loader))
            utils.log_step(
                results={'train/loss_step': loss.item(), 'train/accuracy_step': accuracy, 'train/train_step': step},
                logger=logger
            )

    results["train/time"] = time() - start
    results["train/epoch"] = epoch
    utils.print_and_log(results=results, n=len(loader), logger=logger)
    return results


@torch.no_grad()
def eval_one_epoch(model, loader, method, criterion, labels, logger, epoch, set_name='val'):
    model.eval()
    results = {f'{set_name}/accuracy': 0, f'{set_name}/loss': 0}

    for batch_idx, batch in enumerate(loader):
        images = batch[0].cuda()

        features = model(images)

        # apply few_shot method
        probas, accuracy = method(X=features, labels=labels, mode='val')
        q_labels = labels if len(labels) == len(probas) else labels[-len(probas):]

        loss = criterion(probas, q_labels)

        results[f"{set_name}/loss"] += loss.item()
        results[f"{set_name}/accuracy"] += accuracy

        if batch_idx % 50 == 0:
            step = batch_idx+((epoch-1) * len(loader))
            print(f"Batch {batch_idx + 1}/{len(loader)}: ")
            utils.log_step(
                results={f'{set_name}/loss_step': loss.item(), f'{set_name}/accuracy_step': accuracy,
                         f'{set_name}/{set_name}_step': step},
                logger=logger
            )

    results["val/epoch"] = epoch
    utils.print_and_log(results=results, n=len(loader), logger=logger)
    return results


if __name__ == '__main__':
    main()
