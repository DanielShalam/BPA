import argparse
import torch
import utils
from self_optimal_transport import SOT


def get_args():
    """ Description: Parses arguments at command line. """
    parser = argparse.ArgumentParser()

    # global args
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='miniimagenet', choices=['miniimagenet'])
    parser.add_argument('--data_path', type=str, default='C:/Users/dani3/Documents/Datasets/few_shot/miniImagenet')
    parser.add_argument('--backbone', type=str, default='WRN')
    parser.add_argument('--method', type=str, default='pt_map', choices=['pt_map'])
    parser.add_argument('--num_workers', type=int, default=8)

    # wandb args
    parser.add_argument('--project', type=str, default='')
    parser.add_argument('--entity', type=str, default='')
    parser.add_argument('--wandb', type=utils.bool_flag, default=False)
    parser.add_argument('--log_step', type=utils.bool_flag, default=False)
    parser.add_argument('--log_epoch', type=utils.bool_flag, default=True)

    # few-shot args
    parser.add_argument('--train_way', type=int, default=5)
    parser.add_argument('--val_way', type=int, default=5)
    parser.add_argument('--num_shot', type=int, default=5)
    parser.add_argument('--num_query', type=int, default=15)
    parser.add_argument('--train_episodes', type=int, default=100)
    parser.add_argument('--eval_episodes', type=int, default=600)
    parser.add_argument('--test_episodes', type=int, default=6000)

    # train args
    parser.add_argument('--pretrained_path', type=str, default=None)
    parser.add_argument('--max_epochs', type=int, default=25)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--scheduler', type=str, default='step')
    parser.add_argument('--step_size', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--eval_freq', type=int, default=1)
    parser.add_argument('--eval_first', type=utils.bool_flag, default=False)
    parser.add_argument('--augment', type=utils.bool_flag, default=True)

    # SOT args
    parser.add_argument('--ot_reg', type=float, default=0.1)
    parser.add_argument('--sink_iters', type=int, default=5)
    parser.add_argument('--distance_metric', type=str, default='cosine')
    parser.add_argument('--sot', type=utils.bool_flag, default=True)
    parser.add_argument('--mask_diag', type=utils.bool_flag, default=True)
    parser.add_argument('--max_scale', type=utils.bool_flag, default=True)
    return parser.parse_args()


def main():
    args = get_args()
    print(vars(args))
    num_labeled_train = args.num_shot * args.train_way
    num_labeled_val = args.num_shot * args.val_way

    utils.set_seed(seed=args.seed)
    out_dir = utils.get_output_dir(args=args)

    # define datasets and loaders
    args.set_episodes = dict(train=args.train_episodes, val=args.eval_episodes, test=args.test_episodes)
    train_loader = utils.get_dataloader(set_name='train', args=args)
    val_loader = utils.get_dataloader(set_name='val', args=args)
    test_loader = utils.get_dataloader(set_name='test', args=args)

    # define model and load pretrained weights if available
    model = utils.get_model(model_name=args.backbone)
    model = model.cuda()
    model = utils.load_weights(model=model, path=args.pretrained_path)

    # define optimizer and scheduler
    optimizer = utils.get_optimizer(args=args, params=model.parameters())
    scheduler = utils.get_scheduler(args=args, optimizer=optimizer)

    # SOT and few-shot classification method (e.g. pt-map...)
    sot = SOT(distance_metric=args.distance_metric, ot_reg=args.ot_reg, mask_diag=args.mask_diag,
              sinkhorn_iterations=args.sink_iters)
    method = utils.get_method(args=args, sot=sot)

    train_labels = utils.get_fs_labels(num_way=args.train_way, num_query=args.num_query, num_shot=args.num_shot,
                                       method=args.method, to_cuda=True)
    val_labels = utils.get_fs_labels(num_way=args.val_way, num_query=args.num_query, num_shot=args.num_shot,
                                     method=args.method, to_cuda=True)

    # set logger and criterion
    logger = utils.get_logger(args=args)
    criterion = utils.get_criterion_by_method(method=args.method)

    # evaluate model before training/fine-tuning
    if args.eval_first:
        print("Evaluate model before training... ")
        eval_one_epoch(model, val_loader, method, criterion, val_labels, num_labeled_val, logger, args.log_step, 0)

    # main loop
    print("Start training...")
    best_loss = 1000
    best_acc = 0
    for epoch in range(1, args.max_epochs + 1):
        print(f"Epoch {epoch}/{args.max_epochs}: ")
        # train
        train_one_epoch(model, train_loader, optimizer, method, criterion, train_labels, num_labeled_train,
                        logger, args.log_step, epoch)
        if scheduler is not None:
            scheduler.step()

        # eval
        if epoch % args.eval_freq == 0:
            result = eval_one_epoch(model, val_loader, method, criterion, val_labels, num_labeled_val, logger,
                                    args.log_step, epoch)

            # save best model
            if result['val_loss'] < best_loss:
                best_loss = result['val_loss']
                torch.save(model.state_dict(), f'{out_dir}/checkpoint.pth')
            elif result['val_accuracy'] > best_acc:
                best_acc = result['val_accuracy']
                torch.save(model.state_dict(), f'{out_dir}/checkpoint.pth')

    print(f"Evaluating best model on test set for {args.test_episodes} episodes...")
    model = utils.load_weights(model=model, path=f'{out_dir}/checkpoint.pth')
    eval_one_epoch(model, test_loader, method, criterion, val_labels, num_labeled_val, logger, args.log_step, 0, 'test')


def train_one_epoch(model, loader, optimizer, method, criterion, labels, num_labeled, logger, log_step, epoch):
    model.train()
    epoch_result = dict(train_accuracy=0, train_loss=0)
    for batch_idx, batch in enumerate(loader):
        images, _ = batch
        images = images.cuda()
        optimizer.zero_grad()

        # apply few_shot method to get logits
        log_probas, accuracy, std = method(X=model(images, return_logits=False), labels=labels, mode='train')
        loss = criterion(log_probas[num_labeled:], labels[num_labeled:])
        epoch_result["train_loss"] += loss.item()
        epoch_result["train_accuracy"] += accuracy

        loss.backward()
        optimizer.step()

        if log_step:
            utils.log_step(results={'train_loss_step': loss.item(), 'train_accuracy_step': accuracy}, logger=logger)

    utils.print_and_log(results=epoch_result, n=len(loader), logger=logger, epoch=epoch)
    return epoch_result


def eval_one_epoch(model, loader, method, criterion, labels, num_labeled, logger, log_step, epoch, set_name='val'):
    model.eval()
    epoch_result = dict(val_accuracy=0, val_loss=0)
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            images, _ = batch
            images = images.cuda()

            # apply few_shot method to get logits
            log_probas, accuracy, std = method(X=model(images, return_logits=False), labels=labels, mode='val')
            loss = criterion(log_probas[num_labeled:], labels[num_labeled:])
            epoch_result[f"{set_name}_loss"] += loss.item()
            epoch_result[f"{set_name}_accuracy"] += accuracy

            if log_step:
                utils.log_step(results={f'{set_name}_loss_step': loss.item(), f'{set_name}_accuracy_step': accuracy},
                               logger=logger)

    utils.print_and_log(results=epoch_result, n=len(loader), logger=logger, epoch=epoch)
    return epoch_result


if __name__ == '__main__':
    main()
