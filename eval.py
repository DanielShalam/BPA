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
    parser.add_argument('--pretrained_path', type=str, default='')
    parser.add_argument('--method', type=str, default='pt_map')
    parser.add_argument('--num_workers', type=int, default=8)

    # wandb args
    parser.add_argument('--project', type=str, default='')
    parser.add_argument('--entity', type=str, default='')
    parser.add_argument('--wandb', type=utils.bool_flag, default=False)
    parser.add_argument('--log_step', type=utils.bool_flag, default=False)

    # few-shot args
    parser.add_argument('--num_way', type=int, default=5)
    parser.add_argument('--num_shot', type=int, default=5)
    parser.add_argument('--num_query', type=int, default=15)
    parser.add_argument('--eval_episodes', type=int, default=10000)

    # SOT args
    parser.add_argument('--ot_reg', type=float, default=0.1)
    parser.add_argument('--distance_metric', type=str, default='cosine')
    parser.add_argument('--sot', type=utils.bool_flag, default=True)
    parser.add_argument('--mask_diag', type=utils.bool_flag, default=True)
    return parser.parse_args()


def main():
    args = get_args()
    print(vars(args))
    utils.set_seed(seed=args.seed)

    # define datasets and loaders
    test_loader = utils.get_dataloader(set_name='test', args=args)

    model = utils.get_model(model_name=args.backbone)
    model = model.cuda()
    model = utils.load_weights(model=model, path=args.pretrained_path)

    sot = SOT(distance_metric=args.distance_metric, ot_reg=args.ot_reg, mask_diag=args.mask_diag)
    method = utils.get_method(args=args, sot=sot)

    # setup logger
    logger = utils.get_logger(args=args)

    labels = utils.get_fs_labels(num_way=args.num_way, num_query=args.num_query, num_shot=args.num_shot,
                                 to_cuda=True)

    print(f"Evaluating model for {args.eval_episodes} episodes... ")
    model.eval()
    epoch_result = dict(test_accuracy=0)
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            images, _ = batch
            images = images.cuda()

            # apply few-shot method to get logits
            log_probas, accuracy, std = method(X=model(images, return_logits=False), labels=labels)
            epoch_result["test_accuracy"] += accuracy

            if args.log_step:
                utils.log_step(results={'test_accuracy_step': accuracy}, logger=logger)

            if batch_idx % 50 == 0:
                print(f"Episode {batch_idx + 1} / {args.eval_episodes},"
                      f" Accuracy={epoch_result['test_accuracy'] / (batch_idx + 1)}")

    utils.log_step(results={'test_accuracy': epoch_result['test_accuracy'] / args.num_episodes}, logger=logger)
    print(f"Test set results:\nAccuracy: {epoch_result['test_accuracy'] / args.num_episodes:.4f}")


if __name__ == '__main__':
    main()
