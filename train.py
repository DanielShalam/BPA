import argparse
import torch
from torch import nn
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
    parser.add_argument('--method', type=str, default='pt_map')

    # few-shot args
    parser.add_argument('--num_way', type=int, default=5)
    parser.add_argument('--num_shot', type=int, default=5)
    parser.add_argument('--num_query', type=int, default=15)

    # train args
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--train_episodes', type=int, default=100)
    parser.add_argument('--eval_episodes', type=int, default=600)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--use_sot', type=utils.bool_flag, default=True)
    parser.add_argument('--eval_freq', type=int, default=1)

    # SOT args
    parser.add_argument('--ot_reg', type=float, default=0.1)
    parser.add_argument('--distance_metric', type=str, default='cosine')

    return parser.parse_args()


def main():
    args = get_args()
    print(args)

    # define datasets and loaders
    train_loader = utils.get_dataloader(set_name='train', args=args)
    val_loader = utils.get_dataloader(set_name='val', args=args)

    model = utils.get_model(model_name=args.backbone)
    model = model.cuda()

    optimizer = utils.get_optimizer(params=model.parameters(), optimizer=args.optimizer, lr=args.lr)

    # few-shot classification method (e.g. pt-map...)
    sot = SOT(distance_metric=args.distance_metric, ot_reg=args.ot_reg)
    method = utils.get_method(args=args, sot=sot)

    criterion = nn.NLLLoss()

    query_labels, labels = utils.get_fs_labels(num_way=args.num_way, num_query=args.num_query, num_shot=args.num_shot)
    labels = labels.cuda()
    query_labels = query_labels.cuda()

    # main loop
    for epoch in range(args.max_epochs):

        # train
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            images, _ = batch
            images = images.cuda()
            optimizer.zero_grad()

            # apply few_shot method to get logits
            accuracy, log_probas = method(X=model(images, return_logits=False), labels=labels)

            loss = criterion(log_probas, query_labels)

            loss.backward()
            optimizer.step()

            print(f'Epoch: {epoch}, Accuracy: {accuracy:.4f}, Loss: {loss.item():.4f}')

        if epoch % args.eval_freq != 0:
            continue

        # evaluate every eval_freq epochs
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                images, _ = batch
                images = images.cuda()

                # apply few_shot method to get logits
                accuracy, log_probas = method(X=model(images, return_logits=False), labels=labels)
                loss = criterion(log_probas, query_labels)

                print(f'Epoch: {epoch}, Accuracy: {accuracy:.4f}, Loss: {loss.item():.4f}')


if __name__ == '__main__':
    main()
