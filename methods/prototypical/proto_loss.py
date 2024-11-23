import torch
from torch import nn

from bpa import BPA


class ProtoLoss(nn.Module):
    """
    Prototypical loss function
    """
    def __init__(self, args: dict, bpa: BPA = None):
        super().__init__()
        self.num_shot = args['num_shot']
        self.num_query = args['num_query']
        self.way_dict = dict(train=args['train_way'], val=args['val_way'])
        self.temperature = args['temperature']
        self.BPA = bpa
        self.num_labeled = None

    @staticmethod
    def get_accuracy(probas: torch.Tensor, labels: torch.Tensor):
        y_hat = probas.argmin(dim=-1)
        matches = labels.eq(y_hat).float()
        m = matches.mean().item()
        # pm = matches.std(unbiased=False).item() * 1.96
        return m

    def forward(self, X: torch.Tensor, labels: torch.Tensor, mode: str):
        num_way = self.way_dict[mode]
        self.num_labeled = num_way * self.num_shot

        # apply the BPA transform
        if self.BPA is not None:
            X = self.BPA(X, y=labels[:self.num_labeled])

        # split to support and queries
        X_support, X_query = X.split((self.num_labeled, X.size(0)-self.num_labeled), dim=0)

        # compute centroids
        # -> assuming input data sorted as [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, ...]
        X_centroid = X_support.reshape(self.num_shot, num_way, -1).transpose(0, 1).mean(dim=1)

        # compute distances between queries and the centroids
        D = (X_query.unsqueeze(1) - X_centroid.unsqueeze(0)).norm(dim=2).pow(2)
        D = D / self.temperature

        return -D, ProtoLoss.get_accuracy(D, labels)





