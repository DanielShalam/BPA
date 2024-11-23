import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from bpa import ot


class BPA(nn.Module):
    supported_distances = ['cosine', 'euclidean']

    def __init__(self,
                 distance_metric: str = 'cosine',
                 ot_reg: float = 0.1,
                 sinkhorn_iterations: int = 10,
                 sigmoid: bool = False,
                 mask_diag: bool = True,
                 max_scale: bool = True):
        """
        :param distance_metric - Compute the cost matrix.
        :param ot_reg - Sinkhorn entropy regularization (lambda). For few-shot classification, 0.1-0.2 works best.
        :param sinkhorn_iterations - Maximum number of sinkhorn iterations.
        :param sigmoid - If to apply sigmoid(log_p) instead of the usual exp(log_p).
        :param mask_diag - Set to true to apply diagonal masking before and after the OT.
        :param max_scale - Re-scale the BPA values to range [0,1].
        """
        super().__init__()

        assert distance_metric.lower() in BPA.supported_distances and sinkhorn_iterations > 0

        self.sinkhorn_iterations = sinkhorn_iterations
        self.distance_metric = distance_metric.lower()
        self.mask_diag = mask_diag
        self.sigmoid = sigmoid
        self.ot_reg = ot_reg
        self.max_scale = max_scale
        self.diagonal_val = 1e5                         # value to mask self-values with

    def mask_diagonal(self, M: Tensor, value: float):
        """
        Fill the diagonal of a given matrix (or a batch of them) with given value
        """
        if self.mask_diag:
            if M.dim() > 2:
                M[torch.eye(M.shape[1]).repeat(M.shape[0], 1, 1).bool()] = value
            else:
                M.fill_diagonal_(value)
        return M

    def adjust_labeled(self, x: Tensor, y: Tensor):
        """
        Adjust BPA scores using additional labels (e.g. support set in few shot classification)
        We do so by filling the final values of labeled pairs by 0 and 1, according to if they share the same class
        """
        labels_one_hot = F.one_hot(y).float()
        mask = (labels_one_hot @ labels_one_hot.T).bool()   # mask[i,j] 1 if y[i] == y[j]

        # pad mask with ones
        pad_size = x.size(0) - mask.size(0)
        pad = (0, pad_size, 0, pad_size)
        # (padding_left, padding_right, padding_top, padding_bottom)
        x.masked_fill_(F.pad(mask, pad, "constant", 0), value=1)  # mask known positives
        x.masked_fill_(F.pad(~mask, pad, "constant", 0), value=0)   # mask known negatives
        return x

    def compute_cost_matrix(self, x: Tensor) -> Tensor:
        """
        Compute the cost matrix under euclidean or cosine distances
        """
        # Euclidean distances
        if self.distance_metric == 'euclidean':
            # dim_offset = 0 if x.dim() <= 2 else 1
            # pairwise_dist = (x.unsqueeze(1+dim_offset) - x.unsqueeze(0+dim_offset)).norm(dim=-1).pow(2)
            pairwise_dist = torch.cdist(x, x, p=2)
            pairwise_dist = pairwise_dist / pairwise_dist.max()  # scale distances to [0, 1]

        # Cosine distances
        elif self.distance_metric == 'cosine':
            x_norm = F.normalize(x, dim=-1, p=2)
            pairwise_dist = 1 - (x_norm @ x_norm.transpose(-2, -1))
        return pairwise_dist

    def forward(self, x: Tensor, y: Tensor = None) -> Tensor:
        """
        Compute the BPA feature transform
        """
        # get masked cost matrix
        C = self.compute_cost_matrix(x)
        C = self.mask_diagonal(C, value=self.diagonal_val)

        # compute self-OT
        x_bpa = ot.log_sinkhorn(C, reg=self.ot_reg, num_iters=self.sinkhorn_iterations)
        if self.sigmoid:
            x_bpa = torch.sigmoid(x_bpa)
        else:
            x_bpa = torch.exp(x_bpa)

        # divide the BPA matrix by its maximum value to scale its range into [0, 1]
        if self.max_scale:
            z_max = x_bpa.max().item() if x_bpa.dim() <= 2 else x_bpa.amax(dim=(1, 2), keepdim=True)
            x_bpa = x_bpa / z_max

        # adjust labeled samples (e.g. support set) if given labels
        if y is not None:
            x_bpa = self.adjust_labeled(x_bpa, y)

        # set self-values to 1
        return self.mask_diagonal(x_bpa, value=1)

