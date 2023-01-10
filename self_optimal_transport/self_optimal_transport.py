import torch
from self_optimal_transport import ot


class SOT(object):
    supported_distances = ['cosine', 'euclidean']

    def __init__(self, distance_metric: str = 'cosine', ot_reg: float = 0.1, sinkhorn_iterations: int = 10,
                 sigmoid: bool = False, mask_diag: bool = True, max_scale: bool = True):
        """
        :param distance_metric - Compute the cost matrix.
        :param ot_reg - Sinkhorn entropy regularization (lambda). For few-shot classification, 0.1-0.2 works best.
        :param sinkhorn_iterations - Maximum number of sinkhorn iterations.
        :param sigmoid - If to apply sigmoid(log_p) instead of the usual exp(log_p).
        :param mask_diag - Set to true to apply diagonal masking before and after the OT.
        :param max_scale - Re-scale the SOT values to range [0,1].
        """
        super().__init__()

        assert distance_metric.lower() in SOT.supported_distances and sinkhorn_iterations > 0

        self.sinkhorn_iterations = sinkhorn_iterations
        self.distance_metric = distance_metric.lower()
        self.mask_diag = mask_diag
        self.sigmoid = sigmoid
        self.ot_reg = ot_reg
        self.max_scale = max_scale
        self.diagonal_val = 1e3                         # value to mask self-values with

    def compute_cost(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute cost matrix.
        """
        if self.distance_metric == 'euclidean':
            M = torch.cdist(X, X, p=2)
            # scale euclidean distances to [0, 1]
            return M / M.max()

        elif self.distance_metric == 'cosine':
            # cosine distance
            return 1 - SOT.cosine_similarity(X)

    def mask_diagonal(self, M: torch.Tensor, value: float):
        """
        Set new value at a diagonal matrix.
        """
        if self.mask_diag:
            if M.dim() > 2:
                M[torch.eye(M.shape[1]).repeat(M.shape[0], 1, 1).bool()] = value
            else:
                M.fill_diagonal_(value)
        return M

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute the SOT features for X
        """
        # get masked cost matrix
        C = self.compute_cost(X=X)
        M = self.mask_diagonal(C, value=self.diagonal_val)

        # compute self-OT
        z_log = ot.log_sinkhorn(M=M, reg=self.ot_reg, num_iters=self.sinkhorn_iterations)

        if self.sigmoid:
            z = torch.sigmoid(z_log)
        else:
            z = torch.exp(z_log)

        # divide the SOT matrix by its max to scale it up
        if self.max_scale:
            z_max = z.max().item() if z.dim() <= 2 else z.amax(dim=(1, 2), keepdim=True)
            z = z / z_max

        # set self-values to 1
        return self.mask_diagonal(z, value=1)

    @staticmethod
    def cosine_similarity(a: torch.Tensor, eps: float = 1e-8):
        """
        Compute the pairwise cosine similarity between a matrix to itself.
        """
        d_n = a / a.norm(dim=-1, keepdim=True)
        if len(a.shape) > 2:
            C = torch.bmm(d_n, d_n.transpose(1, 2))
        else:
            C = torch.mm(d_n, d_n.transpose(0, 1))
        return C