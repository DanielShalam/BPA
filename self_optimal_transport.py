import torch
import math


class SOT(torch.nn.Module):
    supported_distances = ['cosine', 'euclidean']

    def __init__(self, distance_metric: str = 'euclidean', ot_reg: float = 0.1, sinkhorn_iterations: int = 10,
                 sigmoid: bool = False, mask_diag: bool = True):
        """
        :param distance_metric - For Cost matrix calculation (currently optinal to [cosine, euclidean]).
        :param ot_reg - The entropy regularization (lambda).
        :param sinkhorn_iterations - Number of iteration for the sinkhorn algorithm.
        :param sigmoid - If to apply sigmoid(log_p) instead of the usual exp(log_p). Usually helps.
        """
        super().__init__()

        # check if the requested distance metric is available
        assert distance_metric.lower() in SOT.supported_distances
        assert sinkhorn_iterations > 0

        self.sinkhorn_iterations = sinkhorn_iterations
        self.distance_metric = distance_metric.lower()
        self.mask_diag = mask_diag
        self.sigmoid = sigmoid
        self.ot_reg = ot_reg
        self.diagonal_val = 1e3                         # value to mask self-values with
        self.pos_support_mask = None
        self.ways = None
        self.w_idx = None

    def get_M(self, X: torch.Tensor):
        # calculate M according to self.distance_metric
        if self.distance_metric == 'euclidean':
            M = torch.cdist(X, X, p=2)
            # scale euclidean distances to [0, 1], just to make strong difference between the distances and the diagonal
            return M / M.max()

        elif self.distance_metric == 'cosine':
            # cosine similarity -> dissimilarity
            return 1 - SOT.cosine_similarity(X)

    def forward(self, X: torch.Tensor, n_samples: int = 20, y_support: torch.Tensor = None,
                max_temperature: bool = True):
        """
        Calculating The SOT for X
        For few-shot learning setup (or any other task with some known labels):
        In case of few-shot classification, the support set needs to be on the upper rows.
        :param X - BxNxD (or NxD) matrix where each row represent a feature vector.
        :param n_samples (optional) - The total number of samples for each class (n shots + n queries) if known.
        :param y_support (optional) - For few-shot classification. Support set labels (indexed as the first rows of X).
        :param max_temperature - Scale the transformed matrix to [0, 1]. usually helps.
        """

        batched = X.dim() > 2
        # sinkhorn calculation after masking the diagonal
        M = self.get_M(X=X)
        M = self.mask_diagonal(M=M, value=self.diagonal_val)
        P_log = self.compute_log_sinkhorn(M=M, batched=batched)

        # taking
        if self.sigmoid:
            P = torch.sigmoid(P_log)
        else:
            P = torch.exp(P_log)

        # divide the transportation matrix by its maximum for better contrastive effect (usually helps)
        if max_temperature:
            P_max = P.max().item() if not batched else P.amax(dim=(1, 2), keepdim=True)
            P = P / P_max

        # For few-shot classification -
        # if the support set labels are known, we can manually increase the probability
        # of support samples from the same class to match and decrease set all other support probabilities 0
        # if y_support is not None:
        #     num_support = len(y_support)
        #     if self.pos_support_mask is None or num_support != self.pos_support_mask.shape[0]:
        #         # shape = (num_support, num_support)
        #         shape = (P.shape[1], P.shape[1])
        #         self.set_support_mask(shape=shape, y_support=y_support, n_samples=n_samples, bs=X.shape[0] if batched else 0)
        #
        #     # Set both negative and positive support values to 0
        #     if batched:
        #         P[self.pos_support_mask] = self.support_values[self.pos_support_mask]
        #         # support_scaled = self.support_values / P_max
        #     else:
        #         P[:num_support, :num_support] = self.pos_support_mask / P_max

        # set self-values to 1
        P = self.mask_diagonal(M=P, value=1)
        return P

    def mask_diagonal(self, M: torch.Tensor, value: float):
        if self.mask_diag:
            # Filling the diagonal of the partition matrix (self-values)
            if M.dim() > 2:
                M[torch.eye(M.shape[1]).repeat(M.shape[0], 1, 1).bool()] = value
            else:
                M.fill_diagonal_(value)
        return M

    def shot_balancing(self, M: torch.Tensor):
        if self.ways is not None:
            for i, w in enumerate(self.ways):
                M[self.w_idx[i][0]] = torch.min(M[self.w_idx[i][0]])
                # rows_idx = self.w_idx[i][0]
                # M[self.w_idx[i][0]] = torch.sum(M[self.w_idx[i][0]]) / len(rows_idx)
        return M

    def compute_log_sinkhorn(self, M: torch.Tensor, batched: bool = False, shot_balancing: bool = False):
        """
        Log-space-sinkhorn algorithm for better stability.
        :param shot_balancing:
        :param M - cost matrix
        :param batched - if to use the batched version of the OT.
        :return - the distributional partition matrix of the SOT
        """
        if batched:
            return self.compute_batched_log_sinkhorn(M=M)

        # Initialize dual variable v (u is implicitly defined in the loop)
        log_v = torch.zeros(M.size()[1]).to(M.device)  # ==torch.log(torch.ones(m.size()[1]))

        # Exponentiate the pairwise distance matrix
        log_K = -M / self.ot_reg

        # Main loop
        for i in range(self.sinkhorn_iterations):
            # Match r marginals
            log_u = - SOT.log_sum_exp(log_K + log_v[None, :], dim=1)
            # if shot_balancing:
            #     log_u = self.shot_balancing(M=log_u)

            # Match c marginals
            log_v = - SOT.log_sum_exp(log_u[:, None] + log_K, dim=0)
            # if shot_balancing:
            #     log_v = self.shot_balancing(M=log_v.T).T

        # Compute optimal plan, cost, return everything
        log_P = log_u[:, None] + log_K + log_v[None, :]

        # if shot_balancing:
        #     P = torch.exp(log_P)
        #     print(P[0, :].sum().item(), P[:, 0].sum().item())

        return log_P

    def compute_batched_log_sinkhorn(self, M):
        batch_size, x_points, _ = M.shape
        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).squeeze().to(M.device)
        nu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).squeeze().to(M.device)

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        def C(M, u, v, reg):
            """Modified cost for logarithmic updates"""
            return (-M + u.unsqueeze(-1) + v.unsqueeze(-2)) / reg

        # Sinkhorn iterations
        for i in range(self.sinkhorn_iterations):
            u1 = u  # useful to check the update
            u = self.ot_reg * (torch.log(mu + 1e-8) - torch.logsumexp(C(M, u, v, self.ot_reg), dim=-1)) + u
            v = self.ot_reg * (torch.log(nu + 1e-8) - torch.logsumexp(C(M, u, v, self.ot_reg).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        log_p = C(M, U, V, self.ot_reg)
        return log_p

    @staticmethod
    def euclidean_metric(a: torch.Tensor, b: torch.Tensor):
        n = a.shape[0]
        m = b.shape[0]
        a = a.unsqueeze(1).expand(n, m, -1)
        b = b.unsqueeze(0).expand(n, m, -1)
        return ((a - b) ** 2).sum(dim=2)

    @staticmethod
    def cosine_similarity(a: torch.Tensor, eps: float = 1e-8):
        # calculate the self-cosine similarity for a
        d_n = a / a.norm(dim=-1, keepdim=True)
        if len(a.shape) > 2:
            C = torch.bmm(d_n, d_n.transpose(1, 2))
        else:
            C = torch.mm(d_n, d_n.transpose(0, 1))
        return C

    @staticmethod
    def log_sum_exp(u: torch.Tensor, dim: int):
        # Reduce log sum exp along axis
        u_max, __ = u.max(dim=dim, keepdim=True)
        log_sum_exp_u = torch.log(torch.exp(u - u_max).sum(dim)) + u_max.sum(dim)
        return log_sum_exp_u

    def set_support_mask(self, shape: tuple, bs: int, y_support: torch.Tensor, n_samples: int):
        # precalculate the support-set known locations (indexes)
        support_mask = torch.zeros(shape).cuda()
        support_values = torch.zeros(shape).cuda()
        num_classes = len(torch.unique(y_support))

        # calculating their relative part according to the n_samples
        if self.sigmoid:
            support_val = torch.sigmoid(torch.log(torch.tensor(1 / n_samples)))
        else:
            support_val = 1 / n_samples

        support_values += support_val

        for cls in range(num_classes):
            # gather class and non-class indexes
            class_indexes = torch.where(y_support == cls)[0]
            for idx in class_indexes:
                support_mask[idx, class_indexes] = 1
                support_mask[idx, idx] = 0

        self.pos_support_mask = support_mask.bool()
        self.support_values = support_values
        # assuming ordered shots
        if bs > 0:
            self.pos_support_mask = self.pos_support_mask.unsqueeze(0).expand(bs, -1, -1)
            self.support_values = self.support_values.unsqueeze(0).expand(bs, -1, -1)
