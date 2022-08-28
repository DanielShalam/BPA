import torch
import math


class SOT(torch.nn.Module):
    supported_distances = ['cosine', 'euclidean']

    def __init__(self, distance_metric: str = 'euclidean', ot_reg: float = 0.1, sinkhorn_iterations: int = 10,
                 sigmoid: bool = True):
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
        self.sigmoid = sigmoid
        self.ot_reg = ot_reg
        self.diagonal_val = 1e3                         # value to mask self-values with
        self.positive_support_mask = None

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

        batched = True if len(X.shape) > 2 else False
        # calculate the self-distance matrix according to the requested distance metric
        if self.distance_metric == 'euclidean':
            M = torch.cdist(X, X, p=2)
            # scale euclidean distances to [0, 1], just to make strong difference between the distances and the diagonal
            M = M / M.max(dim=-1, keepdim=True)[0]
        else:
            # cosine similarity -> dissimilarity
            M = 1 - SOT.cosine_similarity(X)

        # sinkhorn calculation after making the diagonal
        if batched:
            diag_mask = torch.eye(X.shape[1]).repeat(X.shape[0], 1, 1).bool()
            M[diag_mask] = self.diagonal_val
            log_P = self.compute_batched_log_sinkhorn(M=M)
        else:
            log_P = self.compute_log_sinkhorn(M=M.fill_diagonal_(self.diagonal_val))

        # taking
        if self.sigmoid:
            features = torch.sigmoid(log_P)
        else:
            features = torch.exp(log_P)

        # divide the transportation matrix by its maximum for better contrastive effect (usually helps)
        if max_temperature:
            max_probability = features.max().item() if not batched else features.amax(dim=(1, 2), keepdim=True)
            features = features / max_probability
        else:
            max_probability = 1

        # For few-shot classification -
        # if the support set labels are known, we can manually increase the probability
        # of support samples from the same class to match and decrease set all other support probabilities 0
        if y_support is not None:
            num_support = len(y_support)
            if self.positive_support_mask is None:
                shape = (num_support, num_support) if not batched else (X.shape[0], num_support, num_support)
                self.set_support_mask(shape=shape, y_support=y_support, n_samples=n_samples)

            # Set both negative and positive support values to 0
            features[:num_support, :num_support] = self.positive_support_mask / max_probability

        # Filling the diagonal of the partition matrix (self-values)
        if batched:
            features[diag_mask] = 1
            return features

        return features.fill_diagonal_(1)

    def compute_log_sinkhorn(self, M: torch.Tensor):
        """
        Log-space-sinkhorn algorithm for better stability.
        :param M - cost matrix
        :return - the distributional partition matrix of the SOT
        """
        # == torch.log(torch.ones(m.size()[0]))
        log_r = torch.zeros(M.size()[0]).to(M.device)
        log_c = torch.zeros(M.size()[1]).to(M.device)

        # Initialize dual variable v (u is implicitly defined in the loop)
        log_v = torch.zeros(M.size()[1]).to(M.device)  # ==torch.log(torch.ones(m.size()[1]))

        # Exponentiate the pairwise distance matrix
        log_K = -M / self.ot_reg

        # Main loop
        for i in range(self.sinkhorn_iterations):
            # Match r marginals
            log_u = log_r - SOT.log_sum_exp(log_K + log_v[None, :], dim=1)

            # Match c marginals
            log_v = log_c - SOT.log_sum_exp(log_u[:, None] + log_K, dim=0)

        # Compute optimal plan, cost, return everything
        log_P = log_u[:, None] + log_K + log_v[None, :]

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

    def set_support_mask(self, shape, y_support, n_samples):
        # precalculate the support-set known locations (indexes)
        self.positive_support_mask = torch.zeros(shape).cuda()
        num_classes = len(torch.unique(y_support))
        # calculating their relative part according to the n_samples
        x = 1 / n_samples
        support_val = 1 / (1 + math.exp(-math.log(x))) if self.sigmoid else x

        for cls in range(num_classes):
            # gather class and non-class indexes
            class_indexes = torch.where(y_support == cls)[0]

            # assuming ordered shots
            for row in class_indexes:
                if len(shape) > 2:
                    self.positive_support_mask[:, row, class_indexes] = support_val
                else:
                    self.positive_support_mask[row, class_indexes] = support_val