import torch
import math


class SOT(torch.nn.Module):
    supported_distances = ['cosine', 'euclidean']

    def __init__(self, distance_metric: str = 'euclidean', ot_reg: float = 0.1, sinkhorn_iterations: int = 5,
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
        :param X - N*D matrix where each row represent a feature vector.
        :param n_samples (optional) - The total number of samples for each class (n shots + n queries) if known.
        :param y_support (optional) - For few-shot classification. Support set labels (indexed as the first rows of X).
        :param max_temperature - Scale the transformed matrix to [0, 1]. usually helps.
        """

        # calculate the self-distance matrix according to the requested distance metric
        if self.distance_metric == 'euclidean':
            self_distances = torch.cdist(X, X, p=2)
            # scale euclidean distances to [0, 1], just to make strong difference between the distances and the diagonal
            self_distances = self_distances / self_distances.max(dim=-1, keepdim=True)[0]
        else:
            # cosine similarity -> dissimilarity
            self_distances = 1 - SOT.cosine_similarity(X)

        # sinkhorn calculation after making the diagonal
        log_P = self.compute_log_sinkhorn(m=self_distances.fill_diagonal_(self.diagonal_val))

        # taking
        if self.sigmoid:
            features = torch.sigmoid(log_P)
        else:
            features = torch.exp(log_P)

        # divide the transportation matrix by its maximum for better contrastive effect (usually helps)
        if max_temperature:
            max_probability = features.max().item()
            features = features / max_probability
        else:
            max_probability = 1

        # For few-shot classification -
        # if the support set labels are known, we can manually increase the probability
        # of support samples from the same class to match and decrease set all other support probabilities 0
        if y_support is not None:
            num_support = len(y_support)
            if self.positive_support_mask is None:
                self.set_support_mask(shape=(num_support, num_support), y_support=y_support, n_samples=n_samples)

            # Set both negative and positive support values to 0
            features[:num_support, :num_support] = self.positive_support_mask / max_probability

        # Filling the diagonal of the partition matrix (self-values)
        return features.fill_diagonal_(1)

    def compute_log_sinkhorn(self, m: torch.Tensor):
        """
        Log-space-sinkhorn algorithm for better stability.
        :param m - cost matrix
        :return - the distributional partition matrix of the SOT
        """
        # == torch.log(torch.ones(m.size()[0]))
        log_r = torch.zeros(m.size()[0]).to(m.device)
        log_c = torch.zeros(m.size()[1]).to(m.device)

        # Initialize dual variable v (u is implicitly defined in the loop)
        log_v = torch.zeros(m.size()[1]).to(m.device)  # ==torch.log(torch.ones(m.size()[1]))

        # Exponentiate the pairwise distance matrix
        log_K = -m / self.ot_reg

        # Main loop
        for i in range(self.sinkhorn_iterations):
            # Match r marginals
            log_u = log_r - SOT.log_sum_exp(log_K + log_v[None, :], dim=1)

            # Match c marginals
            log_v = log_c - SOT.log_sum_exp(log_u[:, None] + log_K, dim=0)

        # Compute optimal plan, cost, return everything
        log_P = log_u[:, None] + log_K + log_v[None, :]

        return log_P

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
        d_n = a / a.norm(dim=1)[:, None]
        C = torch.mm(d_n, d_n.transpose(0, 1))
        return C

    @staticmethod
    def log_sum_exp(u: torch.Tensor, dim: int):
        # Reduce log sum exp along axis
        u_max, __ = u.max(dim=dim, keepdim=True)
        log_sum_exp_u = torch.log(torch.exp(u - u_max).sum(dim)) + u_max.sum(dim)
        return log_sum_exp_u

    def set_support_mask(self, shape, y_support, n_samples):
        # precalculating the support-set known locations
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
                self.positive_support_mask[row, class_indexes] = support_val
