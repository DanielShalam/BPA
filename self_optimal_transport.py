import torch


class SOT(torch.nn.Module):
    supported_distances = ['cosine', 'euclidean']

    def __init__(self, distance_metric: str = 'cosine', ot_reg: float = 0.1, sinkhorn_iterations: int = 10,
                 sigmoid: bool = True):
        """
        :param distance_metric - For Cost matrix calculation (currently optinal to [cosine, euclidean]).
        :param ot_reg - The entropy regularization (lambda).
        :param sinkhorn_iterations - Number of iteration for the sinkhorn algorithm.
        :param sigmoid - Sometimes it helps to take the sigmoid of the log partitions instead the exponent.
        """
        super().__init__()

        # check if the requested distance metric is available
        assert distance_metric in SOT.supported_distances
        assert sinkhorn_iterations > 0

        self.sinkhorn_iterations = sinkhorn_iterations
        self.distance_metric = distance_metric
        self.diagonal_val = 1e3
        self.sigmoid = sigmoid
        self.ot_reg = ot_reg

    def transform(self, X: torch.Tensor, n_classes: int = 5, n_samples: int = 20, support_labels: torch.Tensor = None,
                  max_temperature: bool = True):
        """
        Calculating The SOT for X

        For few-shot learning setup (or any other task with some known labels):
        In case of few-shot classification, the support set needs to be on the upper rows.

        :param X - N*D matrix where each row represent a feature vector.
        :param n_classes - The number of classes in the batch.
        :param n_samples - The total number of samples for each class (n shots + n queries) if known.
        :param support_labels -  For few-shot classification, support set labels (indexed as the first rows of X).
        :param max_temperature - Scale the transformed matrix to [0, 1], usually helps.
        """

        # calculate the self-distance matrix according to the requested distance metric
        if self.distance_metric == 'euclidean':
            self_distances = torch.cdist(X, X, p=2)
            # scale euclidean distances to [0, 1], just to make strong difference between the distances and the diagonal
            self_distances = self_distances / self_distances.max(dim=-1, keepdim=True)[0]

        elif self.distance_metric == 'cosine':
            # cosine similarity -> dissimilarity
            self_distances = 1 - SOT.cosine_similarity(X)

        # sinkhorn calculation after making the diagonal
        P, log_P = self.compute_log_sinkhorn(m=self_distances.fill_diagonal_(self.diagonal_val))

        # sometimes taking the sigmoid log partition can increase results. the support value may be needed later
        if self.sigmoid:
            features = torch.sigmoid(log_P.clone())
            support_val = torch.sigmoid(torch.log(torch.tensor(1 / n_samples))).to(X.device)
        else:
            features = P.clone()
            support_val = 1 / n_samples

        # divide the matrix by its maximum to make the maximum value 1 (usually helps)
        if max_temperature:
            max_probability = features.max().detach()
            features = features / max_probability
        else:
            max_probability = 1

        # For few-shot classification -
        # if the support set labels are known, we can manually increase the probability
        # of support samples from the same class to match the other samples and set all other support to 0
        if support_labels is not None:

            for cls in range(n_classes):
                # gather class and non-class indexes
                class_indexes = torch.where(support_labels == cls)[0]
                non_class_indexes = torch.where(support_labels != cls)[0]

                # assuming ordered shots
                for row in class_indexes:
                    features[row, class_indexes] = support_val / max_probability
                    features[row, non_class_indexes] = 0

        # Filling the diagonal of the partition matrix (self-values)
        return features.fill_diagonal_(1)

    def compute_log_sinkhorn(self, m: torch.Tensor):
        """
        Log-space-sinkhorn algorithm for better stability.
        :param m - The cost matrix.
        :return - The distributional(log) partition matrix of the SOT.
        """

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

        return torch.exp(log_P), log_P

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


