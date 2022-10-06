import torch
import math

from torch import nn
import torch.nn.functional as F


def centerDatas(X: torch.Tensor, n_lsamples: int):
    """
    Center labeled and un-labeled data separately.
    """
    X[:n_lsamples, :] = X[:n_lsamples, :] - X[:n_lsamples, :].mean(0, keepdim=True)
    X[n_lsamples:, :] = X[n_lsamples:, :] - X[n_lsamples:, :].mean(0, keepdim=True)
    return X


def QRreduction(X: torch.Tensor):
    X = torch.linalg.qr(X.permute(0, 2, 1)).R
    X = X.permute(0, 2, 1)
    return X


# ---------  GaussianModel

class GaussianModel:
    def __init__(self, num_way: int, num_shot: int, num_query: int, lam: float, SOT):
        self.num_way = num_way
        self.num_shot = num_shot
        self.num_query = num_query
        self.n_lsamples = num_way * num_shot
        self.n_usamples = num_way * num_query
        self.SOT = SOT
        self.lam = lam
        self.mus = None  # shape [n_ways][feat_dim]

    def cuda(self):
        self.mus = self.mus.cuda()

    def init_from_labelled(self, X: torch.Tensor):
        self.mus = X.reshape(self.num_shot + self.num_query, self.num_way, -1)[:self.num_shot, ].mean(0)

    def update_from_estimate(self, estimate, alpha):
        Dmus = estimate - self.mus
        self.mus = self.mus + alpha * Dmus

    def compute_optimal_transport(self, M: torch.Tensor, r: torch.Tensor, c: torch.Tensor, epsilon: float = 1e-6):
        n, m = M.shape
        P = torch.exp(-self.lam * M)
        P /= P.sum()

        u = torch.zeros(n, device='cuda')

        maxiters = 1000
        iters = 1
        # normalize this matrix
        while torch.max(torch.abs(u - P.sum(-1))) > epsilon:
            u = P.sum(dim=-1)
            P *= (r / u).view((-1, 1))
            P *= (c / P.sum(dim=0, keepdim=True)).view((1, -1))
            if iters == maxiters:
                break
            iters += 1

        return P

    def get_probas(self, X: torch.Tensor, labels: torch.Tensor):
        dist = torch.cdist(X, self.mus)

        p_xj = torch.zeros_like(dist)
        p_xj_test = self.SOT.compute_log_sinkhorn(M=dist[self.n_lsamples:, :])
        p_xj[self.n_lsamples:] = p_xj_test

        p_xj[:self.n_lsamples].fill_(0)
        p_xj[:self.n_lsamples].scatter_(1, labels[:self.n_lsamples].unsqueeze(1), 1)

        return p_xj

    def estimate_from_mask(self, X: torch.Tensor, mask: torch.Tensor):
        emus = mask.T.matmul(X).div(mask.sum(dim=0).unsqueeze(1))
        return emus


# =========================================
#    MAP
# =========================================

class MAP:
    def __init__(self, labels, alpha: float, num_labeled: int, n_runs=1):
        self.labels = labels
        self.alpha = alpha
        self.num_labeled = num_labeled
        self.n_runs = n_runs

    def get_accuracy(self, probas: torch.Tensor):
        y_hat = probas.argmax(dim=-1)
        matches = self.labels.eq(y_hat).float()
        acc_test = matches[self.num_labeled:].mean(0)

        m = acc_test.mean().item()
        pm = acc_test.std(unbiased=False).item() * 1.96
        return m, pm

    def perform_epoch(self, model: GaussianModel, X: torch.Tensor):
        p_xj = model.get_probas(X=X, labels=self.labels)
        m_estimates = model.estimate_from_mask(X=X, mask=torch.exp(p_xj))

        # update centroids
        model.update_from_estimate(m_estimates, self.alpha)

    def loop(self, X: torch.Tensor, model: GaussianModel, n_epochs: int = 20):
        for epoch in range(1, n_epochs + 1):
            self.perform_epoch(model=model, X=X)

        # get final accuracy and return it
        P = model.get_probas(X=X, labels=self.labels)
        return P


class PT_MAP(nn.Module):
    def __init__(self, args: dict, lam: float = 10, alpha: float = 0.2, sot=None):
        super().__init__()
        self.num_way = args['num_way']
        self.num_shot = args['num_shot']
        self.num_query = args['num_query']
        self.num_labeled = self.num_way * self.num_shot
        self.lam = lam
        self.alpha = alpha
        # if sot is None, regular pt-map will be used (without applying sot)
        self.SOT = sot

    def scale(self, X):
        # normalize, center and normalize again
        X = F.normalize(X, p=2, dim=-1)
        X = centerDatas(X, self.num_labeled)
        X = F.normalize(X, p=2, dim=-1)
        return X

    def forward(self, X: torch.Tensor, labels: torch.Tensor):
        # power transform (PT part)
        X = torch.pow(X + 1e-6, 0.5)

        # scale data according to pt-map
        Z = self.scale(X=X)

        # applying SOT or continue with regular pt-map
        if self.SOT is not None:
            Z = self.SOT(X=Z, n_samples=self.num_shot + self.num_query, y_support=labels[:self.num_labeled])

        # MAP
        gaussian_model = GaussianModel(num_way=self.num_way, num_shot=self.num_shot, num_query=self.num_query,
                                       lam=self.lam, SOT=self.SOT)
        gaussian_model.init_from_labelled(X=Z)

        optim = MAP(labels=labels, alpha=self.alpha, num_labeled=self.num_labeled)
        P = optim.loop(X=Z, model=gaussian_model, n_epochs=20)
        accuracy, std = optim.get_accuracy(probas=P)

        return P, accuracy, std
