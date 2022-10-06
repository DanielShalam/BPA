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
    def __init__(self, num_way: int, num_shot: int, num_query: int, lam: float, num_runs: int = 1):
        self.num_way = num_way
        self.num_shot = num_shot
        self.num_query = num_query
        self.num_runs = num_runs
        self.n_lsamples = num_way * num_shot
        self.n_usamples = num_way * num_query
        self.lam = lam

        self.mus = None  # shape [n_runs][n_ways][n_nfeat]

    def cuda(self):
        self.mus = self.mus.cuda()

    def init_from_labelled(self, X: torch.Tensor, n_runs=1):
        self.mus = X.reshape(n_runs, self.num_shot + self.num_query, self.num_way, -1)[:, :self.num_shot, ].mean(1)

    def updateFromEstimate(self, estimate, alpha):
        Dmus = estimate - self.mus
        self.mus = self.mus + alpha * Dmus

    def compute_optimal_transport(self, M: torch.Tensor, r: torch.Tensor, c: torch.Tensor, epsilon: float = 1e-6):
        r = r.cuda()
        c = c.cuda()
        n_runs, n, m = M.shape
        P = torch.exp(-self.lam * M)
        P /= P.view((n_runs, -1)).sum(1).unsqueeze(1).unsqueeze(1)

        u = torch.zeros(n_runs, n).cuda()

        maxiters = 1000
        iters = 1
        # normalize this matrix
        while torch.max(torch.abs(u - P.sum(2))) > epsilon:
            u = P.sum(2)
            P *= (r / u).view((n_runs, -1, 1))
            P *= (c / P.sum(1)).view((n_runs, 1, -1))
            if iters == maxiters:
                break
            iters = iters + 1

        return P, u

    @staticmethod
    def _pairwise_dist(a: torch.Tensor, b: torch.Tensor):
        return (a.unsqueeze(2) - b.unsqueeze(1)).norm(dim=3).pow(2)

    def getProbas(self, X: torch.Tensor, labels: torch.Tensor):
        dist = GaussianModel._pairwise_dist(X, self.mus)

        p_xj = torch.zeros_like(dist)
        r = torch.ones(self.num_runs, self.n_usamples, device='cuda')
        c = torch.ones(self.num_runs, self.num_way, device='cuda') * self.num_query
        p_xj_test, _ = self.compute_optimal_transport(dist[:, self.n_lsamples:], r, c, epsilon=1e-6)
        p_xj[:, self.n_lsamples:] = p_xj_test

        p_xj[:self.n_lsamples].fill_(0)
        p_xj[:self.n_lsamples].scatter_(1, labels[:self.n_lsamples].unsqueeze(1), 1)

        return p_xj

    def estimateFromMask(self, X: torch.Tensor, mask: torch.Tensor):
        emus = mask.permute(0, 2, 1).matmul(X).div(mask.sum(dim=1).unsqueeze(2))
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
        olabels = probas.argmax(dim=2)
        matches = self.labels.eq(olabels).float()
        acc_test = matches[:, self.num_labeled:].mean(1)

        m = acc_test.mean().item()
        pm = acc_test.std().item() * 1.96 / math.sqrt(self.n_runs)
        return m, pm

    def perform_epoch(self, model: GaussianModel, X: torch.Tensor):
        p_xj, _ = model.getProbas(X=X, labels=self.labels)
        m_estimates = model.estimateFromMask(X=X, mask=p_xj)

        # update centroids
        model.updateFromEstimate(m_estimates, self.alpha)

    def loop(self, X: torch.Tensor, model: GaussianModel, n_epochs: int = 20):
        probas, _ = model.getProbas(X=X, labels=self.labels)

        for epoch in range(1, n_epochs + 1):
            self.perform_epoch(model=model, X=X)

        # get final accuracy and return it
        P, log_P = model.getProbas(X=X, labels=self.labels)
        acc = self.get_accuracy(P)
        return acc, log_P[0]


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
        Z = Z.unsqueeze(0)

        # MAP
        gaussian_model = GaussianModel(num_way=self.num_way, num_shot=self.num_shot, num_query=self.num_query,
                                       lam=self.lam)
        gaussian_model.init_from_labelled(X=Z)

        optim = MAP(labels=labels, alpha=self.alpha, num_labeled=self.num_labeled)
        acc, probas = optim.loop(X=Z, model=gaussian_model, n_epochs=20)

        return acc, probas
