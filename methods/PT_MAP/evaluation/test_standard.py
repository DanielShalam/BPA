import argparse
import torch
import math
from tqdm import tqdm
from . import FSLTask
import sys

sys.path.append('/root/Daniel/SOT/')
from self_optimal_transport import SOT

use_gpu = torch.cuda.is_available()
softmax = torch.nn.Softmax(dim=1)


def centerDatas(datas):
    datas[:, :n_lsamples] = datas[:, :n_lsamples, :] - datas[:, :n_lsamples].mean(1, keepdim=True)
    datas[:, n_lsamples:] = datas[:, n_lsamples:, :] - datas[:, n_lsamples:].mean(1, keepdim=True)
    return datas


def scaleEachUnitaryDatas(datas):
    norms = datas.norm(dim=2, keepdim=True)
    return datas / norms


def QRreduction(datas):
    ndatas = torch.linalg.qr(datas.permute(0, 2, 1)).R
    ndatas = ndatas.permute(0, 2, 1)
    return ndatas


class Model:
    def __init__(self, n_ways):
        self.n_ways = n_ways


# ---------  GaussianModel
class GaussianModel(Model):
    def __init__(self, n_ways, lam):
        super(GaussianModel, self).__init__(n_ways)
        self.mus = None  # shape [n_runs][n_ways][n_nfeat]
        self.lam = lam
        self.v = None

    def clone(self):
        other = GaussianModel(self.n_ways)
        other.mus = self.mus.clone()
        return self

    def cuda(self):
        self.mus = self.mus.cuda()

    def initFromLabelledDatas(self):
        self.mus = ndatas.reshape(n_runs, n_shot + n_queries, n_ways, n_nfeat)[:, :n_shot, ].mean(1)

    def updateFromEstimate(self, estimate, alpha):

        Dmus = estimate - self.mus
        self.mus = self.mus + alpha * Dmus

    def compute_optimal_transport(self, M, r, c, epsilon=1e-6):
        r = r.cuda()
        c = c.cuda()
        n_runs, n, m = M.shape
        P = torch.exp(- self.lam * M)
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
        return P

    @staticmethod
    def _pairwise_dist(a, b):
        return (a.unsqueeze(2) - b.unsqueeze(1)).norm(dim=3).pow(2)

    def _batch_dist(self, a, b, _dist_batch_size):
        num_a = a.size(1)
        num_b = b.size(1)
        _s = _dist_batch_size

        to_stack = list()
        for i_a in range(0, num_a, _s):
            a_slice = slice(i_a, min(i_a + _s, num_a))
            cur_row = list()
            for i_b in range(0, num_b, _s):
                b_slice = slice(i_b, min(i_b + _s, num_b))
                row_dist = self._pairwise_dist(a[:, a_slice, ...], b[:, b_slice, ...], )
                cur_row.append(row_dist)
            to_stack.append(torch.cat(cur_row, dim=2))

        return torch.cat(to_stack, dim=1)

    def getProbas(self):
        global ndatas, n_nfeat
        # compute squared dist to centroids [n_runs][n_samples][n_ways]
        dist = self._pairwise_dist(ndatas, self.mus)
        # dist = self._batch_dist(ndatas, self.mus, 5)
        p_xj = torch.zeros_like(dist)
        r = torch.ones(n_runs, n_usamples, device='cuda')
        c = torch.ones(n_runs, n_ways, device='cuda') * n_queries
        # sinkhorn = SinkhornDistance(eps=0.1, max_iter=200)
        p_xj_test = self.compute_optimal_transport(dist[:, n_lsamples:], r, c, epsilon=1e-6)
        p_xj[:, n_lsamples:] = p_xj_test

        p_xj[:, :n_lsamples].fill_(0)
        p_xj[:, :n_lsamples].scatter_(2, labels[:, :n_lsamples].unsqueeze(2), 1)

        return p_xj

    def estimateFromMask(self, mask):
        emus = mask.permute(0, 2, 1).matmul(ndatas).div(mask.sum(dim=1).unsqueeze(2))
        return emus


# =========================================
#    MAP
# =========================================

class MAP:
    def __init__(self, alpha=None):
        self.verbose = False
        self.progressBar = False
        self.alpha = alpha

    def getAccuracy(self, probas):
        olabels = probas.argmax(dim=2)
        matches = labels.eq(olabels).float()
        acc_test = matches[:, n_lsamples:].mean(1)

        m = acc_test.mean().item()
        pm = acc_test.std().item() * 1.96 / math.sqrt(n_runs)
        return m, pm

    def performEpoch(self, model, epochInfo=None):

        p_xj = model.getProbas()
        self.probas = p_xj

        if self.verbose:
            print("accuracy from filtered probas", self.getAccuracy(self.probas))

        m_estimates = model.estimateFromMask(self.probas)

        # update centroids
        model.updateFromEstimate(m_estimates, self.alpha)

        if self.verbose:
            op_xj = model.getProbas()
            acc = self.getAccuracy(op_xj)
            print("output model accuracy", acc)

    def loop(self, model, n_epochs=20):

        self.probas = model.getProbas()
        if self.verbose:
            print("initialisation model accuracy", self.getAccuracy(self.probas))

        if self.progressBar:
            if type(self.progressBar) == bool:
                pb = tqdm(total=n_epochs)
            else:
                pb = self.progressBar

        for epoch in range(1, n_epochs + 1):
            if self.verbose:
                print("----- epoch[{:3d}]  lr_p: {:0.3f}  lr_m: {:0.3f}".format(epoch, self.alpha))
            self.performEpoch(model, epochInfo=(epoch, n_epochs))
            if self.progressBar: pb.update()

        # get final accuracy and return it
        op_xj = model.getProbas()
        acc = self.getAccuracy(op_xj)
        return acc


def get_args():
    """ Description: Parses arguments at command line. """
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='C:/Users/dani3/Documents/GitHub/SOT/')
    parser.add_argument('--path', type=str,
                        default='C:/Users/dani3/Documents/GitHub/SOT/checkpoints/wrn/miniImagenet/WideResNet28_10_S2M2_R/last/output.plk')
    parser.add_argument('--dataset', type=str, default='miniimagenet', choices=['miniimagenet'])
    parser.add_argument('--num_way', type=int, default=5)
    parser.add_argument('--num_shot', type=int, default=5)
    parser.add_argument('--num_query', type=int, default=15)
    parser.add_argument('--num_runs', type=int, default=10000)

    # SOT args
    parser.add_argument('--ot_reg', type=float, default=0.1)
    parser.add_argument('--sinkhorn_iterations', type=int, default=10)
    parser.add_argument('--distance_metric', type=str, default='cosine')
    return parser.parse_args()


if __name__ == '__main__':
    # ---- data loading
    args = get_args()
    n_shot = args.num_shot
    n_ways = args.num_way
    n_queries = args.num_query
    n_runs = args.num_runs
    n_lsamples = n_ways * n_shot
    n_usamples = n_ways * n_queries
    n_samples = n_lsamples + n_usamples

    cfg = {'shot': n_shot, 'ways': n_ways, 'queries': n_queries}
    FSLTask.loadDataSet(args.dataset, root=args.root, path=args.path)
    FSLTask.setRandomStates(cfg)
    ndatas = FSLTask.GenerateRunSet(cfg=cfg, end=n_runs)
    ndatas = ndatas.permute(0, 2, 1, 3).reshape(n_runs, n_samples, -1)

    labels = torch.arange(n_ways).view(1, 1, n_ways).expand(n_runs, n_shot + n_queries, n_ways).clone().view(n_runs,
                                                                                                             n_samples)

    sot = SOT(distance_metric=args.distance_metric, ot_reg=args.ot_reg, sinkhorn_iterations=args.sinkhorn_iterations, )

    # Power transform
    ndatas[:, ] = torch.pow(ndatas[:, ] + 1e-6, 0.5)
    ndatas = QRreduction(ndatas)
    ndatas = scaleEachUnitaryDatas(ndatas)

    # trans-mean-sub
    ndatas = centerDatas(ndatas)
    ndatas = scaleEachUnitaryDatas(ndatas)

    # switch to cuda
    ndatas = ndatas.cuda()
    labels = labels.cuda()

    # # transform data
    ndatas = sot(ndatas, n_samples=n_shot + n_queries, y_support=labels[0, :n_lsamples], max_temperature=True)

    n_nfeat = ndatas.size(2)
    print("size of the datas...", ndatas.size())

    # MAP
    lam = 10
    model = GaussianModel(n_ways, lam)
    model.initFromLabelledDatas()

    alpha = 0.2
    optim = MAP(alpha)

    optim.verbose = False
    optim.progressBar = True

    acc_test = optim.loop(model, n_epochs=20)

    print("final accuracy found {:0.2f} +- {:0.2f}".format(*(100 * x for x in acc_test)))
