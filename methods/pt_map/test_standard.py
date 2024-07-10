import argparse
import math

from tqdm import tqdm
import sys

import torch
import torch.nn.functional as F

from methods.pt_map import FSLTask
from bpa import BPA


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def centerDatas(datas):
    datas[:, :n_lsamples] = datas[:, :n_lsamples, :] - datas[:, :n_lsamples].mean(1, keepdim=True)
    datas[:, n_lsamples:] = datas[:, n_lsamples:, :] - datas[:, n_lsamples:].mean(1, keepdim=True)
    return datas


def scaleEachUnitaryDatas(datas):
    norms = datas.norm(dim=-1, keepdim=True)
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
    def __init__(self, n_ways, lam, distance_metric: str = 'euclidean'):
        super(GaussianModel, self).__init__(n_ways)
        self.mus = None  # shape [n_runs][n_ways][n_nfeat]
        self.lam = lam
        self.distance_metric = distance_metric

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

    def getProbas(self):
        global ndatas, n_nfeat
        # compute squared dist to centroids [n_runs][n_samples][n_ways]
        if self.distance_metric == 'cosine':
            dist = 1-torch.bmm(F.normalize(ndatas), F.normalize(self.mus.transpose(1, 2)))
        elif self.distance_metric == 'ce':
            dist = -torch.bmm(torch.log(ndatas + 1e-5), self.mus.transpose(1, 2))
        else:
            dist = self._pairwise_dist(ndatas, self.mus)

        p_xj = torch.zeros_like(dist)
        r = torch.ones(n_runs, n_usamples, device='cuda')
        c = torch.ones(n_runs, n_ways, device='cuda') * n_queries
        p_xj_test = self.compute_optimal_transport(dist[:, n_lsamples:], r, c, epsilon=1e-4)
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
    def __init__(self, alpha=None, verbose: bool = False, progressBar: bool = False):
        self.verbose = verbose
        self.progressBar = progressBar
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
    parser.add_argument('--features_path', type=str,
                        default='/checkpoints/wrn/miniImagenet/WideResNet28_10_S2M2_R/last/output.plk')
    parser.add_argument('--dataset', type=str, default='miniimagenet', choices=['miniimagenet'])
    parser.add_argument('--num_way', type=int, default=5)
    parser.add_argument('--num_shot', type=int, default=5)
    parser.add_argument('--num_query', type=int, default=15)
    parser.add_argument('--num_runs', type=int, default=10000)
    parser.add_argument('--num_repeat', type=int, default=1,
                        help='repeat the evaluation n times for averaging purposes.')
    parser.add_argument('--verbose', type=bool_flag, default=False)

    # BPA args
    parser.add_argument('--ot_reg', type=float, default=0.1)
    parser.add_argument('--sink_iters', type=int, default=10)
    parser.add_argument('--distance_metric', type=str, default='cosine')
    parser.add_argument('--norm_type', type=str, default='sinkhorn')
    parser.add_argument('--mask_diag', type=bool_flag, default=True)
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
    FSLTask.loadDataSet(args.dataset, root=args.root, features_path=args.root + args.features_path)
    FSLTask.setRandomStates(cfg)
    ndatas = FSLTask.GenerateRunSet(cfg=cfg, end=n_runs)
    ndatas = ndatas.permute(0, 2, 1, 3).reshape(n_runs, n_samples, -1)
    labels = torch.arange(n_ways).view(1, 1, n_ways).expand(n_runs, n_shot + n_queries, n_ways).clone().view(n_runs,
                                                                                                             n_samples)
    labels = labels.cuda()
    ndatas = ndatas.cuda()

    # Power transform + QR + Normalize
    ndatas[:, ] = torch.pow(ndatas[:, ] + 1e-6, 0.5)
    ndatas = QRreduction(ndatas)
    ndatas = scaleEachUnitaryDatas(ndatas)
    # trans-mean-sub
    ndatas = centerDatas(ndatas)
    _ndatas = scaleEachUnitaryDatas(ndatas)
    # # transform data
    bpa = BPA(
        args.distance_metric,
        ot_reg=args.ot_reg,
        sinkhorn_iterations=args.sink_iters,
        mask_diag=args.mask_diag,
    )

    for dm in ['euclidean']:
        print(f"DM {dm}")
        for mask_diag in [False, True]:
            bpa.mask_diag = mask_diag
            print(f"sot mask_diag {bpa.mask_diag }")
            # for max_temp in [False, True]:
            #     print(f"sot max_temp {max_temp}")
            for reg in [0.1, 0.2, 0.3, 0.4, 0.5]:
                bpa.ot_reg = reg
                print(f"sot lambda {bpa.ot_reg}")

                ndatas = bpa(_ndatas)
                n_nfeat = ndatas.size(2)
                print("size of the datas...", ndatas.size())

                # MAP
                model = GaussianModel(n_ways=n_ways, lam=10, distance_metric=dm)
                model.initFromLabelledDatas()

                optim = MAP(alpha=0.2, verbose=args.verbose)
                acc_test = optim.loop(model, n_epochs=20)

                print("final accuracy found {:0.2f} +- {:0.2f}".format(*(100 * x for x in acc_test)))
