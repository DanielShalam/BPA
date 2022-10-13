import torch
from torch import nn
from torch.nn import functional as F


class Frobenius(nn.Module):
    def __init__(self, n: int, ):
        super().__init__()
        self.n = n
        self.eye = torch.eye(n, device='cuda')
        self.ones = torch.ones((n, 1)).cuda(non_blocking=True)
        self.ReLU = nn.ReLU(inplace=True)

    @staticmethod
    def P_1(X: torch.Tensor):
        n = X.shape[1]
        eye = torch.eye(n).to(X.device)
        ones = torch.ones((n, 1)).to(X.device)
        mid_term = (eye/n + ones.T.mm(X).mm(ones)/(n**2) * eye - X/n)
        p_x = X + torch.mm(torch.mm(mid_term, ones), ones.T) - (ones.mm(ones.T).mm(X)) / n
        return p_x

    @staticmethod
    def P_1_temp(X: torch.Tensor):
        n = X.shape[1]
        eye = torch.eye(n).to(X.device)
        sum_col = X.sum(dim=0)
        mid_term = (eye/n + sum_col.sum()/(n**2) * eye - X/n)
        # torch.mm(mid_term, ones) computing sum of sum(dim=-1)
        p_x = X + mid_term.sum(1).unsqueeze(1) - sum_col.unsqueeze(0) / n
        return p_x

    def forward(self, X: torch.Tensor):
        """
        Finds the closest doubly stochastic matrix K' under Frobenius Norm.
        Note that K needs to be a positive-semi-definite similarity matrix.
        """
        i = 0
        while True:
            X = Frobenius.P_1(X)

            # terminate if M is positive
            if torch.min(X) >= 0:
                return X

            # set all negative values to 0
            X = F.relu(X)
            i += 1

            if i >= self.max_iters:
                if self.verbose:
                    print(f"Algorithm did not converge after {i} iterations. ")
                    print(X[0].sum().item(), X[:, 0].sum().item())

                return X


class L1(nn.Module):
    def __init__(self, n: int = 0, world_size: int = 0, max_iters: int = 0, verbose: bool = False):
        super().__init__()

    def forward(self, K: torch.Tensor):
        """
        Finds the closest doubly stochastic matrix K' under the L1 error norm.
        Note that K needs to be a positive-semi-definite similarity matrix.
        """
        K_t = K - torch.diag(K.sum(dim=-1)) + torch.eye(K.shape[0])
        return K_t
