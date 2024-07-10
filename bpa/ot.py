import torch


def log_sum_exp(u: torch.Tensor, dim: int):
    # Reduce log sum exp along axis
    u_max, __ = u.max(dim=dim, keepdim=True)
    log_sum_exp_u = torch.log(torch.exp(u - u_max).sum(dim)) + u_max.sum(dim)
    return log_sum_exp_u


def log_sinkhorn(M: torch.Tensor, reg: float, num_iters: int):
    """
    Log-space-sinkhorn algorithm for better stability.
    """
    if M.dim() > 2:
        return batched_log_sinkhorn(M=M, reg=reg, num_iters=num_iters)

    # Initialize dual variable v (u is implicitly defined in the loop)
    log_v = torch.zeros(M.size()[1]).to(M.device)  # ==torch.log(torch.ones(m.size()[1]))

    # Exponentiate the pairwise distance matrix
    log_K = -M / reg

    # Main loop
    for i in range(num_iters):
        # Match r marginals
        log_u = - log_sum_exp(log_K + log_v[None, :], dim=1)

        # Match c marginals
        log_v = - log_sum_exp(log_u[:, None] + log_K, dim=0)

    # Compute optimal plan, cost, return everything
    log_P = log_u[:, None] + log_K + log_v[None, :]
    return log_P


def batched_log_sinkhorn(M, reg: float, num_iters: int):
    """
    Batched version of log-space-sinkhorn.
    """
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
    for i in range(num_iters):
        u1 = u  # useful to check the update
        u = reg * (torch.log(mu + 1e-8) - torch.logsumexp(C(M, u, v, reg), dim=-1)) + u
        v = reg * (torch.log(nu + 1e-8) - torch.logsumexp(C(M, u, v, reg).transpose(-2, -1), dim=-1)) + v
        err = (u - u1).abs().sum(-1).mean()

        actual_nits += 1
        if err.item() < thresh:
            break

    U, V = u, v
    # Transport plan pi = diag(a)*K*diag(b)
    log_p = C(M, U, V, reg)
    return log_p
