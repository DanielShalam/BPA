import torch


def generate_std_dims_list(stds, dims, plot_means, plot_3d, args):
    # initial std is 0.1 (0.01 is just for plotting)
    std_list = [0.01, 0.1] if plot_means else [0.1]
    for idx in range(stds):
        std_list.append(std_list[-1] * args.std_mul)

    # initial dimension is 10
    dim_list = [10]
    for idx in range(dims):
        dim_list.append(int(dim_list[-1] * args.dim_mul))

    # adding 3D just for visualization
    if plot_3d:
        dim_list = [3] + dim_list

    return std_list, dim_list


def generate_data(num_clusters, num_per, dim, std, centroids=None):
    if centroids is None:
        # generate centroids
        centroids = torch.zeros((num_clusters, dim))
        dist_means = torch.randint(low=-100, high=100, size=(num_clusters, dim)).float()
        # take values from a gaussian distribution
        for c in range(num_clusters):
            centroids[c, :] = torch.normal(mean=dist_means[c], std=1)

    # project them to the unit sphere
    normalize_centroids = centroids / torch.linalg.norm(centroids, dim=-1, keepdim=True)

    # generate samples around cluster centers
    clusters = torch.zeros((num_clusters, num_per, dim))
    for c in range(num_clusters):
        for j in range(dim):
            # take values from a gaussian distribution given by the centroids
            clusters[c, :, j] = torch.normal(mean=normalize_centroids[c, j], std=std, size=(1, num_per))

    return clusters.reshape(num_clusters * num_per, dim), centroids
