from comet_ml import Experiment
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import ScalarFormatter, FuncFormatter
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import vis_utils
# from test_visualize import visualize_tsne
from self_optimal_transport import SOT


def generate_std_dims_list(stds, dims, plot_means, plot_3d):
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


def generate_data(n_clusters, n_per, n_dim, std, centroids=None):
    # generate centroids
    if clusters_means is None:
        centroids = torch.zeros((n_clusters, n_dim))
        dist_means = torch.randint(low=-100, high=100, size=(n_clusters, n_dim)).float()
        for c in range(n_clusters):
            centroids[c, :] = torch.normal(mean=dist_means[c], std=1)

    # project centroids to the unit sphere
    unit_centroids = centroids / torch.linalg.norm(centroids, dim=-1, keepdim=True)

    # generate samples around centroids
    clusters = torch.zeros((n_clusters, n_per, n_dim))
    for c in range(n_clusters):
        for j in range(n_dim):
            clusters[c, :, j] = torch.normal(mean=unit_centroids[c, j], std=std, size=(1, n_per))

    return clusters.reshape(n_clusters * n_per, dim), centroids


def plot_matrices(mat_ss, mat_pd, mat_cs, mat_cd, mat_f, title):
    for idx, (matrix, t) in enumerate([[mat_ss, 'Sinkhorn Similarity'], [mat_pd, 'Sinkhorn Distances'],
                                       [mat_cs, 'Cosine Similarity'], [mat_cd, 'Cosine Distances'],
                                       [mat_f, 'Feature Distances']]):
        high_pct = np.percentile(matrix, q=95)
        low_pct = np.percentile(matrix, q=5)
        matrix[matrix >= high_pct] = high_pct
        matrix[matrix <= low_pct] = low_pct

        mat_ax = plt.subplot(3, 2, idx + 1)
        mat_ax.matshow(matrix, interpolation='bicubic')
        mat_ax.set_title(t)
        mat_ax.set_yticklabels([])
        mat_ax.set_xticklabels([])
        mat_ax.grid(False)

    plt.grid(False)
    plt.tight_layout()
    plt.title(title)
    experiment.log_figure(figure_name=str(exp_idx) + f'dim{dim}std{str(std)[2:6]}', figure=plt)
    plt.show(block=False)
    plt.close()


def plot_std(data_points, data_sot, metric):
    # function to plot line graph over std
    for i, curr_dim in enumerate(dim_list):
        # current color
        color = np.array([vis_utils.colors_per_class[i][::-1]], dtype=float)[0] / 255
        # actual plot
        plt.plot(std_list[:], data_points[curr_dim][metric], c=color, label=str(curr_dim))
        plt.plot(std_list[:], data_sot[curr_dim][metric], linestyle='--', c=color)

    # plot params...
    plt.xlabel('in-cluster STD (log-scale)')
    plt.ylabel(metric)
    plt.title(f'{metric} vs. STD')

    ax = plt.gca()
    if metric == 'Accuracy':
        ax.set_ylim([0.1, 1.1])

    plt.legend(bbox_to_anchor=(1.15, 0.5), loc='center right', title="dims")
    ax.set_xscale('log', base=2)
    ax.set_xticks(std_list[:])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.2g}'.format(y)))
    plt.grid(b=True, which='both', color='w', linestyle='-')

    # log to comet
    if args.use_comet:
        experiment.log_figure(figure_name=f'{metric}vsSTD', figure=plt)

    plt.tight_layout()
    plt.show(block=True)
    plt.close()


def plot_dim(data_points, data_sot, metric):
    # function to plot line graph over dimensions
    # consider non-empty keys for x axis
    d_list = [d for d in dim_list if data_points[d] != {}]

    for i in range(len(std_list)):
        color = np.array([vis_utils.colors_per_class[i][::-1]], dtype=float)[0] / 255
        # actual plot
        y_p = [data_points[j][metric][i] for j in d_list]
        plt.plot(d_list, y_p, c=color, label=f'{std_list[i]:.2f}')
        y_sot = [data_sot[j][metric][i] for j in d_list]
        plt.plot(d_list, y_sot, c=color, linestyle='--')

    # figure params
    plt.xlabel('number of dimensions (log-scale)')
    plt.xscale('log', base=2)
    plt.ylabel(metric)
    plt.legend(bbox_to_anchor=(1.15, 0.5), loc='center right', title="STDs")
    plt.title(f'{metric} vs. Dimension')
    plt.grid(b=True, which='both', color='w', linestyle='-')
    ax = plt.gca()  # get axis
    if metric == 'accuracy':
        ax.set_ylim([0.1, 1.1])  # range for y axis

    ax.set_xticks(d_list)
    ax.set_xticklabels(d_list, rotation=60)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    # log to comet
    if args.use_comet:
        experiment.log_figure(figure_name=f'{metric}vsDIM', figure=plt)

    plt.tight_layout()
    plt.show(block=True)
    plt.close()


if __name__ == '__main__':
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-clusters', type=int, default=5,
                        help='Number of classes to generate.')
    parser.add_argument('--n-per', type=int, default=15,
                        help='Number of samples to generate for each class.')
    parser.add_argument('--n-exp', type=int, default=5,
                        help='Number of different experiments. The results will display as the mean of them.')
    parser.add_argument('--clustering-method', type=str, default='k-means', choices=['k-means', 'spectral'],
                        help='Which clustering method to use.')
    parser.add_argument('--pca_dim', type=int, default=50,
                        help='To which dimension reduce the features (disable PCA when pca_dim <= 0).')
    parser.add_argument('--n_std', type=int, default=9,
                        help='Evaluate results on "n" different stds.')
    parser.add_argument('--n_dims', type=int, default=12,
                        help='Evaluate results on "n" different dimensions.')
    parser.add_argument('--std_mul', type=float, default=1.25,
                        help='The multiplication parameter for in-cluster std increasing.')
    parser.add_argument('--dim_mul', type=float, default=1.5,
                        help='The multiplication parameter for feature dimension increasing.')
    parser.add_argument('--use-comet', action='store_true',
                        help='If logging the results to comet.')
    parser.add_argument('--comet_key', type=str, default='',
                        help='The key for comet API.')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed to reproduce results.')
    parser.add_argument('--plot_3d', action='store_true',
                        help='Plot the 3d data.')
    parser.add_argument('--plot_centroids', action='store_true',
                        help='Plot the centroids.')
    parser.add_argument('--plot_tsne', action='store_true',
                        help='If plot (2d) features after reduce dimension using TSNE.')
    args = parser.parse_args()

    # set seed
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    n_clusters, n_per, n_std, n_dims = args.n_clusters, args.n_per, args.n_std, args.n_dims
    keys = ['Accuracy', 'NMI', 'ARI']

    # the Self-Optimal-Transformer
    sot = SOT(distance_metric='euclidean', ot_reg=0.1)

    # generate lists of stds and dimensions to evaluate on
    std_list, dim_list = generate_std_dims_list(stds=n_std, dims=n_dims, plot_3d=args.plot_3d,
                                                plot_means=args.plot_centroids)
    std_len = len(std_list)

    # prepare labels
    labels = np.arange(n_clusters).repeat(n_per).reshape((n_clusters, n_per))
    labels = labels.reshape(n_clusters * n_per)

    # define the clustering algorithm using sklearn
    if args.clustering_method == 'spectral':
        cluster = SpectralClustering(n_clusters=n_clusters, random_state=0)

    elif args.clustering_method == 'k-means':
        cluster = KMeans(n_clusters=n_clusters, random_state=0)

    # COMET
    experiment = None
    if args.use_comet:
        experiment = Experiment(api_key=args.comet_key, project_name='SOT_synthetic')
        exp_name = f'unit_sphere_NMI_ARI_{args.n_exp}_exp_{n_clusters}_classes_{n_per}_points'
        if args.pca_dim > 0:
            exp_name += '_PCA'
        experiment.set_name(exp_name)

    # PCA
    pca = PCA(n_components=args.pca_dim, svd_solver='full')

    # define final results dicts
    points_results = {k: {} for k in dim_list}
    sot_results = {k: {} for k in dim_list}

    # iterate over dimensions
    for dim_idx, dim in enumerate(dim_list):
        # define result dict for current dimension
        points_std_metrics = {k: np.zeros(std_len) for k in keys}
        sot_std_metrics = {k: np.zeros(std_len) for k in keys}

        # iterate over experiments
        for exp_idx in range(args.n_exp):
            clusters_means = None

            # iterate over stds
            for std_idx, std in enumerate(std_list):
                # generate clusters, when we increase the STD we will always start from the same centroids
                clusters, clusters_means = generate_data(n_clusters=n_clusters, n_per=n_per, n_dim=dim, std=std,
                                                         centroids=clusters_means)

                # applying PCA on the generated samples
                if 0 < args.pca_dim < dim:
                    pca.fit(clusters)
                    clusters = torch.from_numpy(pca.transform(clusters))

                # project samples to the sphere
                normalized_clusters = clusters / torch.norm(clusters, dim=-1, keepdim=True)

                # both setting are just for visualization
                if std != 0.01 and dim != 3:
                    # clustering the original samples
                    cluster.fit(normalized_clusters)
                    points_std_metrics = vis_utils.gather_metrics(points_std_metrics, idx=std_idx, labels=labels,
                                                                  cluster_labels=cluster.labels_)

                    # clustering the Self-Optimal-Transformed data
                    sot_points = sot.transform(X=normalized_clusters)
                    cluster.fit(sot_points)
                    sot_std_metrics = vis_utils.gather_metrics(sot_std_metrics, idx=std_idx, labels=labels,
                                                               cluster_labels=cluster.labels_)

                # plot 3d data for some random experiments
                if dim == 3 and exp_idx in [0, 4]:
                    vis_utils.plot_3d_data(unit_clusters=normalized_clusters, labels=labels, std=std, std_idx=std_idx,
                                           exp_idx=exp_idx, experiment=experiment)

                # TSNE plot
                if args.plot_tsne:
                    # plot interesting setups...
                    if dim > 20 and exp_idx in [0, 4] and std_idx > 3:
                        # reduce dimension of original points
                        tsne = TSNE(n_components=2).fit_transform(normalized_clusters)
                        vis_utils.visualize_tsne(tsne=tsne, labels=labels,
                                                 batch_num=str(exp_idx) + f'dim{dim}std{str(std)[2:6]}',
                                                 title=f't-SNE original points | Dim={dim} | STD={std:.4f} | Exp {exp_idx}',
                                                 experiment=experiment)

                        # plot transformed points
                        tsne = TSNE(n_components=2).fit_transform(sot_points)
                        vis_utils.visualize_tsne(tsne=tsne, labels=labels,
                                                 batch_num=str(exp_idx) + f'dim{dim}std{str(std)[2:6]}',
                                                 title=f't-SNE Self-Optimal-Transform plot | Dim={dim} | STD={std:.4f} | Exp {exp_idx}',
                                                 experiment=experiment)

        if dim != 3:
            # average over all experiments
            for k in points_std_metrics.keys():
                points_std_metrics[k] /= args.n_exp
                sot_std_metrics[k] /= args.n_exp

            # update results
            points_results[dim] = points_std_metrics
            sot_results[dim] = sot_std_metrics

    # remove empty keys
    for k in points_results.keys():
        if points_results[k] == {}:
            del points_results[k]
            del sot_results[k]

    # plot graphs
    for k in keys:
        plot_std(data_points=points_results, data_sot=sot_results, metric=k)
        plot_dim(data_points=points_results, data_sot=sot_results, metric=k)
