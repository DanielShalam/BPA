from comet_ml import Experiment
import argparse
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FuncFormatter

# local modules
import vis_utils
from data_utils import generate_std_dims_list, generate_data
from self_optimal_transport import SOT


def plot_x_std(data_points, data_sot, metric):
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
    plt.legend(bbox_to_anchor=(1.15, 0.5), loc='center right', title="dims")
    ax.set_xscale('log', base=2)
    ax.set_xticks(std_list[:])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.2g}'.format(y)))
    if metric == 'Accuracy':
        ax.set_ylim([0.1, 1.1])

    plt.grid(b=True, which='both', color='w', linestyle='-')

    # log to comet
    if args.use_comet:
        experiment.log_figure(figure_name=f'{metric}vsSTD', figure=plt)

    plt.tight_layout()
    plt.show(block=True)
    plt.close()


def plot_x_dim(data_points, data_sot, metric):
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
    parser.add_argument('--n_exp', type=int, default=5,
                        help='Number of different experiments. The results will display as the average of them.')
    parser.add_argument('--n_per', type=int, default=15,
                        help='Number of samples to generate for each class.')
    parser.add_argument('--n_clusters', type=int, default=5,
                        help='Number of classes to generate.')
    parser.add_argument('--clustering_method', type=str, default='k-means', choices=['k-means', 'spectral'],
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
    parser.add_argument('--use_comet', action='store_true',
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
                        help='If plot (2d) features using TSNE.')
    args = parser.parse_args()

    # set seed
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    num_clusters, num_per, num_std, num_dims = args.n_clusters, args.n_per, args.n_std, args.n_dims
    metrics = ['Accuracy', 'NMI', 'ARI']

    # the Self-Optimal-Transformer
    SOT = SOT(distance_metric='euclidean', ot_reg=0.1)

    # generate lists of stds and dimensions to evaluate on
    std_list, dim_list = generate_std_dims_list(stds=num_std, dims=num_dims, plot_3d=args.plot_3d,
                                                plot_means=args.plot_centroids, args=args)

    # prepare labels
    labels = np.arange(num_clusters).repeat(num_per).reshape((num_clusters, num_per)).reshape(num_clusters * num_per)

    # define the clustering algorithm using sklearn
    if args.clustering_method == 'spectral':
        cluster = SpectralClustering(n_clusters=num_clusters, random_state=0)
    else:
        cluster = KMeans(n_clusters=num_clusters, random_state=0)

    # configure COMET if needed
    experiment = None
    if args.use_comet:
        experiment = Experiment(api_key=args.comet_key, project_name='SOT_synthetic')
        exp_name = f'unit_sphere_NMI_ARI_{args.n_exp}_exp_{num_clusters}_clusters_{num_per}_points'
        if args.pca_dim > 0:
            exp_name += '_PCA'
        experiment.set_name(exp_name)

    # PCA
    if args.pca_dim > 0:
        pca = PCA(n_components=args.pca_dim, svd_solver='full')

    points_res = {k: {} for k in dim_list}
    sot_res = {k: {} for k in dim_list}

    # iterate over dimensions
    for dim_idx, dim in enumerate(dim_list):
        # define result dict for current dimension
        points_std_metrics = {k: np.zeros(len(std_list)) for k in metrics}
        sot_std_metrics = {k: np.zeros(len(std_list)) for k in metrics}

        # iterate over experiments
        for exp_idx in range(args.n_exp):
            clusters_means = None

            # iterate over stds
            for std_idx, std in enumerate(std_list):
                # generate clusters, if only the STD has changed, the points will be generated from the previous centers
                clusters, clusters_means = generate_data(num_clusters=num_clusters, num_per=num_per, dim=dim, std=std,
                                                         centroids=clusters_means)

                # applying PCA on the generated samples
                if 0 < args.pca_dim < dim:
                    pca.fit(clusters)
                    clusters = torch.from_numpy(pca.transform(clusters))

                # project samples to the sphere
                normalized_clusters = F.normalize(clusters, dim=-1, p=2)

                # plot the 3d data for some random experiments
                if args.plot_3d and dim == 3 and exp_idx in [0, 4]:
                    vis_utils.plot_3d_data(unit_clusters=normalized_clusters, labels=labels, std=std, std_idx=std_idx,
                                           exp_idx=exp_idx, experiment=experiment)

                # clustering the random samples
                cluster.fit(normalized_clusters)
                points_std_metrics = vis_utils.gather_metrics(points_std_metrics, idx=std_idx, labels=labels,
                                                              cluster_labels=cluster.labels_)

                # clustering the SOT data
                SOT_points = SOT(X=normalized_clusters)
                cluster.fit(SOT_points)
                sot_std_metrics = vis_utils.gather_metrics(sot_std_metrics, idx=std_idx, labels=labels,
                                                           cluster_labels=cluster.labels_)

                # plot the original and transformed clusters using TSNE
                if args.plot_tsne:
                    # plot some interesting configurations...
                    if dim > 20 and exp_idx in [0, 4] and std_idx > 3:
                        exp_name = str(exp_idx) + f'dim{dim}std{str(std)[2:6]}'
                        tsne = TSNE(n_components=2).fit_transform(normalized_clusters)
                        vis_utils.visualize_tsne(tsne=tsne, labels=labels, name=exp_name, experiment=experiment,
                                                 title=f't-SNE (Points) | Dim={dim} | STD={std:.4f} | Exp {exp_idx}')

                        tsne = TSNE(n_components=2).fit_transform(SOT_points)
                        vis_utils.visualize_tsne(tsne=tsne, labels=labels, name=exp_name, experiment=experiment,
                                                 title=f't-SNE (SOT) | Dim={dim} | STD={std:.4f} | Exp {exp_idx}')

        # average over all experiments
        for k in points_std_metrics.keys():
            points_std_metrics[k] /= args.n_exp
            sot_std_metrics[k] /= args.n_exp

        # update results
        points_res[dim] = points_std_metrics
        sot_res[dim] = sot_std_metrics

    # remove empty keys
    for k in points_res.keys():
        if points_res[k] == {}:
            del points_res[k]
            del sot_res[k]

    # plot graphs
    for metric in metrics:
        plot_x_std(data_points=points_res, data_sot=sot_res, metric=metric)
        plot_x_dim(data_points=points_res, data_sot=sot_res, metric=metric)
