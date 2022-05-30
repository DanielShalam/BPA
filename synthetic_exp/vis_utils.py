import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

import utils

colors_per_class = {
    0: [254, 202, 87],
    1: [255, 107, 107],
    2: [10, 189, 227],
    3: [255, 159, 243],
    4: [16, 172, 132],
    5: [128, 80, 128],
    6: [128, 229, 88],
    7: [1, 6, 10],
    8: [1, 250, 13],
    9: [250, 10, 10],
    10: [180, 180, 180],
    11: [132, 23, 200],
    12: [200, 240, 1],
}


def plot_3d_data(unit_clusters, labels,  std, std_idx, exp_idx, experiment=None):
    # plot 3d data
    if std_idx == 0:
        sizes = [100, 200, 400]
    else:
        sizes = [-1, 100, 150]

    for s in sizes:
        if s == -1:
            plt.subplot(111, projection='3d').scatter(unit_clusters[:, 0], unit_clusters[:, 1], unit_clusters[:, 2], c=labels)
        else:
            plt.subplot(111, projection='3d').scatter(unit_clusters[:, 0], unit_clusters[:, 1], unit_clusters[:, 2], c=labels, s=s)

        ax = plt.gca()
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_zticklabels([])
        ax.grid(False)

        ax.set(facecolor="white")
        plt.title(f"STD={std:.2f}")
        set_axes_equal(ax)
        if experiment is not None:
            experiment.log_figure(figure_name=f'{exp_idx}3d', figure=plt)

        plt.show(block=False)
        plt.close()


def gather_metrics(metric_dict, idx, labels, cluster_labels):
    metric_dict['Accuracy'][idx] += utils.clustering_accuracy(labels, cluster_labels)[0]
    metric_dict['NMI'][idx] += normalized_mutual_info_score(labels, cluster_labels)
    metric_dict['ARI'][idx] += adjusted_rand_score(labels, cluster_labels)
    return metric_dict


def visualize_tsne(tsne, labels, name='', title='', experiment=None):
    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]
    # scale and move the coordinates so they fit [0; 1] range
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    # visualize the plot: samples as colored points
    visualize_tsne_points(tx, ty, labels, name=name, title=title, experiment=experiment)


def visualize_tsne_points(tx, ty, labels, name, title, experiment=None):
    # initialize matplotlib plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # for every class, we'll add a scatter plot separately
    for label in colors_per_class:
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(labels) if l == label]

        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)

        # convert the class color to matplotlib format:
        # BGR -> RGB, divide by 255, convert to np.array
        color = np.array([colors_per_class[label][::-1]], dtype=float) / 255

        # add a scatter plot with the correponding color and label
        ax.scatter(current_tx, current_ty, c=color, label=label)

    ax.set_yticklabels([])
    ax.set_xticklabels([])

    plt.title(title)

    if experiment is not None:
        experiment.log_figure(figure_name=name, figure=plt)

    # finally, show the plot
    plt.show(block=True)
    plt.close()


def scale_to_01_range(x):
    # scaling values to [0, 1]
    value_range = (np.max(x) - np.min(x))
    starts_from_zero = x - np.min(x)
    return starts_from_zero / value_range


def set_axes_equal(ax):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
