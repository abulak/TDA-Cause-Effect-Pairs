import os
import json
import re

import numpy as np
import numpy.ma as ma


# import matplotlib
# matplotlib.use('Agg')

import GeometricComplex as GC
import matplotlib.pyplot as plt

from matplotlib.collections import LineCollection, PolyCollection


class TopologyPlotter:
    """
    Outlier Plotter; requires outliers model and assumes points.std,
    outliers.$model, and diagram.$model are in the path given.
    """

    def __init__(self, path, model="knn", name=None):
        if name is None:
            self.name = path[-9:-1]
        else:
            self.name = name
        self.suffix = str(model)
        self.points = standardise(np.loadtxt(os.path.join(path, 'points.std')))

        outliers_path = os.path.join(path, 'outliers.' + self.suffix)
        self.outliers = np.loadtxt(outliers_path).astype(np.int)

        assert self.outliers.shape[0] == len(set(list(self.outliers))), \
            "Duplicates in outliers!"

        diagrams_path = os.path.join(path, 'diagrams.' + self.suffix)
        with open(diagrams_path, 'r') as file:
            self.diagrams = json.load(file)

        extrema_path = os.path.join(path, 'extrema.' + self.suffix)
        with open(extrema_path, 'r') as file:
            self.extrema = json.load(file)

    def __mask_points__(self, i):
        masked_points = ma.masked_array(self.points)
        for outlier in self.outliers[:i]:
            masked_points[outlier] = ma.masked
        cleaned_points = masked_points.compressed().reshape(
                    self.points.shape[0] - i, 2)
        return cleaned_points

    def plot_diagram(self, i, mpl_axis, filtration, axis,
                     inverted=False):

        minimal = self.extrema[i]['minima'][axis]
        maximal = self.extrema[i]['maxima'][axis]
        mpl_axis.set_aspect(1)
        mpl_axis.plot([minimal, maximal], [minimal, maximal],
                      color='black', alpha=0.5)
        points = np.array(self.diagrams[i][filtration][0])

        if points.size:  # if the array is not empty...
            if inverted:
                points *= -1

            mpl_axis.scatter(points[:, 0], points[:, 1],
                             marker='+', s=300, color='r',
                             linewidth=1.5)
            for pt in points:
                mpl_axis.plot([pt[0], pt[0]], [pt[0], pt[1]],
                              color='black', alpha=0.3)
        mpl_axis.text(0.5, 0.1, "Filtration: " + filtration,
                      transform=mpl_axis.transAxes)

    def plot_delaunay(self, i):
        if i < 1:
            cleaned_points = self.points
        else:
            cleaned_points = standardise(self.__mask_points__(i-1))
        edges, triangles = delaunay_triangulation(cleaned_points, dimension=2)
        lines = LineCollection(edges, linewidths=1, colors='b', alpha=0.3)
        triangles = PolyCollection(triangles, facecolors='b', alpha=0.1)
        ax = plt.gca()
        plt.scatter(cleaned_points[:, 0], cleaned_points[:, 1],
                    color='red', alpha=1, s=5)
        ax.add_collection(lines)
        ax.add_collection(triangles)
        return ax

    def plot_all_diagrams(self, i):
        if i > len(self.outliers):
            print("argument must be less than", len(self.outliers),
                  "for the pair!")
        else:
            fig = plt.gcf()

            central = fig.add_subplot(335, zorder=10)

            central.set_aspect('equal')
            self.plot_delaunay(i)
            central.tick_params(labeltop='on', labelright='on')

            top = fig.add_subplot(332, sharex=central)
            bottom = fig.add_subplot(338, sharex=central)
            left = fig.add_subplot(334, sharey=central)
            right = fig.add_subplot(336, sharey=central)

            for ax in [top, bottom, left, right]:
                for label in ax.get_xticklabels():
                    label.set_visible(False)
                for label in ax.get_yticklabels():
                    label.set_visible(False)

            #     ax.set_yticklabels([])
            right.invert_xaxis()
            self.plot_diagram(i, mpl_axis=top, filtration="X", axis=0)

            self.plot_diagram(i, mpl_axis=left, filtration="Y", axis=1)
            self.plot_diagram(i, mpl_axis=right,
                              filtration="Y_inverted", axis=1, inverted=True)
            # axs[2,0].set_axis_off()
            self.plot_diagram(i, mpl_axis=bottom,
                              filtration="X_inverted", axis=0, inverted=True)

            plt.tight_layout()
            fig.subplots_adjust(wspace=0, hspace=0)
            return fig


def standardise(points):
    """
    Standardise points
    :param points: np.array
    :return: np.array of points after standardisation (mean=0, st deviation=1)
    """
    for i in range(points[0].shape[0]):
        p = points[:, i]
        mean = np.mean(p)
        std = np.std(p)
        p -= mean
        p /= std
    return points


def delaunay_triangulation(points, dimension=2):
    cmplx = GC.AlphaGeometricComplex(points, dimension=dimension)
    real_edges = cmplx.get_real_edges(cmplx.limited_simplices)
    real_triangles = cmplx.get_real_triangles(cmplx.limited_simplices)

    return real_edges, real_triangles
