import os
import json
import re

import numpy as np
import numpy.ma as ma


# import matplotlib
# matplotlib.use('Agg')

import GeometricComplex as GC
import matplotlib.pyplot as plt

import seaborn as sns

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

        mpl_axis.set_aspect('equal')
        if inverted:
            minimal *= -1
            maximal *= -1

        mpl_axis.plot([minimal, maximal], [minimal, maximal],
                      color='black', alpha=0.5)

        points = np.array(self.diagrams[i][filtration][0])

        if points.size:  # if the array is not empty...

            mpl_axis.scatter(points[:, 0], points[:, 1],
                             marker='x', s=100, color='r',
                             linewidth=2)
            for pt in points:
                mpl_axis.plot(np.array([pt[0], pt[0]]),
                              np.array([pt[0], pt[1]]),
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
            cmplx = fig.add_subplot(2,3,1)
            cmplx.set_aspect('equal')
            self.plot_delaunay(i)

            cmplx.tick_params(labeltop='on', labelleft='on',
                              labelright='off', labelbottom='off')

            inv_cmplx = fig.add_subplot(2,3,6)
            inv_cmplx.set_aspect('equal')
            inv_cmplx.tick_params(labeltop='on', labelright='on')
            inv_cmplx.tick_params(labeltop='off', labelleft='off',
                                  labelright='on', labelbottom='on')
            self.points *= -1
            self.plot_delaunay(i)

            with sns.axes_style("whitegrid"):
                y_inc = fig.add_subplot(232, sharey=cmplx)
                x_inc = fig.add_subplot(234, sharex=cmplx)
                self.plot_diagram(i, mpl_axis=x_inc, filtration="X", axis=0)
                self.plot_diagram(i, mpl_axis=y_inc, filtration="Y", axis=1)

                x_dec = fig.add_subplot(233, sharex=inv_cmplx)
                y_dec = fig.add_subplot(235, sharey=inv_cmplx)

                self.plot_diagram(i, mpl_axis=x_dec, filtration="X_inverted", axis=0, inverted=True)
                self.plot_diagram(i, mpl_axis=y_dec, filtration="Y_inverted", axis=1, inverted=True)


            for ax in [x_inc, y_inc, x_dec, y_dec]:
                for label in ax.get_xticklabels():
                    label.set_visible(False)
                for label in ax.get_yticklabels():
                    label.set_visible(False)
            plt.tight_layout()
            fig.subplots_adjust(wspace=0, hspace=0)

            for ax in [x_dec, y_dec, inv_cmplx]:
                pos1 = ax.get_position()
                pos2 = [pos1.x0 + 0.02, pos1.y0 - 0.03, pos1.width, pos1.height]
                ax.set_position(pos2)

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
