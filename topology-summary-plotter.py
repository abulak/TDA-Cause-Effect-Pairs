import numpy as np
import numpy.ma as ma

import os
import sys

import matplotlib
# matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages

import matplotlib.pyplot as plt

import json

class PairTopologyPlotter:
    """
    Outlier Plotter; requires outliers model and assumes outliers_$model and
    orig_points are in the working directory
    """

    def __init__(self, model):
        self.name = os.getcwd()[-8:]
        self.suffix = str(model)
        self.points = np.loadtxt(os.path.join(os.getcwd(), 'std_points'))

        scores_path = os.path.join(os.getcwd(), 'scores_' + self.suffix)
        self.scores = np.loadtxt(scores_path)

        diagrams_path = os.path.join(os.getcwd(), 'diagrams_' + self.suffix)
        with open(diagrams_path, 'r') as file:
            self.diagrams = json.load(file)

        outliers_path = os.path.join(os.getcwd(), 'outliers_' + self.suffix)
        self.outliers = np.loadtxt(outliers_path, dtype=np.int)

        self.extrema = self.__find_extrema__()

    def __mask_points__(self, i):
        masked_points = ma.masked_array(self.points)
        outs = self.outliers[:i+1]
        masked_points[outs] = ma.masked
        cleaned_points = masked_points.compressed().reshape(
                    self.points.shape[0] - i - 1, 2)
        return cleaned_points

    def standardise(self, points):
        """Standardise self.points, i.e.
        mean = 0 and standard deviation = 1 in both dimensions"""
        for i in range(points[0].shape[0]):
            p = points[:, i]
            mean = np.mean(p)
            std = np.std(p)
            p -= mean
            p /= std
        return points

    def plot_scores(self):

        plt.title(self.name + ' ' + self.suffix + " scores")
        plt.plot(self.scores[0], label='x->y')
        plt.plot(self.scores[1], label='y->x')
        plt.legend(loc='best', fancybox=True, framealpha=0.5)

    def plot_diagram(self, i, matplotlib_axis, filtration, axis):

        ax = matplotlib_axis

        minimal = self.extrema[i]['minima'][axis]
        maximal = self.extrema[i]['maxima'][axis]
        ax.set_aspect(1)
        ax.plot([minimal, maximal], [minimal, maximal],
                color='black', alpha=0.5)
        points = np.array(self.diagrams[i][filtration])


        if points.shape[0]:  # if the array is not empty...
            if 'inv' in filtration:
                points *= -1
                points += [maximal, maximal]
                ax.scatter(points[:, 1], points[:, 0],
                           marker='+', s=40, facecolors='none', edgecolors='r')
            else:
                points += [minimal, minimal]
                ax.scatter(points[:, 0], points[:, 1],
                           marker='+', s=40, facecolors='none', edgecolors='r')

        ax.legend(loc=4, labels=[filtration], fancybox=True,
                  framealpha=0.5)

    def delaunay_lines(self, points):
        # plotting the whole Delaunay triangulation:
        from TDA import GeometricComplex
        from matplotlib.collections import LineCollection

        cmplx = GeometricComplex(points, full_initialisation=False)
        real_edges = cmplx.get_real_edges(cmplx.limited_complex)

        ln_coll = LineCollection(real_edges, colors='b', alpha=0.4)
        return ln_coll

    def plot_all_diagrams(self, i, size=(12, 12)):
        if i > len(self.outliers):
            print("argument must be less than", len(self.outliers),
                  "for the pair!")
        else:
            fig = plt.figure()
            fig.set_size_inches(size)
            fig.suptitle(self.name + " " + self.suffix + " outlier: " + str(i))

            central = fig.add_subplot(335)
            top = fig.add_subplot(332, sharex=central)
            bottom = fig.add_subplot(338, sharex=central)
            left = fig.add_subplot(334, sharey=central)
            right = fig.add_subplot(336, sharey=central)

            for ax in {central, top, bottom, left, right}:
                ax.grid(which='both', alpha=0.5)

            for ax in {top, bottom}:
                pass

            for ax in {left, right}:
                pass
                # ax.set_yticks([])

            central.tick_params(labeltop=True, labelright=True)
            cleaned_points = self.standardise(self.__mask_points__(i))
            central.scatter(cleaned_points[:, 0], cleaned_points[:, 1],
                            color='black', alpha=0.7, s=5)
            central.set_aspect(1)

            central.add_collection(self.delaunay_lines(cleaned_points))

            # axs[0, 0].set_axis_off()
            self.plot_diagram(i, matplotlib_axis=top,
                              filtration="x_filtration_H0", axis=0)
            # axs[0, 2].set_axis_off()
            self.plot_diagram(i, matplotlib_axis=left,
                              filtration="y_filtration_H0", axis=1)
            self.plot_diagram(i, matplotlib_axis=right,
                              filtration="y_inv_filtration_H0", axis=1)
            # axs[2,0].set_axis_off()
            self.plot_diagram(i, matplotlib_axis=bottom,
                              filtration="x_inv_filtration_H0", axis=0)

            # axs[2,2].set_axis_off()

            fig.tight_layout(pad=0, w_pad=0, h_pad=0)
            fig.subplots_adjust(wspace=0, hspace=0)

    def plot_all_diagrams2(self, i, size=(12, 12)):
        if i > len(self.outliers):
            print("argument must be less than", len(self.outliers),
                  "for the pair!")
        else:
            fig = plt.figure()
            fig.set_size_inches(size)
            fig.suptitle(self.name + " " + self.suffix + " outlier: " + str(i))

            pts = fig.add_subplot(331)
            pts.set_aspect('equal')
            y_f = fig.add_subplot(332, sharey=pts)
            x_f = fig.add_subplot(334, sharex=pts)

            xinv_f = fig.add_subplot(336, sharex=pts)
            yinv_f = fig.add_subplot(338, sharey=pts)
            invpts = fig.add_subplot(339)

            pts.tick_params(labeltop=True, labelleft=True)

            for ax in {pts, x_f, y_f, xinv_f, yinv_f, invpts}:
                ax.grid(which='both', alpha=0.5)

            cleaned_points = self.standardise(self.__mask_points__(i))
            pts.scatter(cleaned_points[:, 0], cleaned_points[:, 1],
                        color='black', alpha=0.7, s=5)
            lines = self.delaunay_lines(cleaned_points)
            pts.add_collection(lines)
            invpts.scatter(cleaned_points[:, 0], cleaned_points[:, 1],
                        color='black', alpha=0.7, s=5)
            invpts.add_collection(lines)
            print(pts.get_xlim())
            print(pts.get_ylim())
            invpts.set_xlim(pts.get_xlim())
            invpts.set_ylim(pts.get_ylim())
            invpts.invert_xaxis()
            invpts.invert_yaxis()

            self.plot_diagram(i, matplotlib_axis=x_f,
                              filtration="x_filtration_H0", axis=0)
            self.plot_diagram(i, matplotlib_axis=y_f,
                              filtration="y_filtration_H0", axis=1)

            self.plot_diagram(i, matplotlib_axis=yinv_f,
                              filtration="y_inv_filtration_H0", axis=1)
            self.plot_diagram(i, matplotlib_axis=xinv_f,
                              filtration="x_inv_filtration_H0", axis=0)

            fig.tight_layout(pad=0, w_pad=0, h_pad=0)
            fig.subplots_adjust(wspace=0, hspace=0)


def workflow(model):
    from matplotlib.backends.backend_pdf import PdfPages
    pdf_file = os.path.join(os.getcwd(), 'scores_' + model + '.pdf')
    with PdfPages(pdf_file) as pdf:
        plt.figure(figsize=(12, 12))
        p = PairTopologyPlotter(model)
        p.plot_scores()
        pdf.savefig()
        plt.close()

if __name__ == "__main__":
    if len(sys.argv) == 2:
        workflow(sys.argv[1])
    else:
        print("Usage: topology-summary-plotter.py $MODEL")

