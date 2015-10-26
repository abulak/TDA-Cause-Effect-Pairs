import numpy as np
import numpy.ma as ma

import os

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

        os.chdir(os.path.join(os.getcwd(),'test','pair0001'))
        self.name = os.getcwd()[-8:]
        self.suffix = str(model)
        self.points = np.loadtxt(os.path.join(os.getcwd(), 'orig_points'))

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

    def __find_extrema__(self):
        lst = []
        for i in range(self.outliers.shape[0]):
            masked_points = ma.masked_array(self.points)

            outs = self.outliers[:i+1]
            masked_points[outs] = ma.masked
            cleaned_points = masked_points.compressed().reshape(
                    self.points.shape[0] - i - 1, 2)
            xmin = np.amin(cleaned_points[:,0])
            xmax = np.amax(cleaned_points[:,0])
            ymin = np.amin(cleaned_points[:,1])
            ymax = np.amax(cleaned_points[:,1])
            lst.append({"x": [xmin, xmax], "y": [ymin, ymax]})
        return lst

    def plot_scores(self):

        plt.title(self.name + ' ' + self.suffix + " scores")
        plt.plot(self.scores[0], label='x->y')
        plt.plot(self.scores[1], label='y->x')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2,
                   mode="expand", borderaxespad=0.)

    def plot_diagram(self, i, filtration, direction=''):

        ax = plt.gca()
        ax.set(aspect='equal')

        if direction == 'x':
            minimal = self.extrema[i]['x'][0]
            maximal = self.extrema[i]['x'][1]
        if direction == 'y':
            minimal = self.extrema[i]['y'][0]
            maximal = self.extrema[i]['y'][1]
        plt.plot([minimal, maximal], [minimal, maximal],
                     color='black', alpha=0.1)
        points = np.array(self.diagrams[i][filtration])

        if points.shape[0]: # if the array is not empty...
            if 'inv' in filtration:
                points *= -1
                points += [maximal, maximal]
            else:
                points += [minimal, minimal]

            plt.scatter(points[:, 0], points[:, 1],
                    marker='+', facecolors='none', edgecolors='r')
        plt.title(filtration)

    def plot_all_diagrams(self, i):
        plt.title(self.name + " " + self.suffix + " outlier: " + str(i))

        plt.subplot(321)
        to_plot = self.__mask_points__(i)
        plt.scatter(to_plot[:, 0], to_plot[:, 1],
                        color='black', alpha=0.7)

        # plt.subplot(322)

        plt.subplot(323)
        plt.title("x_filtration_H0")
        self.plot_diagram(i, filtration="x_filtration_H0", direction='x')
        plt.subplot(324)
        plt.title("x_inv_filtration_H0")
        self.plot_diagram(i, filtration="x_inv_filtration_H0", direction='x')
        plt.subplot(325)
        plt.title("y_filtration_H0")
        self.plot_diagram(i, filtration="y_filtration_H0", direction='y')
        plt.subplot(326)
        plt.title("y_inv_filtration_H0")
        self.plot_diagram(i, filtration="y_inv_filtration_H0", direction='y')


if __name__ == "__main__":
    pdf_file = os.path.join(os.getcwd(), 'scores.pdf')
    with PdfPages(pdf_file) as pdf:
        plt.figure(figsize=(12, 12))
        p = PairTopologyPlotter("all")
        p.plot_scores()
        pdf.savefig()
        plt.close()

        p = PairTopologyPlotter("knn")
        p.plot_scores()
        pdf.savefig()
        plt.close()
