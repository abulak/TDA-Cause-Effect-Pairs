import numpy as np
import numpy.ma as ma

import os

import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class PairOutlierPlotter:
    """
    Outlier Plotter; requires outliers model and assumes outliers_$model and
    orig_points are in the working directory
    """

    def __init__(self, model):
        self.name = os.getcwd()[-8:]
        self.suffix = str(model)
        outliers_path = os.path.join(os.getcwd(), 'outliers_' + self.suffix)
        self.outliers = np.loadtxt(outliers_path, dtype=np.int)
        self.points = np.loadtxt(os.path.join(os.getcwd(), 'orig_points'))

    def plot_outlier(self, i):
        masked_points = ma.masked_array(self.points)
        if i < 0:
            plt.title(self.name)
            plt.scatter(self.points[:, 0], self.points[:, 1], color='black',
                        alpha=0.7, s=15)
        else:
            outs = self.outliers[:i+1]
            removed_points = self.points[self.outliers[:i + 1]]
            masked_points[outs] = ma.masked
            to_plot = masked_points.compressed().reshape(
                self.points.shape[0] - i - 1, 2)
            plt.title(self.name + ", outlier: " + str(i+1))
            plt.scatter(to_plot[:, 0], to_plot[:, 1],
                        color='black', alpha=0.7, s=15)
            plt.scatter(removed_points[:, 0], removed_points[:, 1],
                        color='red', alpha=0.7, s=15)

    def save_plots_pdf(self):
        print("Generating outlier plots of ", self.name, "for model:",
              self.suffix)

        pdf_file = os.path.join(os.getcwd(), 'outliers_' + self.suffix + '.pdf')
        with PdfPages(pdf_file) as pdf:
            for i in range(-1, self.outliers.shape[0]):
                plt.figure(figsize=(12, 12))
                self.plot_outlier(i)
                pdf.savefig()
                plt.close()
        print("Done:", self.name, self.suffix)

if __name__ == "__main__":
    p = PairOutlierPlotter("all")
    p.save_plots_pdf()
    p = PairOutlierPlotter("knn")
    p.save_plots_pdf()
