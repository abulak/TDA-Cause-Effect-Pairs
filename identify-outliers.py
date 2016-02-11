import os
import sys

import numpy as np
import numpy.ma as ma
import logging

from sklearn.neighbors import NearestNeighbors
import scipy.spatial as spsp


def standardise(points):
    """
    Standardise points, i.e. mean = 0 and standard deviation = 1 in both
    dimensions
    :param points: np.array
    :return: np.array
    """
    for i in range(points.shape[1]):
        p = points[:, i]
        mean = np.mean(p)
        std = np.std(p)
        p -= mean
        if std:
            p /= std
    return points


class OutlierRemoval:

    """A Class encapsulating everything what we do to the pairs-data
    string      self.name           file name
    np.array    self.raw_data       data loaded from file
                self.orig_points    (potentially) sampled points
                self.points         points we use for all computations
                                    watch out using them!
                self.cleaned_points orig_points - outliers
    list        self.outliers       list of indices of detected outlers in
                                    orig_points
    string      self.target_dir     (path) where to save all the points/outliers
    """

    def __init__(self, model):
        self.current_dir = os.getcwd()
        self.name = self.current_dir[-8:]

        logging.basicConfig(filename=self.name+".log", level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

        points = np.loadtxt(os.path.join(self.current_dir, 'std_points'))
        self.points = standardise(points)
        self.model = model
        self.dimension = self.points.shape[1]

        self.n_of_outliers = int(15 * self.points.shape[0] / 100.0)

    @staticmethod
    def find_single_outlier_knn(points, k_nearest):
        neigh = NearestNeighbors()
        neigh.fit(points)

        distances, indices = neigh.kneighbors(points, k_nearest)
        distances_partial = distances[:, 1:k_nearest+1]
        distances_vector = distances_partial.sum(1)
        outlier = distances_vector.argmax()

        return outlier

    @staticmethod
    def compute_offset(outliers, i):
        offset = 0
        for j, x in reversed(list(enumerate(outliers[:i]))):
            if x <= outliers[i]+offset:
                offset += 1

        return offset

    def find_outliers_knn(self, k_nearest):
        masked_points = ma.MaskedArray(self.points)
        shape = self.points.shape
        outliers = []
        for i in range(self.n_of_outliers):
            pts = masked_points.compressed().reshape((shape[0] - i,
                                                      self.dimension))
            pts = standardise(pts)
            out_index = self.find_single_outlier_knn(pts, k_nearest)
            outliers.append(out_index)
            masked_points = ma.MaskedArray(pts)
            masked_points[out_index] = ma.masked
            logging.debug("%d of %d", out_index, self.n_of_outliers)

        offsets = [self.compute_offset(outliers, i)
                   for i in range(len(outliers))]
        true_outliers = [out + offsets[i] for i, out in enumerate(outliers)]
        if len(true_outliers) != len(set(true_outliers)):
            logging.warning("There are some duplicates in the outliers! %s",
                            str(true_outliers))
        # assert len(true_outliers) == len(set(true_outliers)), \
        #     "There are some duplicates in the outliers!"
        return true_outliers

    def find_outliers_knn_old(self, k_nearest):

        neigh = NearestNeighbors()
        neigh.fit(self.points)
        distances, indices = neigh.kneighbors(self.points,
                                              k_nearest + self.n_of_outliers)
        outliers = []

        for out in range(self.n_of_outliers):
            logging.debug("%d of %d", out, self.n_of_outliers)
            distances_partial = distances[:, 1:k_nearest+1]
            distances_vector = distances_partial.sum(1)
            outlier = distances_vector.argmax()
            outliers.append(outlier)

            distances[outlier] = np.zeros(k_nearest + self.n_of_outliers)

            for i, row in enumerate(indices):
                if outlier in row:
                    distances[i][np.where(row == outlier)[0][0]] = 1000
                    distances[i].sort()
        return outliers

    def find_outliers_all(self):

        distances_matrix = spsp.distance_matrix(self.points, self.points)
        outliers = []

        distances_vector = ma.masked_array(np.sum(distances_matrix, axis=1))
        for out in range(self.n_of_outliers):
            outlier = distances_vector.argmax()
            logging.debug("%d of %d", self.n_of_outliers, out)
            outliers.append(outlier)
            distances_vector -= distances_matrix[:, outlier]
            distances_vector[outlier] = ma.masked
        return outliers

    def find_outliers(self):
        """Procedure finding outliers based on nearest neighbours.
        if neighbours == 0 then all other points are taken into the account
        Outliers (their indexes in self.points) are stored in self.outliers"""

        logging.info("Finding %s %d outliers in %s", self.model,
                     self.n_of_outliers, self.name)

        nearest_neighbours = int(2 * self.points.shape[0] / 100) + 2

        if self.model == 'all':  # outlier based on max distance to all others
            self.outliers = self.find_outliers_all()

        elif self.model == 'knn_old':
            self.outliers = self.find_outliers_knn_old(nearest_neighbours)
        elif self.model == 'knn':
            self.outliers = self.find_outliers_knn(nearest_neighbours)
        else:
            logging.warning('Unknown model of outliers! Available are: all, '
                            'knn_old, knn')
        logging.info('Done with outliers!')

    def save_outliers(self):
        np.savetxt(os.path.join(self.current_dir, "outliers_" + self.model),
                   np.asarray(self.outliers, dtype=int), fmt='%d')


def workflow(model):
    p = OutlierRemoval(model)
    p.find_outliers()
    p.save_outliers()

if __name__ == "__main__":

    if len(sys.argv) == 2:
        model = sys.argv[1]
        workflow(model)
    else:
        print("Usage: identify-outliers.py $MODEL")
