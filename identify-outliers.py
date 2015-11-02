import os
import sys

import numpy as np
import numpy.ma as ma
import logging

from sklearn.neighbors import NearestNeighbors
import scipy.spatial as spsp


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

        self.points = np.loadtxt(os.path.join(self.current_dir, 'std_points'))
        self.model = model
        self.dimension = self.points[0].shape[0]

        self.n_of_outliers = int(15 * self.points.shape[0] / 100.0)

    def find_outliers_knn(self, k_nearest):

        logging.info("Finding 'knn' %d outliers in %s", self.n_of_outliers,
                     self.name)
        neigh = NearestNeighbors()
        neigh.fit(self.points)
        distances, indices = neigh.kneighbors(self.points,
                                              k_nearest + self.n_of_outliers)
        self.outliers = []

        for out in range(self.n_of_outliers):
            logging.debug("%d of %d", out, self.n_of_outliers)
            distances_partial = distances[:, 1:k_nearest+1]
            distances_vector = distances_partial.sum(1)
            outlier = distances_vector.argmax()
            self.outliers.append(outlier)

            distances[outlier] = np.zeros(k_nearest + self.n_of_outliers)

            for i, row in enumerate(indices):
                if outlier in row:
                    distances[i][np.where(row == outlier)[0][0]] = 1000
                    distances[i].sort()
        return self.outliers

    def find_outliers_all(self):

        logging.info("Finding 'all' %d outliers in %s", self.n_of_outliers,
                     self.name)

        distances_matrix = spsp.distance_matrix(self.points, self.points)
        self.outliers = []

        distances_vector = ma.masked_array(np.sum(distances_matrix, axis=1))
        for out in range(self.n_of_outliers):
            outlier = distances_vector.argmax()
            logging.debug("%d of %d", self.n_of_outliers, out)
            self.outliers.append(outlier)
            distances_vector -= distances_matrix[:, outlier]
            distances_vector[outlier] = ma.masked
        return self.outliers

    def find_outliers(self):
        """Procedure finding outliers based on nearest neighbours.
        if neighbours == 0 then all other points are taken into the account
        Outliers (their indexes in self.points) are stored in self.outliers"""

        if self.model == 'all':  # outlier based on max distance to all others
            self.outliers = self.find_outliers_all()

        if self.model == 'knn':
            nearest_neighbours = 2 * int(self.points.shape[0] / 100) + 2
            self.outliers = self.find_outliers_knn(nearest_neighbours)

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
