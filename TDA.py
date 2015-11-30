import numpy as np
import numpy.ma as ma
import os
import sys
import logging

import GeometricComplex as GC

path = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
sys.path.append(os.path.join(path, "Dionysus-python3/build/bindings/python"))


class CauseEffectPair:
    """
    Encapsulates the whole logical concept behind Cause-Effect Pair.
    I.e. contains points, the whole list of outliers, metadata of a pair, etc.
    """

    def __init__(self, model):
        self.current_dir = os.getcwd()
        self.name = self.current_dir[-8:]

        logging.basicConfig(filename=self.name+"_"+model+".log",
                            level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

        logging.info("Starting CauseEffectPair")
        self.model = model
        pairs_dir = os.path.abspath(os.path.join(self.current_dir, os.pardir,
                                                 os.pardir, 'pairs'))
        all_pairs_metadata = np.loadtxt(os.path.join(pairs_dir, 'pairmeta.txt'))
        self.metadata = all_pairs_metadata[int(self.name[-4:]) - 1]
        # metadata is a list of the form
        # 0: pair-number,
        # 1: cause-first-coord, (all starting from one)
        # 2: cause-last-coord,
        # 3: effect-first-coord,
        # 4: effect-last-coord,
        # 5: weight
        if self.metadata[1] ==1:
            self.x_range = range(int(self.metadata[1])-1, int(self.metadata[2]))
            self.y_range = range(int(self.metadata[3])-1, int(self.metadata[4]))
        else:
            self.x_range = range(int(self.metadata[3])-1, int(self.metadata[4]))
            self.y_range = range(int(self.metadata[1])-1, int(self.metadata[2]))

        self.prepare_points()

    def prepare_points(self):

        std_points_file = os.path.join(self.current_dir, "std_points")
        self.std_points = np.loadtxt(std_points_file)
        self.dimension = self.std_points.shape[1]

        outliers_file = os.path.join(self.current_dir, "outliers_" + self.model)

        outliers = np.loadtxt(outliers_file).astype(np.int)
        o, index = np.unique(outliers, return_index=True)
        true_outliers = len(outliers)*[-1]
        for i in index:
            true_outliers[i] = outliers[i]
        self.outliers = np.array([x for x in true_outliers if x != -1],
                                 dtype=int)

    def remove_outliers(self, i):
        """
        :param i: number outliers to remove from self.std_points
        :return: numpy array of std_points without outliers[:i]
        """
        points_masked = ma.MaskedArray(self.std_points)
        for outlier in self.outliers[:i]:
            points_masked[outlier] = ma.masked
        cleaned_points = points_masked.compressed().reshape(
                self.std_points.shape[0] - i, self.dimension)

        return cleaned_points

    def compute_topological_summary(self):
        """
        For each in self.outliers generate cleaned_points. Then construct
        GeometricComplex(cleaned_points) and compute its persistant 0-th
        homology.

        We save the 0-persistence pairs in the _list_
        self.persistence_pairs.

        persistence_pairs[outlier] contains dictionary with self-explaining
        keys:
        x_filtration_H0
        x_inv_filtration_H0
        y_filtration_H0
        y_inv_filtration_H0

        values are lists of persistence pairs
        """
        results = [self.single_outlier(i) for i in range(len(self.outliers))]

        self.extrema = [x[0] for x in results]
        self.persistence_pairs = [x[1] for x in results]

    def single_outlier(self, i):
        logging.info("Outlier: %d of %d", i+1, self.outliers.shape[0])
        # print(i, end=' ', flush=True)
        cleaned_points = self.remove_outliers(i)
        if self.dimension <= 3:
            geometric_cmplx = GC.AlphaGeometricComplex(
                cleaned_points, self.x_range, self.y_range,
                full_initialisation=True)
        else:  # i.e. self.dimension >= 4:
            geometric_cmplx = GC.RipsGeometricComplex(
                cleaned_points, self.x_range, self.y_range,
                full_initialisation=True)

        extrema = {"maxima": list(geometric_cmplx.maxima),
                   "minima": list(geometric_cmplx.minima)}

        persistence_pairs ={
            "X": self.get_homology(geometric_cmplx, 'X',
                                   range(len(self.x_range))),
            "X_inverted": self.get_homology(geometric_cmplx, 'X_inverted',
                                            range(len(self.x_range))),
            "Y": self.get_homology(geometric_cmplx, 'Y',
                                   range(len(self.y_range))),
            "Y_inverted": self.get_homology(geometric_cmplx, 'Y_inverted',
                                            range(len(self.y_range)))}
        logging.info("Outlier: %d of finished!", i+1)
        return extrema, persistence_pairs

    @staticmethod
    def get_homology(g_complex, direction, dim_range):
        """
        Fetches finite-lived (i.e. 'dying') homology pairs of
        g_complex filtered in direction given by 'direction,
        accross the dimension range dim_range (as list)
        :param g_complex: GeometricComplex.GeometricComplex object
        :param direction: string: one of 'X', 'Y', 'X_inverted', 'Y_inverted'
        :param dim_range: iterator: over dimensions in given direction,
                                    starts from 0
        :return: list: (indexed by range) of dying homology pairs
        """
        filtered_complexes = [
            g_complex.filtered_complexes[direction][i]
            for i in dim_range]
        homology = [x.homology_0['dying'] for x in filtered_complexes]
        return homology

    def save_topological_summary(self):
        """
        Saves $model-OutlierModels persistence_pairs to "diagram_$model"
        located in the pairXXXX directory.

        persistence_pairs is an outlier-indexed dictionary of 4-tuples:
        0: x_filtration pairs
        1: x_inv_filtration paris
        2: y_filtration pairs
        3: y_inv_filtration pairs
        """

        self.save_diagrams()
        self.save_extrema()

    def save_diagrams(self, filename='diagrams_'):
        import json
        file = os.path.join(self.current_dir, filename + self.model)
        with open(file, 'w') as f:
            json.dump(self.persistence_pairs, f)
            # for line in self.knn.persistence_pairs:
            #     f.write(line)

    def save_extrema(self, filename='extrema_'):
        import json
        file = os.path.join(self.current_dir, filename + self.model)
        with open(file, 'w') as f:
            json.dump(self.extrema, f)


def workflow(model):
    p = CauseEffectPair(model)
    p.compute_topological_summary()
    p.save_topological_summary()

if __name__ == "__main__":
    if len(sys.argv) == 2:
        workflow(sys.argv[1])
    else:
        print("Usage: TDA.py $MODEL")
