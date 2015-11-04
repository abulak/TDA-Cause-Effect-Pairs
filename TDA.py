import numpy as np
import numpy.ma as ma
import os
import sys
import logging

import GeometricComplex as GC

path = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir,
                                    os.pardir))
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
        self.x_range = range(int(self.metadata[1])-1, int(self.metadata[2]))
        self.y_range = range(int(self.metadata[3])-1, int(self.metadata[4]))

        self.prepare_points()

        self.compute_topological_summary()

    def prepare_points(self):

        std_points_file = os.path.join(self.current_dir, "std_points")
        self.std_points = np.loadtxt(std_points_file)
        self.dimension = int(self.std_points.shape[1])

        outliers_file = os.path.join(self.current_dir, "outliers_" + self.model)
        self.outliers = np.loadtxt(outliers_file).astype(np.int)

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

        values are arrays of persistance pairs
        """

        self.persistence_pairs = []
        self.extrema = []
        for i, outlier in enumerate(self.outliers, start=1):
            logging.info("Outlier: %d of %d", i, self.outliers.shape[0])

            cleaned_points = self.remove_outliers(i)
            if self.dimension <= 3:
                self.geometric_complex = GC.AlphaGeometricComplex(
                    cleaned_points, self.x_range, self.y_range,
                    full_initialisation=True)
            elif self.dimension >= 4:
                self.geometric_complex = GC.RipsGeometricComplex(
                    cleaned_points, self.x_range, self.y_range,
                    full_initialisation=True)

            self.extrema.append({
                "maxima": list(self.geometric_complex.maxima),
                "minima": list(self.geometric_complex.minima)
                })

            self.persistence_pairs.append(
                {"x_filtration_H0":
                    self.get_homology('X', range(len(self.x_range))),
                 "x_inv_filtration_H0":
                    self.get_homology('X_inverted', range(len(self.x_range))),
                 "y_filtration_H0":
                    self.get_homology('Y', range(len(self.y_range))),
                 "y_inv_filtration_H0":
                    self.get_homology('Y_inverted', range(len(self.y_range)))})

    def get_homology(self, key, range):
        filtered_complexes = [
            self.geometric_complex.filtered_complexes[key][i] for i in range]
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
    p.save_topological_summary()

if __name__ == "__main__":
    if len(sys.argv) == 2:
        workflow(sys.argv[1])
    else:
        print("Usage: TDA.py $MODEL")
