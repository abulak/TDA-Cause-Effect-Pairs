import os
import sys

import numpy as np
import json
import logging

back_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(os.path.join(back_path,
                             "Dionysus-python3/build/bindings/python"))

dionysus = __import__('dionysus')


class PairsResults:
    try:
        all_pairs_metadata = np.loadtxt(
            os.path.join(os.getcwd(), 'pairs', 'pairmeta.txt'))
    except FileNotFoundError:
        logging.warning("No metadata found! All is set to 0")
        all_pairs_metadata = np.zeros((88, 6))

    empty_diagram = dionysus.PersistenceDiagram(0)
    empty_diagram.append((0, 0))

    def __init__(self, name, path_to_diagrams, threshold=0):
        if name[-4:] == '.txt':
            name = name[:-4]
        self.name = name
        self.number = int(name[-4:])

        self.metadata = self.all_pairs_metadata[self.number-1]

        if self.metadata[1] == 1:     # i.e. X --> Y
            self.causality_true = 1
        else:                         # i.e. Y --> X
            self.causality_true = -1
        self.weight = self.metadata[5]

        self.x_range = range(int(self.metadata[2] - self.metadata[1]))
        self.y_range = range(int(self.metadata[4] - self.metadata[3]))

        self.prepare_persistence_diagrams(path_to_diagrams)

        self.causality_inferred = self.decide_causality(threshold)

    def prepare_persistence_diagrams(self, path):
        """
        Turns json dump (path) of persistence diagrams into easily
        accessible attributes.
        Each Object has diags accessible in X, Y, X_inverted, Y_inverted
        attribute.
        :param path: str:   path to json-dumped TDA.CEPair.persistence_pair
                            (e.g. by TDA.CEPair.save_diagrams())
        :return: none
        """

        with open(path, 'r') as diagrams:
            persistence_list = json.load(diagrams)

        persistence_diagrams = [OutlierPersistence(i) for i in persistence_list]
        self.persistence_diagrams = persistence_diagrams

        self.X_diagrams = [out.X for out in persistence_diagrams]
        self.Y_diagrams = [out.Y for out in persistence_diagrams]
        self.X_inv_diagrams = [out.X_inverted
                               for out in persistence_diagrams]
        self.Y_inv_diagrams = [out.Y_inverted
                               for out in persistence_diagrams]

    def compute_score_stability(self, persistence_diagrams, p=0):
        """
        Computes scores stability for a list (outlier indexed) of directions
        :param persistence_diagrams: list of the diagrams of direction
                                     (i.e. indexed by outliers)
        :param p: float:    determines which p-Wasserstein metric to use
        :return: list:      of scores
        """

        stability = []
        for _ in persistence_diagrams:
            scores = []
            for diagram in persistence_diagrams:
                scores.append(self.distance(diagram, p))
            max_score = max(scores)

            stability.append(max_score)
        return stability

    def decide_causality(self, threshold=0):
        """
        Decides pair causality taking into account all persistence diagrams.
        :param threshold: float:    if confidence is below the threshold do not
                                    decide the causality
        :return: 0:  if undecided
                 1:  if X -> Y
                -1:  if Y -> X
        """

        self.X_distances = np.array(
            self.compute_score_stability(self.X_diagrams))
        self.X_inv_distances = np.array(
            self.compute_score_stability(self.X_inv_diagrams))

        self.Y_distances = np.array(
            self.compute_score_stability(self.Y_diagrams))
        self.Y_inv_distances = np.array(
            self.compute_score_stability(self.Y_inv_diagrams))

        l = self.X_distances.shape[0]

        def weighting_function(x):
            # return 1.0
            return -np.abs(x - l/2) + l/2
            # return np.exp(-np.power(x - l/2, 2.0) / (2 * np.power(l/6, 2.0)))

        domain = np.arange(0, l, 1)
        w = weighting_function(domain)
        weighting = w/sum(w)

        x_integral = np.dot(weighting,
                            np.maximum(self.X_distances, self.X_inv_distances))
        y_integral = np.dot(weighting,
                            np.maximum(self.Y_distances, self.Y_inv_distances))

        self.confidence = np.abs(x_integral - y_integral)

        if self.confidence <= threshold:
            causality = 0
        else:
            causality = int((y_integral - x_integral)/np.abs(y_integral -
                                                             x_integral))
        return causality

    def distance(self, persistence_diagram, p=0):
        """
        Returns p-th Wasserstein distance between the persistence 0-diagram
        and the empty diagram.
        If p=0 then the bottleneck is returned.

        :param persistence_diagram: dionysus.PersistenceDiagram
        :param p: float:            p parameter of the norm
        :return: float:             distance between the diagram and the empty
                                    diagram
        """

        if p > 0:
            return dionysus.wasserstein_distance(
                persistence_diagram, self.empty_diagram, p)
        else:
            return dionysus.bottleneck_distance(
                persistence_diagram, self.empty_diagram)


class OutlierPersistence:

    def __init__(self, diagrams_for_outlier):

        self.X = self.diagrams_of_direction('X', diagrams_for_outlier)
        self.X_inverted = self.diagrams_of_direction('X_inverted',
                                                     diagrams_for_outlier)
        self.Y = self.diagrams_of_direction('Y', diagrams_for_outlier)
        self.Y_inverted = self.diagrams_of_direction('Y_inverted',
                                                     diagrams_for_outlier)

    @staticmethod
    def diagrams_of_direction(direction, diagrams_dict):
        """
        :param direction: one of the {'X', 'X_inverted', 'Y', 'Y_inverted'}
        :param diagrams_dict: dictionary (indexed by directions) of projections
        :return: list of dionysus.PersistenceDiagram for every projection
        """
        projection_list = diagrams_dict[direction]
        persistence_diagrams_list = []
        for diagram in projection_list:
            p_diagram = dionysus.PersistenceDiagram(0)
            if not diagram:
                diagram = [(0, 0)]
            for pair in diagram:
                    p_diagram.append(tuple(pair))
            persistence_diagrams_list.append(p_diagram)

        return persistence_diagrams_list
