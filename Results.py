import os
import sys

import numpy as np
import json
import logging

import re

import matplotlib.pyplot as plt

back_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(os.path.join(back_path,
                             "Dionysus-python3/build/bindings/python"))

dionysus = __import__('dionysus')


class Pair:

    empty_diagram = dionysus.PersistenceDiagram(0)
    empty_diagram.append((0, 0))

    def __init__(self, name, path_to_diagrams, metadata):
        if name[-4:] == '.txt':
            name = name[:-4]
        self.name = name
        self.number = int(name[-4:])

        self.metadata = metadata

        if self.metadata[1] == 1:     # i.e. X --> Y
            self.causality_true = 1
        else:                         # i.e. Y --> X
            self.causality_true = -1
        self.weight = self.metadata[5]

        self.x_range = range(int(self.metadata[2] - self.metadata[1]))
        self.y_range = range(int(self.metadata[4] - self.metadata[3]))

        self.prepare_persistence_diagrams(path_to_diagrams)

        self.causality_inferred, self.confidence = self.decide_causality()

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

    def compute_score_stability(self, persistence_diagrams_out, p=0):
        """
        Computes scores stability for a list (outlier indexed) of directions
        :param persistence_diagrams_out: list of the diagrams of direction
                                     (i.e. indexed by outliers)
        :param p: float:    determines which p-Wasserstein metric to use
        :return: list:      of scores
        """

        stability = []
        for out in persistence_diagrams_out:
            scores = []
            for diagram in out:
                d = self.distance(diagram, p)
                scores.append(d)
            max_score = max(scores)

            stability.append(max_score)
        return stability

    def decide_causality(self, weight_function='uniform', p=0):
        """
        Decides pair causality taking into account all persistence scores.
        Returns BOTH: causality and confidence score

        By default we weight all scores uniformly, but You may supply any
        function from range(len(outliers)) -> R as weighting. Built-in can be
        accesed also as strings (l stands for len(outliers)):
        'uniform'   f(x) = 1
        'triangle'  f(x) = -abs(x-l/2) + l/2
        'gaussian'  f(x) = exp( (x-l/2)**2 / 2(l/6)**2
        where 'triangle' and 'gaussian' are
        :param weight_function: function used to weight scores
        :param p: float: to determine Wasserstein distance (0 => bottleneck)
        :return[0]: 0:  if undecided
                    1:  if X -> Y
                   -1:  if Y -> X
        :return[1]: float: confidence score
        """

        self.X_distances = np.array(
            self.compute_score_stability(self.X_diagrams, p))
        self.X_inv_distances = np.array(
            self.compute_score_stability(self.X_inv_diagrams, p))

        self.Y_distances = np.array(
            self.compute_score_stability(self.Y_diagrams, p))
        self.Y_inv_distances = np.array(
            self.compute_score_stability(self.Y_inv_diagrams, p))

        l = self.X_distances.shape[0]
        domain = np.arange(0, l, 1)

        if type(weight_function) == type(lambda x: x):
            pass
        elif weight_function == 'uniform':
            def weight_function(x):
                return 1.0 + 0*x
        elif weight_function == 'triangle':
            def weight_function(x):
                return -np.abs(x - l/2) + l/2
        elif weight_function == 'gaussian':
            def weight_function(x):
                return np.exp(-np.power(x - l/2, 2) /
                              (2 * np.power(l/6, 2)))
        else:
            def weight_function(x):
                return 1.0 + 0*x
            print("unknown_function!, using uniform!")

        w = weight_function(domain)
        weighting = w/sum(w)

        x_integral = np.dot(weighting,
                            np.maximum(self.X_distances, self.X_inv_distances))
        y_integral = np.dot(weighting,
                            np.maximum(self.Y_distances, self.Y_inv_distances))

        confidence = np.abs(x_integral - y_integral)

        if confidence == 0:
            causality = 0
        else:
            if y_integral > x_integral:
                causality = 1
            else:
                causality = -1
        return causality, confidence

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
            p_list = []
            if not diagram:
                diagram = [(0, 0)]
            for pair in diagram:
                    p_list.append(tuple(pair))
            p_diagram = dionysus.PersistenceDiagram(0, p_list)
            persistence_diagrams_list.append(p_diagram)

        return persistence_diagrams_list


class Analysis:

    def __init__(self, prefix='test', outlier_model='knn'):

        try:
            self.metadata = np.loadtxt(
                os.path.join(os.getcwd(), prefix, 'pairs', 'pairmeta.txt'))
        except FileNotFoundError:
            logging.warning("No metadata found! All is set to 0")
            self.metadata = np.zeros((100, 6))

        self.prefix = prefix

        pattern = re.compile('pair[0-9]{4,}$')
        prefix_path = os.path.join(os.getcwd(), prefix)
        dir_list = sorted([x for x in os.listdir(prefix_path)
                           if pattern.match(x)])
        self.pairs_dict = {}
        self.pairs = []

        for directory in dir_list:
            pair_dir = os.path.join(os.getcwd(), prefix, directory)
            if outlier_model == 'knn':
                path = os.path.join(pair_dir, "diagrams_knn")
                if os.path.isfile(path):
                    path_to_diagrams = path
                else:
                    path_to_diagrams = os.path.join(pair_dir, "diagrams.knn")
            elif outlier_model == 'all':
                path_to_diagrams = os.path.join(pair_dir, "diagrams_all")
            else:
                logging.warning("Results for model: %s not computed in %s",
                                str(outlier_model), str(outlier_model))
                path_to_diagrams = ''
            pair_number = int(directory[-4:])
            pair_metadata = self.metadata[pair_number - 1]
            p = Pair(directory, path_to_diagrams, pair_metadata)
            self.pairs.append(p)
            self.pairs_dict[p.name] = p

        self.regenerate_results()

        l = [(p_name, [v.causality_inferred, v.weight, v.confidence,
                       v.causality_true])
             for p_name, v in self.pairs_dict.items()]

        self.readable_summary = sorted(l, key=lambda x: x[1][2], reverse=True)

    def generate_causality_confidence(self):
        result = np.array([
            [pair.causality_inferred, pair.weight, pair.confidence,
             pair.causality_true] for pair in self.pairs])
        return result

    def regenerate_results(self, function='uniform', p=0):

        for pair in self.pairs:
            pair.causality_inferred, pair.confidence = \
                pair.decide_causality(weight_function=function, p=p)

        self.pairs.sort(key=lambda x: x.confidence, reverse=True)
        self.pairs_causality_confidence = self.generate_causality_confidence()

    @staticmethod
    def accuracy_plot(pairs_causality_confidence, label='', threshold=0,
                      **kwargs):
        decisions = pairs_causality_confidence[:, 0]
        ground_truth = pairs_causality_confidence[:, 3]
        decisions_right = np.array([decisions[i] == ground_truth[i] for i in
                                    range(decisions.shape[0])])
        weights = pairs_causality_confidence[:, 1]
        weighted_decisions = decisions_right*weights

        weighted_efficiency = []
        for i in range(decisions.shape[0]):
            entry = sum(weighted_decisions[:i+1])/sum(weights[:i+1])
            weighted_efficiency.append(entry)

        # plt.plot(weighted_efficiency, color='black', alpha=0.3)
        m = len([0 for x in pairs_causality_confidence if x[2] > threshold])
        to_plot = weighted_efficiency[:m]
        # percentage = np.linspace(0, 100, decisions.shape[0])
        # plt.plot(percentage[:m],to_plot, label=label, **kwargs)
        ax = plt.gca()
        ax.plot(to_plot,
                 label="({:02d}, {:04f}) ".format(m, to_plot[-1])+label,
                 **kwargs)
        ax.set_ylim(0., 1.03)
        # print(label, "Decisions taken:", m)
        # print(label, "Final accuracy rate:", to_plot[-1])
        # print(label, "Total weight:", sum(weights[:m-1]))
        return sum(weights[:m-1]), to_plot[-1]

