import numpy as np
import random
import sys
import os
import logging

import TDA as tda
import GeometricComplex as GC

sampler = __import__("point-sampler")


class MultiCoreCauseEffectPair(tda.CauseEffectPair):

    def __init__(self, model):
        tda.CauseEffectPair.__init__(self, model)

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

        import multiprocessing as mproc
        mproc.set_start_method('forkserver')
        arguments = [i for i in range(len(self.outliers))]
        with mproc.Pool(processes=mproc.cpu_count()) as pool:
            results = pool.map(self.single_outlier, arguments)

        self.extrema = [x[0] for x in results]
        self.persistence_pairs = [x[1] for x in results]


def workflow(model):
    p = MultiCoreCauseEffectPair(model)
    p.compute_topological_summary()
    p.save_topological_summary()

if __name__ == "__main__":
    if len(sys.argv) == 2:
        workflow(sys.argv[1])
    else:
        print("Usage: pair-multiprocessing.py $MODEL")







































if __name__ == '__main__':

    random.seed(0)

    blacklist = ['pair0023.txt',
                 'pair0033.txt',
                 'pair0037.txt',
                 'pair0047.txt',
                 'pair0070.txt']

    if len(sys.argv) == 3:
        filename = sys.argv[1]
        size = int(sys.argv[2])
        if filename not in blacklist:
            workflow(filename, size)
        else:
            print(filename, "is blacklisted! (it doesn't fit the model?)")
    else:
        print("Usage: points-sampler.py $FILENAME $SIZE")
