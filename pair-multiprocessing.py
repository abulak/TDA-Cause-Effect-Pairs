import sys

import multiprocessing as mproc
import TDA as tda

import logging
import logging.handlers


def initPool(queue, level):
    logger = logging.getLogger('')
    logger.addHandler(logging.handlers.QueueHandler(queue))
    logger.setLevel(level)

class MultiCoreCauseEffectPair(tda.CauseEffectPair):

    def __init__(self, model):

        tda.CauseEffectPair.__init__(self, model)
        # self.queue = mproc.Queue(100)
        # self.logging_listener()


    def logging_listener(self):

        logger = logging.getLogger()

        while True:
            try:
                record = self.queue.get()
                if record is None:
                    break
                logger.handle(record)
            except Exception:
                import traceback
                print('Whoops! Problem:', file=sys.stderr)
                traceback.print_exc(file=sys.stderr)

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

        mproc.set_start_method('forkserver')
        arguments = [i for i in range(len(self.outliers))]
        with mproc.Pool(processes=int(mproc.cpu_count()/2)) as pool:
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
