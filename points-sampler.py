import numpy as np
import random
import sys
import os
import logging


def standardise(points):
    """Standardise self.points, i.e.
    mean = 0 and standard deviation = 1 in both dimensions"""
    for i in range(points[0].shape[0]):
        p = points[:, i]
        mean = np.mean(p)
        std = np.std(p)
        p -= mean
        p /= std
    return points


def workflow(filename, size=1000):

    logging.basicConfig(filename=filename[:-4]+".log",
                        filemode='w',
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info("Sampling up to %d points from %s", size, filename)
    target_dir = os.path.join(os.getcwd())

    os.chdir('../../')
    raw_data = np.loadtxt(os.path.join(os.getcwd(), 'pairs', filename))
    dimension = raw_data.shape[1]
    if size < 1 or raw_data.shape[0] < size:
        logging.info("Data contains less points than %d. Using all available.",
                     size)
        points = raw_data
    else:
        np.random.seed(0)
        logging.info("Data contains more points than %d. Sampling.",
                     size)
        indices = np.random.randint(0, raw_data.shape[0], size)
        points = raw_data[indices]

    for i in range(dimension):
        column_blacklist = []
        if np.isnan(np.dot(points[:,i],points[:,i])):
            logging.info("Column %d contains Nan(s). Discarding", i)
            column_blacklist.append(i)
    if column_blacklist:
        import numpy.ma as ma
        masked_points = ma.MaskedArray(points)
        for i in column_blacklist:
            masked_points[:, i] = ma.masked
        logging.info("Discarded %d columns in total", len(column_blacklist))
        new_shape = (raw_data.shape[0], raw_data.shape[1]-len(column_blacklist))
        points = masked_points.compressed().reshape(new_shape)

    std_points = standardise(points)
    np.savetxt(os.path.join(target_dir, 'std_points'), std_points)
    logging.info("Sampling Done!")

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
