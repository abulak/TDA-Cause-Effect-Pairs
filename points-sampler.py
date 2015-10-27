import numpy as np
import random
import sys
import os


def workflow(filename, size=1000):
    target_dir = os.path.join(os.getcwd())
    os.chdir('../../')
    raw_data = np.loadtxt(os.path.join(os.getcwd(), 'pairs', filename))
    if size < 1 or raw_data.shape[0] < size:
        orig_points = raw_data
    else:
        indices = np.random.randint(0, raw_data.shape[0], size)
        orig_points = raw_data[indices]
    np.savetxt(os.path.join(target_dir, 'orig_points'), orig_points)

if __name__ == '__main__':

    random.seed(0)

    blacklist = ['pair0023.txt',
                 'pair0033.txt',
                 'pair0037.txt',
                 'pair0047.txt',
                 'pair0070.txt']

    if len(sys.argv) < 4:
        print("Usage: points-sampler.py $FILENAME $SIZE")
    else:
        prefix = sys.argv[1]
        filename = sys.argv[2]
        size = int(sys.argv[3])
        if filename not in blacklist:
            workflow(prefix, filename, size)
        else:
            print(filename, "is blacklisted! (it doesn't fit the model?)")

