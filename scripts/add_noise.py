"""
Add noise to data.

Author:
    Ilias Bilionis

Date:
    9/13/2014

"""


import numpy as np
import sys
import os


if __name__ == '__main__':
    if not len(sys.argv) == 3:
        print 'Usage:', sys.argv[0], ' data_file.npy noise'
        quit()
    data_file = sys.argv[1]
    assert os.path.exists(data_file)
    data = np.load(data_file)
    noise = float(sys.argv[2])
    n_data = data + noise * np.random.randn(*data.shape)
    out_file = os.path.abspath(os.path.splitext(data_file)[0] + '_noise')
    print data
    print n_data
    print '- writing', out_file + '.npy'
    np.save(out_file, n_data)
