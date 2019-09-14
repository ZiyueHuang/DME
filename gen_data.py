import numpy as np


for i in range(16):
    s = np.random.standard_t(10, size=10000)
    np.savetxt("./data/data_%d.txt" % i, s, delimiter=' ')

