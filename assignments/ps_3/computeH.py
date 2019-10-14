import numpy as np
from utils import to_homogenous_coords


def computeH(t1, t2):

    n_points = t1.shape[1]
    #t1 = np.concatenate([t1, [[1] * n_points]], axis=0)
    t1 = to_homogenous_coords(t1)

    L = []
    for i in range(n_points):
        L.append(np.concatenate([t1[:, i], [0] * 3, -t2[0, i] * t1[:, i]]))
        L.append(np.concatenate([[0] * 3, t1[:, i], -t2[1, i] * t1[:, i]]))

    L = np.array(L)
    s, u, v = np.linalg.svd(L)
    H = v[-1].reshape(3, 3)

    # A = np.matmul(L.T, L)
    # values, vectors = np.linalg.eig(A)
    # H = vectors[np.argmin(values)].reshape(3, 3)

    return H
