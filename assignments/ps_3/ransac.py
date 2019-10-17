import numpy as np
from sklearn.metrics import mean_squared_error
from computeH import computeH
from utils import to_homogenous_coords


def ransac(t1, t2, scale, max_iter=100):

    min_error = float('inf')
    n_iter = 0
    best_t1 = t1
    best_t2 = t2

    t1_scaled = t1 / scale
    t2_scaled = t2 / scale

    while n_iter < max_iter:

        choice = np.random.choice(range(t1_scaled.shape[1]), 4, replace=False)

        t1_subset = t1_scaled[:, choice]
        t2_subset = t2_scaled[:, choice]

        H = computeH(t1_subset, t2_subset)
        t1_hom = to_homogenous_coords(t1_subset)
        t1_prime_hom = np.matmul(H, t1_hom)
        t1_prime = (t1_prime_hom / t1_prime_hom[-1])[:2]# * scale

        H_inv = np.linalg.inv(H)
        t2_hom = to_homogenous_coords(t2_subset)
        t2_prime_hom = np.matmul(H_inv, t2_hom)
        t2_prime = (t2_prime_hom / t2_prime_hom[-1])[:2]# * scale

        error = np.sqrt(mean_squared_error(t1_subset, t2_prime)) + np.sqrt(mean_squared_error(t2_subset, t1_prime))

        if error < min_error:
            min_error = error
            best_t1 = t1_subset.copy()
            best_t2 = t2_subset.copy()

        n_iter += 1

    return best_t1 * scale, best_t2 * scale
