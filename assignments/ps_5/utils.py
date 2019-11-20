import numpy as np
from scipy.spatial import distance


def get_rows_cols(n):

    cols = [i for i in range(1, int(np.sqrt(n)) + 1) if n % i == 0][-1]
    rows = n // cols

    return rows, cols


def normalized_euclidean_distance(u, v, variances):

    return np.sqrt(np.sum((u - v)**2 / variances))


def train_test_split(data, test_split_pct=0.05):

    test_size = int(len(data) * test_split_pct)
    np.random.shuffle(data)

    train_data, test_data = data[:-test_size], data[-test_size:]
    x_train, y_train = train_data[:, :-1], train_data[:, -1:]
    x_train, train_idxs = x_train[:, :-1], x_train[:, -1:]
    x_test, y_test = test_data[:, :-1], test_data[:, -1:]
    x_test, test_idxs = x_test[:, :-1], x_test[:, -1:]

    return x_train, y_train, train_idxs, x_test, y_test, test_idxs


def image_moment(im, i, j, scaled=True):

    h, w = im.shape
    m = 0

    if scaled:
        for x in range(w):
            for y in range(h):
                m += (x/w)**i * (y/h)**j * im[y, x]
    else:
        for x in range(w):
            for y in range(h):
                m += x**i * y**j * im[y, x]

    return m


def central_moment(im, p, q, x_bar, y_bar, scaled=True):

    h, w = im.shape
    mean = 0

    if scaled:
        for x in range(w):
            for y in range(h):
                mean += ((x/w) - x_bar)**p * ((y/h) - y_bar)**q * im[y, x]
    else:
        for x in range(w):
            for y in range(h):
                mean += (x - x_bar)**p * (y - y_bar)**q * im[y, x]

    return mean


def h1(im, x_bar, y_bar, scaled=True):

    mu_20 = central_moment(im, 2, 0, x_bar, y_bar, scaled)
    mu_02 = central_moment(im, 0, 2, x_bar, y_bar, scaled)

    return mu_20 + mu_02


def h2(im, x_bar, y_bar, scaled=True):

    mu_20 = central_moment(im, 2, 0, x_bar, y_bar, scaled)
    mu_02 = central_moment(im, 0, 2, x_bar, y_bar, scaled)
    mu_11 = central_moment(im, 1, 1, x_bar, y_bar, scaled)

    return  (mu_20 - mu_02)**2 + 4 * mu_11**2


def h3(im, x_bar, y_bar, scaled=True):

    mu_30 = central_moment(im, 3, 0, x_bar, y_bar, scaled)
    mu_12 = central_moment(im, 1, 2, x_bar, y_bar, scaled)
    mu_21 = central_moment(im, 2, 1, x_bar, y_bar, scaled)
    mu_03 = central_moment(im, 0, 3, x_bar, y_bar, scaled)

    return  (mu_30 - 3 * mu_12)**2 + (3 * mu_21 - mu_03)**2


def h4(im, x_bar, y_bar, scaled=True):

    mu_30 = central_moment(im, 3, 0, x_bar, y_bar, scaled)
    mu_12 = central_moment(im, 1, 2, x_bar, y_bar, scaled)
    mu_21 = central_moment(im, 2, 1, x_bar, y_bar, scaled)
    mu_03 = central_moment(im, 0, 3, x_bar, y_bar, scaled)

    return  (mu_30 + mu_12)**2 + (mu_21 + mu_03)**2


def h5(im, x_bar, y_bar, scaled=True):

    mu_30 = central_moment(im, 3, 0, x_bar, y_bar, scaled)
    mu_12 = central_moment(im, 1, 2, x_bar, y_bar, scaled)
    mu_21 = central_moment(im, 2, 1, x_bar, y_bar, scaled)
    mu_03 = central_moment(im, 0, 3, x_bar, y_bar, scaled)

    return  (mu_30 - 3 * mu_12) * (mu_30 + mu_12) * ((mu_30 + mu_12)**2 - 3 * (mu_21 + mu_03)**2) + \
            (3 * mu_21 - mu_03) * (mu_21 + mu_03) * (3 * (mu_30 + mu_12)**2 - (mu_21 + mu_03)**2)


def h6(im, x_bar, y_bar, scaled=True):

    mu_20 = central_moment(im, 2, 0, x_bar, y_bar, scaled)
    mu_02 = central_moment(im, 0, 2, x_bar, y_bar, scaled)
    mu_30 = central_moment(im, 3, 0, x_bar, y_bar, scaled)
    mu_12 = central_moment(im, 1, 2, x_bar, y_bar, scaled)
    mu_21 = central_moment(im, 2, 1, x_bar, y_bar, scaled)
    mu_03 = central_moment(im, 0, 3, x_bar, y_bar, scaled)
    mu_11 = central_moment(im, 1, 1, x_bar, y_bar, scaled)

    return (mu_20 - mu_02) * ((mu_30 + mu_12)**2 - (mu_21 + mu_03)**2) + \
           4 * mu_11 * (mu_30 + mu_12) * (mu_21 + mu_03)


def h7(im, x_bar, y_bar, scaled=True):

    mu_21 = central_moment(im, 2, 1, x_bar, y_bar, scaled)
    mu_03 = central_moment(im, 0, 3, x_bar, y_bar, scaled)
    mu_30 = central_moment(im, 3, 0, x_bar, y_bar, scaled)
    mu_12 = central_moment(im, 1, 2, x_bar, y_bar, scaled)

    return (3 * mu_21 - mu_03) * (mu_30 * mu_12) * ((mu_30 + mu_12)**2 - 3 * (mu_21 + mu_03)**2) - \
           (mu_30 - 3 * mu_12) * (mu_21 + mu_03) * (3 * (mu_30 + mu_12)**2 - (mu_21 + mu_03)**2)
