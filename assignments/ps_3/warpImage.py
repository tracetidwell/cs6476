import numpy as np
from utils import to_homogenous_coords


def warpImage(inputIm, refIm, H, blend_col=0, step=0.02, merge_type='blend'):

    h1, w1, _ = inputIm.shape
    h2, w2, _ = refIm.shape

    scale = max(h1, w1, h2, w2) / 2

    H_inv = np.linalg.inv(H)

    # P_prime = np.array([(x, y) for x in range(refIm.shape[1]) for y in range(refIm.shape[0])]).T / scale
    # P_prime_hom = to_homogenous_coords(P_prime)
    # P = np.matmul(H_inv, P_prime_hom)
    # P_w, P_h = np.ceil(np.max((P / P[2])[:2] * scale, axis=1)).astype(int)

    P = np.array([(j, i) for i in range(inputIm.shape[0]) for j in range(inputIm.shape[1])]).T / scale
    P_hom = to_homogenous_coords(P)
    P_prime = np.matmul(H, P_hom)
    P_prime_w, P_prime_h = np.ceil(np.max((P_prime / P_prime[2])[:2] * scale, axis=1)).astype(int)
    #P_prime_w = np.ceil(np.max(((P_prime / P_prime[2]) * scale)[0])).astype(int)
    #P_prime_h = np.ceil(np.max(((P_prime / P_prime[2]) * scale)[1])).astype(int)

    # P_prime_w = int((w1 + w2) * 1.5)
    # P_prime_h = int((h1 + h2) * 1.2)

    #max_h = max(P_h, P_prime_h, h1, h2)
    #max_w = max(P_w, P_prime_w, w1, w2)
    #max_w = 952

    #warpIm = np.zeros((max_h, max_w, 3))
    if P_prime_w > 2 * max(w1, w2) or P_prime_h > 2 * max(h1, h2):
        P_prime_w = 2 * max(w1, w2)
        P_prime_h = 2 * max(h1, h2)

    warpIm = np.zeros((P_prime_h, P_prime_w, 3))

    for x in range(P_prime_w):

        for y in range(P_prime_h):

            p_prime_hom = to_homogenous_coords(np.array([[x], [y]]) / scale)
            p_hom = np.matmul(H_inv, p_prime_hom)
            p = (p_hom / p_hom[-1])[:2] * scale
            x1, y1 = p[0][0], p[1][0]
            i1, j1 = np.floor(y1).astype(int), np.floor(x1).astype(int)
            a = x1 - j1
            b = y1 - i1
            i2 = min(i1 + 1, h1 - 1)
            j2 = min(j1 + 1, w1 - 1)

            if all(p >= 0) and i1 < h1 and j1 < w1:

                warpIm[y, x] = (((1 - a) * (1 - b) * inputIm[i1, j1]) + \
                                (a * (1 - b) * inputIm[i1, j2]) + \
                                ((1 - a) * b * inputIm[i2, j1]) + \
                                (a * b * inputIm[i2, j2])).astype(int)

    while not np.any(warpIm[-1]):
        warpIm = warpIm[:-1, :, :]

    while not np.any(warpIm[:, -1, :]):
        warpIm = warpIm[:, :-1, :]

    if merge_type == 'blend':

        blend_scale = np.arange(1-step, 0, -step)
        blend_len = len(blend_scale)

        if w2 < warpIm.shape[1]:
            im1 = refIm.copy()
            im2 = warpIm.copy()
        else:
            im1 = warpIm.copy()
            im2 = refIm.copy()

        h1, w1, _ = im1.shape
        h2, w2, _ = im2.shape

        if not blend_col:
            blend_col = w1

        h = max(h1, h2)
        w = max(w1, w2)
        mergeIm = np.zeros((h, w, 3))

        # mergeIm[:h2, :w2, :] = refIm
        # mergeIm[:, w2:, :] = warpIm[:, w2:, :]
        # mergeIm[:, w2-blend_len:w2] *= blend_scale.reshape(1, -1)[:, :, np.newaxis]
        # mergeIm[:, w2-blend_len:w2] += warpIm[:, w2-blend_len:w2] * blend_scale[::-1].reshape(1, -1)[:, :, np.newaxis]

        mergeIm[:h1, :blend_col, :] = im1[:, :blend_col, :]
        mergeIm[:h2, blend_col:, :] = im2[:, blend_col:, :]
        mergeIm[:, blend_col-blend_len:blend_col] *= blend_scale.reshape(1, -1)[:, :, np.newaxis]
        mergeIm[:h2, blend_col-blend_len:blend_col] += im2[:, blend_col-blend_len:blend_col] * blend_scale[::-1].reshape(1, -1)[:, :, np.newaxis]

        if h2 > h1:
            mergeIm[h1:h2, :w2, :] = im2[h1:h2, :w2, :]

    elif merge_type == 'frame':

        mergeIm = refIm.copy()

        h1, w1, _ = warpIm.shape

        for i in range(h1):
            for j in range(w1):
                if all(warpIm[i, j] > 0):
                    mergeIm[i, j] = warpIm[i, j]

    return warpIm.astype('uint8'), mergeIm.astype('uint8')
