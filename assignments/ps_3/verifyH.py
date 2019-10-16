import numpy as np
from utils import to_homogenous_coords, show
import cv2


def verifyH(im1, im2, t1, t2, H):

    h1, w1, _ = im1.shape
    h2, w2, _ = im2.shape
    scale = max(h1, w1, h2, w2) / 2

    n_points = t1.shape[1]

    stacked_im1 = np.zeros((max(h1, h2), w1 + w2, 3))
    stacked_im2 = np.zeros((max(h1, h2), w1 + w2, 3))

    im1_copy1 = im1.copy()
    im2_copy1 = im2.copy()
    im1_copy2 = im1.copy()
    im2_copy2 = im2.copy()

    t1_scaled = t1 / scale
    t1_hom = to_homogenous_coords(t1_scaled)
    t1_prime_hom = np.matmul(H, t1_hom)
    t1_prime = (t1_prime_hom / t1_prime_hom[-1])[:2] * scale

    H_inv = np.linalg.inv(H)
    t2_scaled = t2 / scale
    t2_hom = to_homogenous_coords(t2_scaled)
    t2_prime_hom = np.matmul(H_inv, t2_hom)
    t2_prime = (t2_prime_hom / t2_prime_hom[-1])[:2] * scale

    for i in range(n_points):

        x1, y1 = t1[:, i].astype(int)
        cv2.circle(im1_copy1, (x1, y1), 2, (0, 0, 255), -1)

        x2, y2 = t1_prime[:, i].astype(int)
        cv2.circle(im2_copy1, (x2, y2), 2, (0, 0, 255), -1)

        x3, y3 = t2[:, i].astype(int)
        cv2.circle(im2_copy2, (x3, y3), 2, (0, 0, 255), -1)

        x4, y4 = t2_prime[:, i].astype(int)
        cv2.circle(im1_copy2, (x4, y4), 2, (0, 0, 255), -1)

        #cv2.putText(im1_copy, '{}'.format(i), (x+5, y+5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.75, (0, 0, 255), 1)

    stacked_im1[:h1, :w1, :] = im1_copy1
    stacked_im1[:h2, -w2:, :] = im2_copy1
    #show(stacked_im1.astype('uint8'))

    stacked_im2[:h2, :w2, :] = im2_copy2
    stacked_im2[:h1, -w1:, :] = im1_copy2
    #show(stacked_im2.astype('uint8'))

    return stacked_im1.astype('uint8'), stacked_im2.astype('uint8')
