import itertools
import numpy as np
import cv2
from cyvlfeat import sift
from utils import rgb2gray


def on_click(event, x, y, flags, params):

    im, points, counter = params

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(im, (x, y), 2, (0, 0, 255), -1)
        cv2.putText(im, '{}'.format(next(counter)), (x+5, y+5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.75, (0, 0, 255), 1)


def get_points(im, text='image'):

    points = []
    counter = itertools.count()
    cv2.namedWindow(text)
    cv2.setMouseCallback(text, on_click, [im, points, counter])

    while (True):
        cv2.imshow(text, im)
        if cv2.waitKey(20) == 27:
            break

    cv2.destroyAllWindows()

    return im, points


def get_correspondences(im1, im2, method='manual', top_n=15):

    if method == 'manual':

        h1, w1, _ = im1.shape
        h2, w2, _ = im2.shape
        # h = max(h1, h2)
        # w = w1 + w2
        im1_copy = im1.copy()
        im2_copy = im2.copy()

        stacked_im1 = np.zeros((max(h1, h2), w1 + w2, 3))
        stacked_im1[:h1, :w1, :] = im1_copy
        stacked_im1[:h2, -w2:, :] = im2_copy

        counter1 = 0

        im1_copy, im1_points = get_points(stacked_im1.astype('uint8'), 'correspondences 1')
        #im1_points = get_points(im1.astype('uint8'), 'correspondeces 1')

        # for i, (x, y) in enumerate(im1_points):
        #     cv2.circle(im1_copy, (x, y), 2, (0, 0, 255), -1)
        #     cv2.putText(im1_copy, '{}'.format(i), (x+5, y+5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.75, (0, 0, 255), 1)

        stacked_im2 = np.zeros((stacked_im1.shape))
        stacked_im2[:h2, :w2, :] = im2_copy
        stacked_im2[:h1, -w1:, :] = im1_copy[:h1, :w1, :]

        _, im2_points = get_points(stacked_im2.astype('uint8'), 'correspondences 2')

        # for i, im in enumerate([im1, im2]):
        #
        #     points = []
        #     cv2.namedWindow('image')
        #     cv2.setMouseCallback('image', on_click, points)
        #
        #     while (True):
        #         cv2.imshow('image', im)
        #         if cv2.waitKey(20) == 27:
        #             break
        #
        #     cv2.destroyAllWindows()
        #
        #     out.append(list(points))

        #return np.array(out[0]).T.astype(float), np.array(out[1]).T.astype(float)
        return np.array(im1_points).T.astype(float), np.array(im2_points).T.astype(float)

    elif method == 'auto':

        gray1 = rgb2gray(im1)
        gray2 = rgb2gray(im2)

        kp1, des1 = sift.sift(gray1, compute_descriptor=True)
        kp2, des2 = sift.sift(gray2, compute_descriptor=True)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        matches = sorted(matches, key = lambda x:x.distance)

        c1 = np.zeros((2, top_n))
        c2 = np.zeros((2, top_n))

        for i in range(top_n):
            c1[:, i] = kp1[matches[i].queryIdx][:2][::-1]
            c2[:, i] = kp2[matches[i].trainIdx][:2][::-1]

        return c1, c2
