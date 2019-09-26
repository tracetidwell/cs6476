import math
import cv2
import numpy as np
from scipy.signal import convolve
from utils import round_to, rgb2gray


def detectCircles(im, radius, useGradient, thetaBinSize=0.05, centerBinSize=3,
                  min_val=300, max_val=500):

    H = {}
    tracker = {}

    # gray_im = rgb2gray(im)
    # canny_edges = cv2.Canny(gray_im[85:335, 165:415], 100, 200)
    # rows, cols = np.where(canny_edges==255)
    #
    # if useGradient:
    #     dx = convolve(gray_im[85:335, 165:415], [[1, -1]], 'same')
    #     dy = convolve(gray_im[85:335, 165:415], [[1], [-1]], 'same')

    gray_im = rgb2gray(im)
    canny_edges = cv2.Canny(gray_im, min_val, max_val)
    rows, cols = np.where(canny_edges==255)

    if useGradient:
        dx = convolve(gray_im, [[1, -1]], 'same')
        dy = convolve(gray_im, [[1], [-1]], 'same')


    for x, y in zip(list(cols), list(rows)):
    #for x, y in [(124, 160), (125, 160), (126, 160), (127, 160)]:

        if useGradient:

            #x, y = edge_idxs[i]
            if dx[y, x] == 0:
                theta = 0
            else:
                theta = math.atan(dy[y, x] / dx[y, x])

            a = round_to(x - radius * math.cos(theta), centerBinSize)
            b = round_to(y + radius * math.sin(theta), centerBinSize)

            if H.get((a, b, radius)):
                H[(a, b, radius)] += 1
            else:
                H[(a, b, radius)] = 1

            if tracker.get((a, b, radius)):
                tracker[(a, b, radius)].append((x, y))
            else:
                tracker[((a, b, radius))] = [(x, y)]

        else:

            for theta in np.arange(0, 2.1*math.pi, thetaBinSize*math.pi):

                a = round_to(x - radius * math.cos(theta), centerBinSize)
                b = round_to(y + radius * math.sin(theta), centerBinSize)

                if H.get((a, b, radius)):
                    H[(a, b, radius)] += 1
                else:
                    H[(a, b, radius)] = 1

                if tracker.get((a, b, radius)):
                    tracker[(a, b, radius)].append((x, y))
                else:
                    tracker[((a, b, radius))] = [(x, y)]

    return H, tracker

    #     a = x - r * math.cos(theta)
    #     b = y + r * math.sin(theta)

    #     print('a', a)
    #     print('b', b)

    #     if a - int(a) >= 0.5:
    #         a_s = [np.round(a), np.round(a)-0.5]
    #     else:
    #         a_s = [np.round(a), np.round(a)+0.5]

    #     if b - int(b) >= 0.5:
    #         b_s = [np.round(b), np.round(b)-0.5]
    #     else:
    #         b_s = [np.round(b), np.round(b)+0.5]

    #     for a in a_s:
    #         for b in b_s:

    #             if H.get((a, b, r)):
    #                 H[(a, b, r)] += 1
    #             else:
    #                 H[(a, b, r)] = 1

    #             if tracker.get((a, b, r)):
    #                 tracker[(a, b, r)].append((x, y))
    #             else:
    #                 tracker[((a, b, r))] = [(x, y)]

    #     a = np.round(x - r * math.cos(theta), 2)
    #     b = np.round(y + r * math.sin(theta), 2)



    #     print('a', a)
    #     print('b', b)

    #     for n1 in np.arange(-0.5, 0.51, 0.5):
    #         for n2 in np.arange(-0.5, 0.51, 0.5):

    #             if H.get((a+n1, b+n2, r)):
    #                 H[(a+n1, b+n2, r)] += 1
    #             else:
    #                 H[(a+n1, b+n2, r)] = 1

    #             if tracker.get((a+n1, b+n2, r)):
    #                 tracker[(a+n1, b+n2, r)].append((x, y))
    #             else:
    #                 tracker[((a+n1, b+n2, r))] = [(x, y)]
