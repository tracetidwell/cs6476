import math
import cv2
import numpy as np
from scipy.signal import convolve
from utils import round_to, rgb2gray, get_votes


def detectCircles(im, radius, useGradient, thetaBinSize=0.05, centerBinSize=2,
                  min_val=300, max_val=500):

    H = {}
    tracker = {}
    gray_im = rgb2gray(im)
    canny_edges = cv2.Canny(gray_im, min_val, max_val)
    rows, cols = np.where(canny_edges==255)

    if useGradient:
        dx = convolve(gray_im, [[1, -1]], 'same')
        dy = convolve(gray_im, [[1], [-1]], 'same')


    for x, y in zip(list(cols), list(rows)):

        if useGradient:
            if dx[y, x] == 0:
                thetas = [0]
            else:
                thetas = [math.atan(dy[y, x] / dx[y, x])]
        else:
            thetas = np.arange(0, 2.1*math.pi, thetaBinSize*math.pi)

        for theta in thetas:

            a = x - radius * math.cos(theta)
            b = y + radius * math.sin(theta)

            a_votes = get_votes(a, centerBinSize)
            b_votes = get_votes(b, centerBinSize)

            for a in a_votes:

                for b in b_votes:

                    if H.get((a, b, radius)):
                        H[(a, b, radius)] += 1
                    else:
                        H[(a, b, radius)] = 1

                    if tracker.get((a, b, radius)):
                        tracker[(a, b, radius)].append((x, y))
                    else:
                        tracker[((a, b, radius))] = [(x, y)]

    return H#, tracker
