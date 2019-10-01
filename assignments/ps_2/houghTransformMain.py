import argparse
import math
import operator
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import convolve

from utils import *
from detectCircles import detectCircles


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Quantize images')
    parser.add_argument('--filename', type=str, default='jupiter.jpg', help='Filename of image to load')
    parser.add_argument('--rs', nargs='*', type=int, action='store', default=[13, 32, 50, 110],
                        help='List of multiple radii to find')
    parser.add_argument('--gradient', action='store_true', help='Use gradient calculated from image')
    parser.add_argument('--theta_bin', type=float, default=0.05, help='Bin size of theta values to search')
    parser.add_argument('--center_bin', type=int, default=1, help='Bin size of circle centers')
    parser.add_argument('--min_val', type=int, default=300, help='Min value of Canny Edge detector (cv2)')
    parser.add_argument('--max_val', type=int, default=500, help='Max value of Canny Edge detector (cv2)')
    parser.add_argument('--top_c', type=int, default=3, help='Keep all vote getters within n of top vote getter')
    parser.add_argument('--top_a', type=int, default=10, help='Keep all vote getters within n of top vote getter')
    parser.add_argument('--plot_acc', action='store_true', help='Plot the accumulator array')

    args = parser.parse_args()

    centers = []
    im = load_image(args.filename)
    output = im.copy()

    for r in args.rs:

        H, tracker = detectCircles(im, r, args.gradient, args.theta_bin,
                                   args.center_bin, args.min_val, args.max_val)
        sorted_H = sorted(H.items(), key=operator.itemgetter(1))
        max_votes = sorted_H[-1][1]
        centers += [center for center, count in sorted_H if count>=max_votes-args.top_c]
        #a, b, r = sorted_H[-1][0]
        #a, b, r = (108, 456, 50)
        #centers.append((a, b, r))

    for a, b, r, in centers:
        cv2.circle(output, (a, b), r, (0, 255, 0), 4)
    cv2.imshow("output", np.hstack([output, im]))
    #cv2.imwrite('r={}_{}'.format(args.rs, args.filename), output)
    cv2.imwrite('r={}_grad={}_{}'.format(args.rs, args.gradient, args.filename), output)
    #cv2.imwrite('r={}_grad={}_{}'.format(args.rs, args.gradient, args.filename), np.hstack([output, im]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if args.plot_acc:
        acc_output = np.zeros(im.shape)
        top_H = [center for center, count in sorted_H if count>=max_votes-args.top_a]

        for a, b, r, in top_H:
            cv2.circle(acc_output, (a, b), r, (255, 255, 255), 1)
        cv2.imshow("accumulator output", acc_output)
        cv2.imwrite('r={}_grad={}_acc_{}'.format(args.rs, args.gradient, args.filename), acc_output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
