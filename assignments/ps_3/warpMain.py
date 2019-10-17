import argparse
import numpy as np
import cv2
from getCorrespondences import get_correspondences
from ransac import ransac
from computeH import computeH
from verifyH import verifyH
from warpImage import warpImage
from utils import load_image, show


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Quantize images')
    parser.add_argument('--im_files', nargs='*', type=str, action='store', default=['crop1.jpg', 'crop2.jpg'],
                        help='List of images to be warped')
    parser.add_argument('--load_corrs', action='store_true', help='Load the correspondences')
    parser.add_argument('--load_corrs_files', nargs='*', type=str, action='store', default=['cc1.npy', 'cc2.npy'],
                        help='List of correspondences to be used for homography calculation')
    parser.add_argument('--save_corrs', action='store_true', help='Save the correspondences')
    parser.add_argument('--show_ims', action='store_true', help='Show the new images')
    parser.add_argument('--save_ims', action='store_true', help='Save the new images to file')
    parser.add_argument('--save_prefix', type=str, default='im', help='Prefix for saving images')

    parser.add_argument('--corr_method', type=str, default='manual', help='Prefix for saving images')
    parser.add_argument('--corr_topn', type=int, default=5, help='Column on which to blend images')
    parser.add_argument('--use_ransac', action='store_true', help='Show the new images')

    parser.add_argument('--blend_col', type=int, default=0, help='Column on which to blend images')
    parser.add_argument('--blend_row', type=int, default=0, help='Row on which to blend images')
    parser.add_argument('--blend_step', type=float, default=0.02, help='Step size to use when blending images')
    parser.add_argument('--merge_type', type=str, default='blend', help='Type of image merging to perform')




    # parser.add_argument('--rs', nargs='*', type=int, action='store', default=[13, 32, 50, 110],
    #                     help='List of multiple radii to find')
    # parser.add_argument('--gradient', action='store_true', help='Use gradient calculated from image')
    # parser.add_argument('--theta_bin', type=float, default=0.05, help='Bin size of theta values to search')
    # parser.add_argument('--center_bin', type=int, default=1, help='Bin size of circle centers')
    # parser.add_argument('--min_val', type=int, default=300, help='Min value of Canny Edge detector (cv2)')
    # parser.add_argument('--max_val', type=int, default=500, help='Max value of Canny Edge detector (cv2)')
    # parser.add_argument('--top_c', type=int, default=3, help='Keep all vote getters within n of top vote getter')
    # parser.add_argument('--top_a', type=int, default=10, help='Keep all vote getters within n of top vote getter')
    # parser.add_argument('--plot_acc', action='store_true', help='Plot the accumulator array')

    args = parser.parse_args()

    print(args)

    ims = [load_image(file) for file in args.im_files]
    n_ims = len(ims)

    for i in range(n_ims - 1):

        if not i:
            im1 = ims[0]
            im2 = ims[1]
        else:
            im1 = mergeIm
            im2 = ims[i+1]

        h1, w1, _ = im1.shape
        h2, w2, _ = im2.shape

        scale = max(h1, w1, h2, w2) / 2

        if args.load_corrs:
            if len(args.load_corrs_files) == n_ims - 1:
                t1, t2 = np.load(args.load_corrs_files[0])
            else:
                t1 = np.load(args.load_corrs_files[0]).T
                t2 = np.load(args.load_corrs_files[1]).T

        else:
            t1, t2 = get_correspondences(im1, im2, args.corr_method, args.corr_topn)

        if args.use_ransac:

            t1, t2 = ransac(t1, t2, scale)

        if args.save_corrs:
            np.save('{}_corrs_{}.npy'.format(args.save_prefix, i), [t1, t2])

        t1_scaled = t1 / scale
        t2_scaled = t2 / scale

        H = computeH(t1_scaled, t2_scaled)

        verifiedIm1, verifiedIm2 = verifyH(im1, im2, t1, t2, H)

        warpIm, mergeIm = warpImage(im1, im2, H, args.blend_col, args.blend_row,
                                    args.blend_step, args.merge_type)

        if args.show_ims:
            show(warpIm, 'warped')
            show(mergeIm, 'merged')
            show(verifiedIm1, 'verify H 1 to 2')
            show(verifiedIm2, 'verify H 2 to 1')

        if args.save_ims:
            cv2.imwrite('{}_warpIm_{}.jpg'.format(args.save_prefix, i), warpIm)
            cv2.imwrite('{}_mergeIm_{}.jpg'.format(args.save_prefix, i), mergeIm)
            cv2.imwrite('{}_verifyH_12_{}.jpg'.format(args.save_prefix, i), verifiedIm1)
            cv2.imwrite('{}_verifyH_21_{}.jpg'.format(args.save_prefix, i), verifiedIm2)

        # ims = [mergeIm] + ims[2:]

    # n_ims = len(args.im_files)
    #
    # im1 = load_image(args.im_files[0])
    # im2 = load_image(args.im_files[1])
    #
    # if args.load_corrs:
    #     if len(args.load_corrs_files) == 2:
    #         t1 = np.load(args.load_corrs_files[0]).T
    #         t2 = np.load(args.load_corrs_files[1]).T
    #     else:
    #         t1, t2 = np.load(args.load_corrs_files[0])
    # else:
    #     t1, t2 = get_correspondences(im1, im2)
    #
    #     if args.save_corrs:
    #         np.save('{}_corrs.npy'.format(args.save_prefix), [t1, t2])
    #
    # h1, w1, _ = im1.shape
    # h2, w2, _ = im2.shape
    #
    # scale = max(h1, w1, h2, w2) / 2
    #
    # t1_scaled = t1 / scale
    # t2_scaled = t2 / scale
    #
    # H = computeH(t1_scaled, t2_scaled)
    #
    # warpIm, mergeIm = warpImage(im1, im2, H, args.blend_col, args.blend_step)
    #
    # if args.show_ims:
    #     show(warpIm)
    #     show(mergeIm)
    #
    # if args.save_ims:
    #     cv2.imwrite('{}_warpIm.jpg'.format(args.save_prefix), warpIm)
    #     cv2.imwrite('{}_mergeIm.jpg'.format(args.save_prefix), mergeIm)
