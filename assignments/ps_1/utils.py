import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def plot_energy(im, ei, vcmem, hcmem, title, cmap, save_ims, save_all):

    if save_ims:

        mpimg.imsave('energy_image_{}.png'.format(title), ei, cmap=cmap)
        mpimg.imsave('vert_cum_min_energy_map_{}.png'.format(title), vcmem, cmap=cmap)
        mpimg.imsave('horiz_cum_min_energy_map_{}.png'.format(title), hcmem, cmap=cmap)

    if save_all:

        fig = plt.figure(figsize=(16,12))

        ax1 = fig.add_subplot(221, xticks=[], yticks=[])
        plt.subplot(ax1)
        ax1.imshow(im)

        ax2 = fig.add_subplot(222, xticks=[], yticks=[])
        plt.subplot(ax2)
        ax2.imshow(ei, cmap=cmap)

        ax3 = fig.add_subplot(223, xticks=[], yticks=[])
        plt.subplot(ax3)
        ax3.imshow(vcmem, cmap=cmap)

        ax4 = fig.add_subplot(224, xticks=[], yticks=[])
        plt.subplot(ax4)
        ax4.imshow(hcmem, cmap=cmap)

        plt.savefig('energy_maps_{}.png'.format(title))


def plot_seams(im, vs, hs, title, save_ims, save_all):

    if save_ims:

        fig = plt.figure()
        ax1 = fig.add_subplot(111, xticks=[], yticks=[])
        plt.subplot(ax1)
        ax1.imshow(im)
        plt.savefig('{}.png'.format(title), bbox_inches='tight')

        fig = plt.figure()
        ax2 = fig.add_subplot(111, xticks=[], yticks=[])
        plt.subplot(ax2)
        ax2.imshow(im, extent=[0, im.shape[1], 0, im.shape[0]])
        ax2.plot(vs[::-1], range(len(vs)), '-', linewidth=1, color='firebrick')
        plt.savefig('vertical_seam_{}.png'.format(title), bbox_inches='tight')

        fig = plt.figure()
        ax3 = fig.add_subplot(111, xticks=[], yticks=[])
        plt.subplot(ax3)
        ax3.imshow(im, extent=[0, im.shape[1], 0, im.shape[0]])
        ax3.plot(range(len(hs)), im.shape[0]-hs, '-', linewidth=1, color='firebrick')
        plt.savefig('horizontal_seam_{}.png'.format(title), bbox_inches='tight')

        fig = plt.figure()
        ax4 = fig.add_subplot(111, xticks=[], yticks=[])
        plt.subplot(ax4)
        ax4.imshow(im, extent=[0, im.shape[1], 0, im.shape[0]])
        ax4.plot(vs[::-1], range(len(vs)), '-', linewidth=1, color='firebrick')
        ax4.plot(range(len(hs)), im.shape[0]-hs, '-', linewidth=1, color='firebrick')
        plt.savefig('both_seams_{}.png'.format(title), bbox_inches='tight')

    if save_all:

        fig = plt.figure(figsize=(16, 12))

        ax1 = fig.add_subplot(221, title='Original Image', xticks=[], yticks=[])
        plt.subplot(ax1)
        ax1.imshow(im, extent=[0, im.shape[1], 0, im.shape[0]])

        ax2 = fig.add_subplot(222, title='Vertical Seam', xticks=[], yticks=[])
        plt.subplot(ax2)
        ax2.imshow(im, extent=[0, im.shape[1], 0, im.shape[0]])
        ax2.plot(vs[::-1], range(len(vs)), '-', linewidth=1, color='firebrick')

        ax3 = fig.add_subplot(223, title='Horizontal Seam', xticks=[], yticks=[])
        plt.subplot(ax3)
        ax3.imshow(im, extent=[0, im.shape[1], 0, im.shape[0]])
        ax3.plot(range(len(hs)), im.shape[0]-hs, '-', linewidth=1, color='firebrick')

        ax4 = fig.add_subplot(224, title='Both Seams', xticks=[], yticks=[])
        plt.subplot(ax4)
        ax4.imshow(im, extent=[0, im.shape[1], 0, im.shape[0]])
        ax4.plot(vs[::-1], range(len(vs)), '-', linewidth=1, color='firebrick')
        ax4.plot(range(len(hs)), im.shape[0]-hs, '-', linewidth=1, color='firebrick')

        plt.savefig('seams_{}.png'.format(title))
