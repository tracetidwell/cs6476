import numpy as np

from utils import *


def improvedHuMoments(mhi, scaled=True):

	cropped_mhi = mhi.copy()

	i_idxs, j_idxs = np.where(cropped_mhi != 0)
	i_min, i_max = i_idxs[np.argmin(i_idxs)], i_idxs[np.argmax(i_idxs)]
	j_min, j_max = j_idxs[np.argmin(j_idxs)], j_idxs[np.argmax(j_idxs)]
	cropped_mhi = cropped_mhi[i_min:i_max+1, j_min:j_max+1]

	im_moment_00 = image_moment(cropped_mhi, 0, 0, scaled)
	im_moment_10 = image_moment(cropped_mhi, 1, 0, scaled)
	im_moment_01 = image_moment(cropped_mhi, 0, 1, scaled)

	x_bar = im_moment_10 / im_moment_00
	y_bar = im_moment_01 / im_moment_00

	moments = np.zeros(7)

	for i in range(7):
	    moment = eval(f'h{i+1}')
	    moments[i] = moment(cropped_mhi, x_bar, y_bar, scaled)

	return moments