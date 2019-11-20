import glob
import pdb
import os
import numpy
import matplotlib.pyplot as plt
from scipy.misc import imread



basedir = 'PS5_Data'
actions = ['botharms', 'crouch', 'leftarmup', 'punch', 'rightkick']

for action in actions:

	subdir_name = os.path.join(basedir, action)
	subdir_files = os.listdir(subdir_name)

	for subdir_file in subdir_files:
	# cycle through all sequences for this action category
	depth_files = glob.glob(os.path.join(subdir_name, subdir_file, '*.pgm'))
	depth_files = numpy.sort(depth_files)

	for depth_file in depth_files:
		depth_im = imread(depth_file)
		plt.imshow(depth_im)
		plt.show()
