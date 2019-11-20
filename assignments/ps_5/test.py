import glob
import pdb
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from scipy.spatial import distance

from computeMHI import computeMHI
from generateAllMHIs import generateAllMHIs
from huMoments import huMoments
from generateAllHuMoments import generateAllHuMoments
from utils import *


mhis = np.load('allMHIs.npy')
hu_moments = np.load('allHuMoments.npy')
norm_hu_moments = hu_moments / np.max(hu_moments, axis=0)
labels = np.array(sorted([i for i in range(1, 6)] * 4)).reshape(-1, 1)
data = np.concatenate([norm_hu_moments, labels], axis=1)
