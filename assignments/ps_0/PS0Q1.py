import numpy as np
import matplotlib.pyplot as plt

# Load A
A = np.load('inputAPS0Q1.npy')

# A
plt.subplot(1, 1, 1, yticks=[])
plt.imshow([sorted(A.flatten(), reverse=True) for i in range(500)], cmap='gray')
plt.savefig('4a_plot.png')
plt.show()

# B
plt.subplot(1, 1, 1, label='hist')
plt.hist(A.flatten(), bins=20)
plt.savefig('4b_hist.png')
plt.show()

# C
X = A[50:, 0:50]
plt.subplot(1, 1, 1, label='X')
plt.imshow(X, cmap='gray')
np.save('outputXPS0Q1.npy', X)
plt.savefig('4c_img.png')
plt.show()

# D
Y = np.clip(A - np.mean(A), 0, 1)
plt.subplot(1, 1, 1, label='Y')
plt.imshow(Y, cmap='gray')
np.save('outputYPS0Q1.npy', Y)
plt.savefig('4d_img.png')
plt.show()

# E
Z = np.zeros((100, 100, 3))
r, c = np.where(A > np.mean(A))
Z[r, c, 0] = 1
plt.subplot(1, 1, 1, label='Z')
plt.imshow(Z)
np.save('outputZPS0Q1.npy', Z)
plt.savefig('4e_img.png')
plt.show()
