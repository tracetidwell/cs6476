import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Convert color image to grayscale
def rgb2gray(rgb):
    gray = rgb.copy()
    return np.dot(gray, [0.2989, 0.5870, 0.1140]).astype('uint8')

# Load color image
img = mpimg.imread('inputPS0Q2.jpg')

# 1: Swap red and green channels
img2 = img.copy()
img2[:, :, 0], img2[:, :, 1] = img2[:, :, 1], img2[:, :, 0]
mpimg.imsave('swapImgPS0Q2.png', img2, cmap='gray')

# 2: Convert color image to grayscale
img3 = rgb2gray(img2)
mpimg.imsave('grayImgPS0Q2.png', img3, cmap='gray')

# 3a: Convert grayscale image to negative
img4 = 255 - img3.copy()
mpimg.imsave('negativeImgPS0Q2.png', img4, cmap='gray')

# 3b: Map grayscale image to mirror image
img5 = img3.copy()[:, ::-1]
mpimg.imsave('mirroImgPS0Q2.png', img5, cmap='gray')

# 3c: Average grayscale image with its mirror image
img6 = ((img3.copy() + img5.copy()) / 2).astype('uint8')
mpimg.imsave('avgImgPS0Q2.png', img6, cmap='gray')

# 3d: Add noise to grayscale image and clip at 255
noise = np.random.randint(0, 255, img3.shape)
np.save('noise.npy', noise)
img7 = np.clip(img3.copy() + noise, 0, 255).astype('uint8')
mpimg.imsave('addNoiseImgPS0Q2.png', img7, cmap='gray')

# Create figure with subplots for 6 images above
plt.figure(figsize=(15,20))

plt.subplot(3, 2, 1, title='RG Channel Swap', xticks=[], yticks=[])
plt.imshow(img2, cmap='gray')

plt.subplot(3, 2, 2, title='Grayscale', xticks=[], yticks=[])
plt.imshow(img3, cmap='gray')

plt.subplot(3, 2, 3, title='Negative', xticks=[], yticks=[])
plt.imshow(img4, cmap='gray')

plt.subplot(3, 2, 4, title='Mirror', xticks=[], yticks=[])
plt.imshow(img5, cmap='gray')

plt.subplot(3, 2, 5, title='Average', xticks=[], yticks=[])
plt.imshow(img6, cmap='gray')

plt.subplot(3, 2, 6, title='Noise', xticks=[], yticks=[])
plt.imshow(img7, cmap='gray')

plt.savefig('all_subplots.png')

plt.show()
