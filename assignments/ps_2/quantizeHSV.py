from quantizeRGB import quantizeRGB

def quantizeHSV(im, k):

    w, h, c = im.shape
    hsv_im = rgb2hsv(im)
    segmented_hue, meanHues = quantizeRGB(hsv_im[:, :, 0].reshape(-1, 1), k)
    segmented_hsv_im = hsv_im.copy()
    segmented_hsv_im[:, :, 0] = segmented_hue.reshape(w, h)
    outputImg = hsv2rgb(segmented_hsv_im)

    return outputImg, meanHues
