import cv2


def load_image(filename):
    return cv2.imread(filename)


def rgb2hsv(im):
    return cv2.cvtColor(im, cv2.COLOR_BGR2HSV)


def hsv2rgb(im):
    return cv2.cvtColor(im, cv2.COLOR_HSV2BGR)


def rgb2gray(im):
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


def show(im, text='default'):
    cv2.imshow(text, im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def round_to(n, precision):
    correction = 0.5 if n >= 0 else -0.5
    return int(n/precision+correction) * precision
