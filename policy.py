from PIL import Image, ImageChops, ImageOps
import random
from PIL.ImageEnhance import Contrast, Sharpness, Color, Brightness
test_img = "D:/iqiyi/PYTORCH/cifar-10-python/train/0/b'aeroplane_s_000004.png'.jpg"


def invert(img, prob, mag=7):
    # mag is useless
    if random.random() < prob:
        return ImageOps.invert(img)
    else:
        return img


def contrast(img, prob=0.2, mag=1.3):
    # mag : 0: balck 1: original  3.0 almost max    [0.1, 1.9]
    if random.random() < prob:
        enhancer = Contrast(img)
        return enhancer.enhance(mag)
    else:
        return img


def autocontrast(img, prob, mag=0):
    # maximize the image contrst to black and white
    if random.random() < prob:
        enhancer = Contrast(img)
        return enhancer.enhance(4)
    else:
        return img


def rotate(img, prob=0.7, mag=2):
    # mag: [0, 10] : -30: 30
    if random.random() < prob:
        return img.rotate(mag*6 - 30)
    else:
        return img


def translatex(img, prob, xoff):
    # [15 15]
    if random.random() < prob:
        width, height = img.size
        img = ImageChops.offset(img, xoff, 0)
        if xoff > 0:
            img.paste((0,0,0),(0,0,xoff,height))
        else:
            img.paste((0, 0, 0), (width+xoff, 0, width, height))
        # c.paste((0,0,0),(0,0,width,yoff))
    return img


def translatey(img, prob, yoff):
    # [-15, 15]
    if random.random() < prob:
        width, height = img.size
        img = ImageChops.offset(img, 0, yoff)
        if yoff > 0:
            img.paste((0, 0, 0,), (0, 0, width, yoff))
        else:
            img.paste((0, 0, 0), (0, height+yoff, width, height))
    return img


def sharpness(img, prob, mag):
    # mag [0.1, 1.9]
    if random.random() < prob:
        return Sharpness(img).enhance(mag)
    else:
        return img


def shearx(img, prob, mag=0):
    # mag [-0.3, 0.3]
    if random.random() < prob:
        return img
    else:
        return img


def sheary(img, prob, mag=0):
    # mag [-0.3, 0.3]
    if random.random() < prob:
        return img
    else:
        return img


def equalize(img, prob, mag=0):
    if random.random() < prob:
        return ImageOps.equalize(img)
    else:
        return img


def posterize(img, prob, mag=1):
    # mag [4, 8]
    if random.random() < prob:
        return ImageOps.posterize(img, mag)
    else:
        return img


def color(img, prob, mag=1):
    # mag [0, 1, >>>]
    if random.random() < prob:
        return Color(img).enhance(mag)
    else:
        return img


def brightness(img, prob, mag=1):
    # mag [0, 1, >>>]
    if random.random() < prob:
        return Brightness(img).enhance(mag)
    else:
        return img


def solarize(img, prob, mag=128):
    # mag [0, 256]
    if random.random() < prob:
        return ImageOps.solarize(img, threshold=mag)
    else:
        return img


def cutout(img, prob, mag=0):
    # mag = [0, 5]
    if random.random() < prob:
        wid, hig = img.size
        x = random.randint(mag, wid)
        y = random.randint(mag, hig)
        img.paste((0, 0, 0), (x-mag, y-mag, x, y))
    return img


def sample_pair(img, img2, prob, mag=0):
    if random.random() < prob:

        return Image.blend(img, img2, mag)
    else:
        return img

img = Image.open(test_img)
img.show()
img = cutout(img, 1, 3)
img.show()

