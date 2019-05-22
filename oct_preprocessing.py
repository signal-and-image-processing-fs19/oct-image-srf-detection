"""
Image pre-processing module for OCT image SRF detection.

Binarization, Filtering, Cropping, etc. functions for the SRF detection task.

final exercise from the lecture:
Introduction to Signal and Image Processing FS19
by:
Prof. Raphael Sznitman

See README.md for the full exercise description.
"""

__author__ = "Jan Wälchli, Mario Moser, Dominik Meise"
__copyright__ = "Copyright 2019; Jan Wälchli, Mario Moser, Dominik Meise; All rigths reserved."
__email__ = "dominik.meise@students.unibe.ch"


import numpy as np
import cv2 as cv
from skimage import color, exposure
from skimage.filters import threshold_otsu, gaussian


def otsu_binarize(img, sigma=1):
    """Return binarized image by using gaussian blurring and otsu thresholding."""
    if len(img.shape) == 3:
        img = color.rgb2gray(img)
    elif len(img.shape) != 2:
        raise ValueError('Cannot handle unknown Image dimension!')

    img_blur = gaussian(img, sigma=sigma)
    otsu_thresh = threshold_otsu(img_blur)
    img_bin = img_blur > otsu_thresh

    return img_bin


def crop(img, border=50):
    """Return cropped image using a binarized version of that image as a mask to define the relevant region."""
    # crop the white border
    border = border
    img_no_border = img[border:img.shape[0]-border, border:img.shape[1]-border]

    # make a binary picture
    img_bin = otsu_binarize(img_no_border, 10)

    # search the 4 outermost white pixels, crop the image there
    white_pixels = np.where(img_bin == 1)
    up, bottom = min(white_pixels[0]), max(white_pixels[0])
    left, right = min(white_pixels[1]), max(white_pixels[1])
    img_crop = img_no_border[up:bottom, left:right]

    return img_crop


def hist_equalize(img):
    """Return a histogram equalized version of the image (enhances 'contrast')."""
    if len(img.shape) == 3:
        img = color.rgb2gray(img) * 255
        img = img.astype(np.uint8)

    elif len(img.shape) != 2:
        raise ValueError('Cannot handle unknown Image dimension!')

    if np.amax(img) <= 1 and img.dtype != np.uint8:
        img = img * 255
        img = img.astype(np.uint8)

    eq = exposure.equalize_hist(img) * 255
    eq = eq.astype(np.uint8)

    return eq


def opening_denoising(img, kernel_size=5):
    """Denoise image by opening (erosion then dilation)."""

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
    img_denoise = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)

    return img_denoise
