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
from skimage import io, color, exposure
from skimage.filters import threshold_otsu, gaussian


def load_img_as_gray(img_path):
    """Read in rgb image and convert it to gray-scale 0-255 uint8 np array."""
    img = io.imread(img_path)
    return (color.rgb2gray(img) * 255).astype(np.uint8)


def load_preproc_template(template_path, preproc_methods, denoise_strength):
    """Create template from fixed region of the first train-data srf image, but with the given preprocessing applied."""
    tmpl = load_img_as_gray('Train-Data/SRF/input_1492_1.png')
    tmpl = perform_bulk_perproc(tmpl, preproc_methods, denoise_strength)
    return tmpl[100:140, 300:340]


def perform_bulk_perproc(image, preprocessing_methods, denoise_strength):
    """Perform all specified preprocessing steps as a bulk task.

    :param image: original unprocessed image
    :param preprocessing_methods: list of preprocessing method names (strings). Available:
        'crop', 'eq', 'opening', 'nonloc'
    :param denoise_strength: integer to specify how aggresively denoising should be applied.
    :return: processed image
    """
    img = image.copy()
    if 'crop' in preprocessing_methods:
        img = crop(img)
    if 'eq' in preprocessing_methods:
        img = hist_equalize(img)
    if 'opening' in preprocessing_methods:
        img = opening_denoising(img, kernel_size=denoise_strength)
    if 'nonloc' in preprocessing_methods:
        img = nonloc_denoising(img, denoise_strength)
    # TODO: add all other preprocessing options and decide on ordering

    return img


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


def nonloc_denoising(img, denoise_strength):
    return cv.fastNlMeansDenoising(img, h=denoise_strength)  # TODO: find best h (strength of denoising)
