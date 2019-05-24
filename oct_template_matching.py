"""
Template Matching module for OCT image SRF detection.

Template matching and fluid occlusion detection functions for the SRF detection task.

final exercise from the lecture:
Introduction to Signal and Image Processing FS19
by:
Prof. Raphael Sznitman

See README.md for the full exercise description.
"""

__author__ = "Jan Wälchli, Mario Moser, Dominik Meise"
__copyright__ = "Copyright 2019; Jan Wälchli, Mario Moser, Dominik Meise; All rigths reserved."
__email__ = "dominik.meise@students.unibe.ch"


import cv2 as cv
import numpy as np
from skimage import io, color
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import oct_preprocessing as preproc

matplotlib.rcParams['image.cmap'] = 'gray'


def run_matching(image_paths, template_path, preprocessing_methods, matching_method='cv.TM_SQDIFF'):
    """Run a matching task on a list of images (paths), with specified preprocessing and matching methods.

    :param image_paths:
    :param template_path:
    :param preprocessing_methods: list of preprocessing method names (strings). Available:
        'crop', 'eq', 'opening', 'nonloc'
    :param matching_method: template matching method. Available:
        'cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
        'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED'
    :return: list of best matching score for each image
    """
    template = io.imread(template_path)
    template = (color.rgb2gray(template)*255).astype(np.uint8)

    best_scores = []
    print('\nProcess and match images...\n')
    for i in tqdm(image_paths):
        img_orig = io.imread(i)
        img = img_orig.copy()

        # preprocessing
        if 'crop' in preprocessing_methods:
            img = preproc.crop(img)

        img = (color.rgb2gray(img)*255).astype(np.uint8)

        if 'eq' in preprocessing_methods:
            img = preproc.hist_equalize(img)
        if 'opening' in preprocessing_methods:
            img = preproc.opening_denoising(img)
        if 'nonloc' in preprocessing_methods:
            img = cv.fastNlMeansDenoising(img, h=40)  # TODO: find best h (strength of denoising)
        # TODO: add all other preprocessing options and decide on ordering

        # matching
        res, img = template_matching(img, template, matching_method)

        # store minimum value found (best match)
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if matching_method in ['cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']:
            best_scores.append(np.amin(res))
        else:
            best_scores.append(np.amax(res))

    return best_scores


def template_matching(image, template, meth='cv.TM_SQDIFF'):
    """Match the template against the image, return resolution and location map.

    :param image: input image in which to search for the template matching (either rgb or grayscale)
    :param template: kernel/template which to match against the input image
    :param meth: template matching method. Available:
        'cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
        'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED'
    :return: res: the distance map of the template matching the input image
                  (depending on the method either higher or lower is better)
             img: image with a square at the location of the best match
    """
    img = image.copy()
    w, h = template.shape[::-1]
    method = eval(meth)

    # Apply template Matching
    res = cv.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv.rectangle(img, top_left, bottom_right, 255, 2)

    return res, img


def eval_precision(low, upp, stp, min_dist_srf, min_dist_no, preproc_methods, matching_method):
    """Evaluate precision of the system for a range of thresholds.

    :param low: lower threshold boundary
    :param upp: upper threshold boundary
    :param stp: threshold step size
    :param min_dist_srf: list of minimum distances of each SRF image
    :param min_dist_no: list of minimum distances of each non-SRF image
    :param preproc_methods: list of preprocessing method names (strings)
    :param matching_method: matching method name (string)
    """
    precisions = []
    print('\nCalculating precision for thresholds {} to {}...\n'.format(low, upp))
    for thresh in range(low, upp, stp):
        tp = 0
        count = 0

        for i in min_dist_srf:
            count += 1
            if i >= thresh:
                tp += 1

        for i in min_dist_no:
            count += 1
            if i < thresh:
                tp += 1

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED,
        # smaller scores are better, else larger score are better matching.
        # Thus inverse true positives
        if matching_method in ['cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']:
            tp = count - tp

        precision = tp / count
        precisions.append(precision)
        # print(thresh, ':\t', precision, '% ', tp, '/', count)

    plt.plot(range(low//1000, upp//1000, stp//1000), precisions)
    plt.xlabel('threshold (x1000)')
    plt.ylabel('precision')
    plt.title(', '.join(preproc_methods) + ', ' + matching_method)
    plt.show()
