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
import oct_preprocessing as preproc
import oct_evaluation as evaluate


def run_matching(image_paths, template_path, preprocessing_methods, matching_method='cv.TM_SQDIFF',
                 denoise_strength=20, debug=False):
    """Run a matching task on a list of images (paths), with specified preprocessing and matching methods.

    :param image_paths:
    :param template_path:
    :param preprocessing_methods: list of preprocessing method names (strings). Available:
        'crop', 'eq', 'opening', 'nonloc'
    :param matching_method: template matching method. Available:
        'cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
        'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED'
    :param denoise_strength: integer to specify how aggresively denoising should be applied.
    :param debug: boolean to enable debugging outputs (plots or print-statememts)
    :return: list of best matching score for each image
    """
    # build template
    if template_path == '':
        template = preproc.load_preproc_template(template_path, preprocessing_methods, denoise_strength)
    else:
        template = preproc.load_img_as_gray(template_path)

    best_scores = []
    for i in image_paths:

        if debug:
            print('Processing {}...'.format(i))

        img_orig = preproc.load_img_as_gray(i)

        # preprocessing
        img = preproc.perform_bulk_perproc(img_orig, preprocessing_methods, denoise_strength)

        # checking perprocessing step
        if debug:
            evaluate.plot_original_and_processed(img_orig, img, ', '.join(preprocessing_methods))

        # create image pyramid
        img_pyr = pyramid(img)

        # checking pyramid step
        if debug:
            evaluate.plot_original_and_processed(img, img_pyr)

        # matching
        res, img = template_matching(img_pyr, template, matching_method)

        # checking matching step
        if debug:
            evaluate.plot_original_and_processed(res, img)

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


def pyramid(img):
    """Returns downscaled and smoothed image (with scikit-image)

    :param img: image as uint8, grayscaled
    :param scale: scaling factor
    """

    # dimension of new image as tuple
    scale_min = 0.1
    scale_max = 1

    # create first image with lowest scale --> biggest picture comes first
    dim = (int(img.shape[1] / scale_min), int(img.shape[0] / scale_min))
    first = cv.resize(img, dim, interpolation=cv.INTER_AREA)

    img_pyr = first.copy()

    for scale in np.arange(scale_min+0.01, scale_max, 0.1):
        dim = (int(img.shape[1] / scale), int(img.shape[0] / scale))
        img_d = cv.resize(img, dim, interpolation=cv.INTER_AREA)

        img_d_1 = np.pad(img_d, ((0, 0), (0, first.shape[1] - img_d.shape[1])), mode='constant', constant_values=(0))

        img_pyr = np.concatenate((img_pyr, img_d_1))

    # cv.imshow("", img_pyr)
    # cv.waitKey(0)
    return img_pyr





