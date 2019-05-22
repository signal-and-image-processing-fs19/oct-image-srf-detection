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


def template_matching(image, template, meth='cv.TM_SQDIFF'):
    """Match the template against the image, return resolution and location map.

    :param image: input image in which to search for the template matching (either rgb or grayscale)
    :param template: kernel/template which to match against the input image
    :param meth: template matching method. one of the following:
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
