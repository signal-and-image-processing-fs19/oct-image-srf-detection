"""
OCT image SRF detection module.

Classification of OCT images for the retinal disease bio-marker SRF (sub-retinal fluid).
This is the file to run for completing the task and output the classification of the
given images and some summary statistics.

final exercise from the lecture:
Introduction to Signal and Image Processing FS19
by:
Prof. Raphael Sznitman

See README.md for the full exercise description.
"""

__author__ = "Jan Wälchli, Mario Moser, Dominik Meise"
__copyright__ = "Copyright 2019; Jan Wälchli, Mario Moser, Dominik Meise; All rigths reserved."
__email__ = "dominik.meise@students.unibe.ch"


import glob
import matplotlib
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import oct_preprocessing as preproc
import oct_template_matching as tmpmatch

matplotlib.rcParams['image.cmap'] = 'gray'


def main():
    images_srf = glob.glob('Train-Data/SRF/*')
    images_no = glob.glob('Train-Data/NoSRF/*')
    template = io.imread('dummy_template.png')
    method = 'cv.TM_SQDIFF'

    class_srf = []
    for i in images_srf:
        # preprocessing
        img_orig = io.imread(i)
        img_crop = preproc.crop(img_orig)
        img_eq = preproc.hist_equalize(img_crop)
        img_denoise = preproc.opening_denoising(img_eq)

        # matching
        res, img = tmpmatch.template_matching(img_denoise, template)

        # store minimum value found (best match)
        min_val = np.amin(res)
        class_srf.append(min_val)

    class_no = []
    for i in images_no:
        # preprocessing
        img_orig = io.imread(i)
        img_crop = preproc.crop(img_orig)
        img_eq = preproc.hist_equalize(img_crop)
        img_denoise = preproc.opening_denoising(img_eq)

        # matching
        res, img = tmpmatch.template_matching(img_denoise, template, method)

        # store minimum value found (best match)
        min_val = np.amin(res)
        class_no.append(min_val)

    # testing range of thresholds
    precisions = []
    low = 2000000
    upp = 6000000
    stp = 50000
    for thresh in range(low, upp, stp):
        tp = 0
        count = 0

        for i in class_srf:
            count += 1
            if i <= thresh:
                tp += 1

        for i in class_no:
            count += 1
            if i > thresh:
                tp += 1

        precision = tp / count
        precisions.append(precision)
        print(thresh, ':\t', precision, '% ', tp, '/', count)

    plt.plot(range(low//1000, upp//1000, stp//1000), precisions)
    plt.xlabel('threshold (x1000)')
    plt.ylabel('precision')
    plt.title(method)
    plt.show()


if __name__ == '__main__':
    main()
