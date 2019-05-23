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
    images_NOsrf = glob.glob('Train-Data/NoSRF/*')
    template_path = 'dummy_template.png'
    preproc_methods = ['crop', 'eq', 'opening']
    matching_method = 'cv.TM_SQDIFF'

    min_dist_srf = tmpmatch.run_matching(images_srf, template_path, preproc_methods, matching_method)

    min_dist_NOsrf = tmpmatch.run_matching(images_NOsrf, template_path, preproc_methods, matching_method)

    # testing range of thresholds
    precisions = []
    low = 2000000
    upp = 6000000
    stp = 50000
    for thresh in range(low, upp, stp):
        tp = 0
        count = 0

        for i in min_dist_srf:
            count += 1
            if i <= thresh:
                tp += 1

        for i in min_dist_NOsrf:
            count += 1
            if i > thresh:
                tp += 1

        precision = tp / count
        precisions.append(precision)
        print(thresh, ':\t', precision, '% ', tp, '/', count)

    plt.plot(range(low//1000, upp//1000, stp//1000), precisions)
    plt.xlabel('threshold (x1000)')
    plt.ylabel('precision')
    plt.title(', '.join(preproc_methods) + ', ' + matching_method)
    plt.show()


if __name__ == '__main__':
    main()
