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
import oct_template_matching as tmpmatch


def main():
    images_srf = glob.glob('Train-Data/SRF/*')
    images_no = glob.glob('Train-Data/NoSRF/*')
    template_path = 'dummy_template_crop_eq_opening.png'
    preproc_methods = ['crop', 'eq', 'opening']
    matching_method = 'cv.TM_SQDIFF'
    denoise_strength = 7
    debug = False

    best_scores_srf = tmpmatch.run_matching(images_srf, template_path, preproc_methods,
                                            matching_method, denoise_strength, debug)

    best_scores_no = tmpmatch.run_matching(images_no, template_path, preproc_methods,
                                           matching_method, denoise_strength, debug)

    # testing range of thresholds
    tmpmatch.eval_precision(0, 6000000, 1000, best_scores_srf, best_scores_no, preproc_methods, matching_method)


if __name__ == '__main__':
    main()
