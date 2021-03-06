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


import os
import sys
import glob
import itertools
from tqdm import tqdm
import oct_template_matching as tmpmatch
import oct_evaluation as evaluate


# ======================================================================================================================
# Final state of this program. Execute main() to solve the task.
def main():
    # the following settings are optimized according to the results from our training
    image_paths = glob.glob('Test-Data/handout/*')
    image_names = [os.path.basename(image_path) for image_path in image_paths]
    template_path = ''  # leave empty to select the template used to optimize the system
    preproc_methods = ['crop', 'eq', 'nonloc']
    matching_method = 'cv.TM_CCOEFF_NORMED'
    denoise_strength = 23
    result_filename = 'project_Waelchli_Moser_Meise.csv'
    debug = False

    # run template matching against all input images
    print('Starting srf-detection of {} oct-images...'.format(len(image_paths)))
    print('Set system parameters:')
    print('\tPreprocessing methods: {}\n\tDenoise strength: {}\n\tMatching method: {}\n'.format(preproc_methods,
                                                                                                denoise_strength,
                                                                                                matching_method))
    best_scores = tmpmatch.run_matching(image_paths, template_path, preproc_methods,
                                        matching_method, denoise_strength, debug)

    # calculate best threshold for the given method parameters
    print('\n\nCalculate best threshold based on the training data...')
    prec, auc, thresh = evaluate.evaluate_threshold(template_path, preproc_methods, matching_method, denoise_strength)
    print('\n\nBest precision: {}'.format(round(prec, 3)))
    print('at threshold: {}'.format(round(thresh, 3)))
    print('AUC: {}'.format(round(auc, 3)))
    print('Saving threshold-plot in \'figures/\'...')

    # classify the images based on their score and the calculated threshold
    print('\nClassify results based on threshold {}...'.format(round(thresh, 3)))
    img_classes = evaluate.classify_by_threshold(thresh, best_scores, matching_method)

    # create output file as specified in the Test-Data/submission_guidelines.txt
    print('Saving results in {}...'.format(result_filename))
    evaluate.write_csv(image_names, img_classes, result_filename)
# ======================================================================================================================


# For single run testing purposes
def run_one_train_setting():
    images_srf = glob.glob('Train-Data/SRF/*')
    images_no = glob.glob('Train-Data/NoSRF/*')
    template_path = ''
    preproc_methods = ['crop', 'eq', 'nonloc']
    matching_method = 'cv.TM_CCOEFF_NORMED'
    denoise_strength = 23
    debug = False

    best_scores_srf = tmpmatch.run_matching(images_srf, template_path, preproc_methods,
                                            matching_method, denoise_strength, debug)

    best_scores_no = tmpmatch.run_matching(images_no, template_path, preproc_methods,
                                           matching_method, denoise_strength, debug)

    # testing range of thresholds
    evaluate.eval_precision(0, 1, 0.0001, best_scores_srf, best_scores_no, preproc_methods, matching_method)


# For batch testing and parameter optimization
def run_all_combinations():
    images_srf = glob.glob('Train-Data/SRF/*')
    images_no = glob.glob('Train-Data/NoSRF/*')
    template_path = ''

    # constructing all preprocessing settings
    all_preproc_options = ['crop', 'eq', 'opening', 'nonloc']
    preproc_sets = []
    for l in range(1, len(all_preproc_options)+1):
        for subset in itertools.combinations(all_preproc_options, l):
            if subset[0] == 'crop':  # cropping should always happen!
                preproc_sets.append(list(subset))

    # constructing all matching settings
    all_matching_options = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
                            'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

    results = {}

    # iterate over all preprocessing sets
    for preproc_methods in tqdm(preproc_sets):
        print('\nCalculating with setting: {}'.format(preproc_methods))

        # ignore iterating over denoise strength if no denoising is performed in this set
        if {'opening', 'nonloc'} & set(preproc_methods):
            # constructing all strength settings
            denoise_strengths = range(1, 42, 2)
        else:
            denoise_strengths = [0]

        # iterate denoising strength
        for denoise_strength in tqdm(denoise_strengths):
            print('\ndenoise strength: {}'.format(denoise_strength))

            # iterate over all matching methods
            for matching_method in all_matching_options:
                print('\nmatching method: {}'.format(matching_method))

                # build label for this run
                setting_string = '_'.join(preproc_methods) + '_' + str(denoise_strength) + \
                                 '_' + matching_method

                # run on all srf images
                best_scores_srf = tmpmatch.run_matching(images_srf, template_path, preproc_methods,
                                                        matching_method, denoise_strength)
                # run on all non-srf images
                best_scores_no = tmpmatch.run_matching(images_no, template_path, preproc_methods,
                                                       matching_method, denoise_strength)

                # testing range of thresholds
                if 'NORMED' in matching_method:
                    low = 0
                    upp = 1
                    stp = 0.0001
                else:
                    low = 0
                    upp = 10000000
                    stp = 1000

                # evaluate system for this set of settings
                prec, auc, thresh = evaluate.eval_precision(low, upp, stp, best_scores_srf, best_scores_no,
                                                    preproc_methods, matching_method, setting_string, stdout=False)

                # add current result to the results dictionnary
                results[setting_string] = (prec, auc)

    # save results
    evaluate.sort_result_and_save_as_txt(results)


# Quick and dirty 'stdout copy to log file' from stackoverflow:
# https://stackoverflow.com/questions/616645/how-to-duplicate-sys-stdout-to-a-log-file
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("log.dat", "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)


if __name__ == '__main__':
    sys.stdout = Logger()
    # run_one_train_setting()
    # run_all_combinations()
    main()
