"""
Evaluation module for OCT image SRF detection.

Producing results and final output, as well as results and summaries for method testing and hyperparameter tuning.

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
import matplotlib.pyplot as plt
from sklearn import metrics


def eval_precision(low, upp, stp, min_dist_srf, min_dist_no, preproc_methods,
                   matching_method, setting_string='', stdout=True):
    """Evaluate precision of the system for a range of thresholds.

    :param low: lower threshold boundary
    :param upp: upper threshold boundary
    :param stp: threshold step size
    :param min_dist_srf: list of minimum distances of each SRF image
    :param min_dist_no: list of minimum distances of each non-SRF image
    :param preproc_methods: list of preprocessing method names (strings)
    :param matching_method: matching method name (string)
    :param setting_string: string-label of the given setting
    :param stdout: indicating if results should be written to stdout or saved as txt-file
    """

    precisions = []
    # iterating over all thresholds
    for thresh in np.arange(low, upp, stp):
        tp = 0
        count = 0

        # count how many srf images are correctly identified (above the threshold)
        for i in min_dist_srf:
            count += 1
            if i >= thresh:
                tp += 1

        # count how many non-srf are correctly identified (below the threshold)
        for i in min_dist_no:
            count += 1
            if i < thresh:
                tp += 1

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED,
        # smaller scores are better, else larger score are better matching.
        # Thus inverse true positives.
        if matching_method in ['cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']:
            tp = count - tp

        precision = tp / count
        precisions.append(precision)
        # print(thresh, ':\t', precision, '% ', tp, '/', count)

    best_prec = max(precisions)

    if stdout:
        print('Best precision:' + str(best_prec))

    # auc calculation
    prec = sorted(precisions)
    coord = np.arange(len(prec))*0.001
    auc = metrics.auc(prec, coord)

    if stdout:
        print('auc: ', auc)

    # plotting
    plt.plot(np.arange(low, upp, stp), precisions)
    plt.xlabel('threshold ')
    plt.ylabel('precision')
    plt.title(', '.join(preproc_methods) + ', ' + matching_method +
              '\nbest prec = {}, AUC = {}'.format(round(best_prec, 3), round(auc, 3)))

    # show results or save as txt
    if stdout:
        plt.show()
    else:
        plt.savefig('figures/' + setting_string + '.png')
        plt.close()

    return best_prec, auc


def sort_result_and_save_as_txt(result):
    """Sorting dict by value and then saving as a txt-file."""
    with open('results.txt', 'w') as f:
        f.write('setting:\t(prec, auc)\n')
        for key, value in sorted(result.items(), key=lambda item: item[1], reverse=True):
            f.write('{}:\t{}\n'.format(key, value))


def plot_original_and_processed(original, processed, process_title=''):
    """Plotting two images side by side for comparison."""
    fig, axs = plt.subplots(1, 2)

    axs[0].imshow(original)
    axs[0].set_title('original image')
    axs[1].imshow(processed)
    axs[1].set_title(process_title)

    plt.show()
