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


import csv
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import metrics
import oct_template_matching as tmpmatch

matplotlib.rcParams['image.cmap'] = 'gray'


def evaluate_threshold(template_path, preproc_methods, matching_method, denoise_strength):
    """Determine the optimal threshold based on the given preprocessing and matching methods and the template used.

    :param template_path: Filepath of the template used for matching
    :param preproc_methods: list of strings of the used preprocessing methods
    :param matching_method: method used for template matching
    :param denoise_strength: integer value which set the degree of denoising applied during preprocessing
    :return: prec: highest precision achieved for the defined range of thresholds for the available train-data
             auc: area under the curve value for the defined range of thresholds
             thresh: the threshold value which achieved the highest precision for the available train-data
    """
    # load available train-data
    images_srf = glob.glob('Train-Data/SRF/*')
    images_no = glob.glob('Train-Data/NoSRF/*')

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
    prec, auc, thresh = eval_precision(low, upp, stp, best_scores_srf, best_scores_no,
                                       preproc_methods, matching_method, setting_string, stdout=False)

    return prec, auc, thresh


def classify_by_threshold(threshold, scores, matching_method):
    """Classifies each score based on the given threshold.

    :param threshold: given (ideally optimized) threshold to discriminate the two classes
    :param scores: dissimilatiry or distance scores to classify
    :param matching_method: method used for template matching (determines which class lies on which side of the thresh)
    :return: list of same length as input with score values replaced by 0 or 1 depending on the classification
    """
    img_classes = np.asarray(scores)

    if 'SQDIFF' in matching_method:
        img_classes[img_classes <= threshold] = 1
        img_classes[img_classes > threshold] = 0
    else:
        img_classes[img_classes >= threshold] = 1
        img_classes[img_classes < threshold] = 0

    return img_classes.astype(int)


def write_csv(image_names, img_classes, filename):
    """Produce csv output file according to the project specifications from a list of image names and classification."""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'label'])
        writer.writerows(zip(image_names, img_classes))


def eval_precision(low, upp, stp, min_dist_srf, min_dist_no, preproc_methods,
                   matching_method, setting_string='default', stdout=True):
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
    thresh = np.arange(low, upp, stp)[np.argmax(precisions)]

    # auc calculation
    prec = sorted(precisions)
    coord = np.arange(len(prec))*0.001
    auc = metrics.auc(prec, coord)

    if stdout:
        print('Best precision:' + str(best_prec))
        print('at threshold: ' + str(thresh))
        print('auc: ', auc)

    # plotting
    plt.plot(np.arange(low, upp, stp), precisions)
    plt.xlabel('threshold ')
    plt.ylabel('precision')
    plt.title(', '.join(preproc_methods) + ', ' + matching_method +
              '\nbest prec = {}, at threshold = {}, AUC = {}'.format(round(best_prec, 3),
                                                                     round(thresh, 3), round(auc, 3)))

    # show results or save as txt
    if stdout:
        plt.show()
    else:
        plt.savefig('figures/' + setting_string + '.png')
        plt.close()

    return best_prec, auc, thresh


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
