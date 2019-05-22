import os
import glob
import matplotlib
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from skimage import io, color, exposure
from skimage.feature import canny
from skimage.filters import try_all_threshold, threshold_otsu, gaussian
matplotlib.rcParams['image.cmap'] = 'gray'


def main():
    #     VARIOUS TESTING
    # simple_canny_example('Train-Data/SRF/input_1492_1.png')
    #simple_binarization_example('Train-Data/SRF/input_1492_1.png')
    images_SRF = glob.glob('Train-Data/SRF/*')
    images_NO = glob.glob('Train-Data/NoSRF/*')

    class_SRF = []
    for i in images_SRF:
        orig = io.imread(i)
        orig_crop = crop(orig)

        #plot_original_and_processed(orig, orig_crop, 'cropped')

        eq = testing_hist_equalize(orig_crop)
        #io.imsave('equalized_histograms/'+i+'hist.png', eq)
        #plot_original_and_processed(orig_crop, eq, 'hist_equalized')

        #kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
        #opened = cv.morphologyEx(orig_crop, cv.MORPH_OPEN, kernel)
        #plot_original_and_processed(orig_crop, opened, 'opened')

        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7,7))
        opened_eq = cv.morphologyEx(eq, cv.MORPH_OPEN, kernel)
        #plot_original_and_processed(eq, opened_eq, 'opened_eq')

        res, img, meth = template_matching(opened_eq, i)

        min_val = np.amin(res)

        class_SRF.append(min_val)


        #plt.subplot(121), plt.imshow(res, cmap='gray')
        #plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        #plt.subplot(122), plt.imshow(img, cmap='gray')
        #plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        #plt.suptitle('{} , {} , {}'.format(meth, i, str(min_val)))
        #plt.show()


    class_NO = []
    for i in images_NO:
        orig = io.imread(i)
        orig_crop = crop(orig)

        # plot_original_and_processed(orig, orig_crop, 'cropped')

        eq = testing_hist_equalize(orig_crop)
        # io.imsave('equalized_histograms/'+i+'hist.png', eq)
        # plot_original_and_processed(orig_crop, eq, 'hist_equalized')

        # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
        # opened = cv.morphologyEx(orig_crop, cv.MORPH_OPEN, kernel)
        # plot_original_and_processed(orig_crop, opened, 'opened')

        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
        opened_eq = cv.morphologyEx(eq, cv.MORPH_OPEN, kernel)
        # plot_original_and_processed(eq, opened_eq, 'opened_eq')

        res, img, meth = template_matching(opened_eq, i)

        min_val = np.amin(res)

        class_NO.append(min_val)

    for thresh in range(3000000, 5000000, 100000):
        tp = 0
        count = 0

        for i in class_SRF:
            count += 1
            if i <= thresh:
                tp += 1

        for i in class_NO:
            count += 1
            if i > thresh:
                tp += 1

        precision = tp / count
        print(thresh, ':\t', precision, '% ', tp, '/', count)

def template_matching(img, path):
    img2 = img.copy()
    template = io.imread('dummy_template.png')
    w, h = template.shape[::-1]

    # All the 6 methods for comparison in a list
    methods = ['cv.TM_SQDIFF']  # 'cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR', 'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED'

    for meth in methods:
        img = img2.copy()
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

        return res, img, meth




def testing_hist_equalize(img, plotting=False):
    orig_crop = color.rgb2gray(img) * 255
    orig_crop = orig_crop.astype(np.uint8)

    eq = exposure.equalize_hist(orig_crop) * 255
    eq = eq.astype(np.uint8)

    if plotting:
        plt.subplot(221); plt.hist(orig_crop.flatten(), 256, range=(0, 256))
        plt.subplot(222); plt.imshow(orig_crop)

        plt.subplot(223); plt.hist(eq.flatten(), 256, range=(0, 256))
        plt.subplot(224); plt.imshow(eq)
        plt.show()

    return eq

def crop(img, border=50):

    # crop the white border
    border = border
    img_no_border = img[border:img.shape[0]-border, border:img.shape[1]-border]

    # make a binary picture
    img_bin = otsu_binar(img_no_border, 10)

    # search the 4 outest white pixels, crop the image there
    white_pixels = np.where(img_bin == 1)
    up, bottom = min(white_pixels[0]), max(white_pixels[0])
    left, right = min(white_pixels[1]), max(white_pixels[1])
    img_crop = img_no_border[up:bottom, left:right]

    return img_crop


def otsu_binar(orig, sigma=1):
    orig = color.rgb2gray(orig)
    orig_blur = gaussian(orig, sigma=sigma)
    otsu_thresh = threshold_otsu(orig_blur)
    orig_bin = orig_blur > otsu_thresh

    return orig_bin


def simple_binarization_example(filepath):
    orig = io.imread(filepath)

    orig = color.rgb2gray(orig)

    fig, ax = try_all_threshold(orig, figsize=(10, 8), verbose=False)
    plt.show()


def simple_canny_example(filepath):
    orig = io.imread(filepath)

    sigma = 7
    orig_canny = canny(color.rgb2gray(orig), sigma)

    plot_original_and_processed(orig, orig_canny, 'canny edge with sigma = {}'.format(sigma))


def plot_original_and_processed(original, processed, process_title):

    fig, axs = plt.subplots(1, 2)

    axs[0].imshow(original)
    axs[0].set_title('original image')
    axs[1].imshow(processed)
    axs[1].set_title(process_title)

    fig.suptitle('example image for SRF')

    plt.show()


if __name__ == '__main__':
    main()
