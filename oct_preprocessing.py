import glob
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageOps
from skimage import io, color
from skimage.feature import canny
from skimage.filters import try_all_threshold, threshold_otsu, gaussian
matplotlib.rcParams['image.cmap'] = 'gray'


def main():
    ###     VARIOUS TESTING
    #simple_canny_example('Train-Data/SRF/input_1492_1.png')
    #simple_binarization_example('Train-Data/SRF/input_1492_1.png')
    images = glob.glob('Train-Data/SRF/*')
    images = images + glob.glob('Train-Data/NoSRF/*')
    for i in images[:3]:
        orig = io.imread(i)
        orig_bin = otsu_binar(orig)
        orig_bin_crop = crop(orig_bin)
        plot_original_and_processed(orig, orig_bin_crop, 'otsu')


def crop(img_bin):

    #crop the white border
    border = (50,50,50,50) #left, up, right, bottom
    img_no_border = ImageOps.crop(img_bin, border)

    #search the 4 outest white pixels, crop the image there
    white_pixels = np.where(img_no_border == 1)
    up, bottom = min(white_pixels[0]), max(white_pixels[0])
    left, right = min(white_pixels[1]), max(white_pixels[1])
    border =(left, up, img_no_border.shape[1]-right, img_no_border.shape[0]-bottom)
    img_crop = ImageOps.crop(img_no_border, border)

    return img_crop


def otsu_binar(orig):
    orig = color.rgb2gray(orig)
    orig_blur = gaussian(orig, sigma=2)
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
