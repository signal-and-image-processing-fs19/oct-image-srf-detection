"""
==================================================
Comparing edge-based and region-based segmentation
==================================================

Try to detect the white line in the images
code from https://scikit-image.org/docs/dev/auto_examples/applications/plot_coins_segmentation.html
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
from skimage import color, morphology
from skimage.feature import canny
from skimage.filters import gaussian, sobel
from oct_preprocessing import crop
from scipy import ndimage as ndi

#img = data.img()
images = glob.glob('Train-Data/SRF/*')
for i in images:
    orig = plt.imread(i)
    orig_crop = crop(orig)
    img = color.rgb2gray(orig_crop)

    #HOW MUCH BLURRING? HIGH SIGMA DECREASE THE NOISE BUT ALSO THE POWER
    img_blur = gaussian(img, sigma=0)
    img = img_blur*255

    # Edge-based segmentation

    # Canny edge-detector.
    edges = canny(img)
    # These contours are then filled using mathematical morphology.
    fill_img = ndi.binary_fill_holes(edges)
    # Small spurious objects are removed by setting a minimum size
    img_cleaned = morphology.remove_small_objects(fill_img, 21)

    #find an elevation map using the Sobel gradient of the image.
    elevation_map = sobel(img)


    # Next we find markers of the background
    markers = np.zeros_like(img)
    markers[img < 130] = 2
    markers[img > 160] = 1

    segmentation = morphology.watershed(elevation_map, markers)


    from skimage.color import label2rgb

    segmentation = ndi.binary_fill_holes(segmentation - 1)
    labeled_img, _ = ndi.label(segmentation)
    image_label_overlay = label2rgb(labeled_img, image=img)

    #plot
    plt.subplot(1,2,1)
    plt.imshow(orig_crop, cmap=plt.cm.gray, interpolation='nearest')
    plt.title('original')
    plt.subplot(1,2,2)
    plt.imshow(segmentation, cmap=plt.cm.gray, interpolation='nearest')
    plt.title('segmentation')


    #Other method, the img can be segmented and labeled individually.
    from skimage.color import label2rgb

    segmentation = ndi.binary_fill_holes(segmentation - 1)
    labeled_img, _ = ndi.label(segmentation)
    image_label_overlay = label2rgb(labeled_img, image=img)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
    axes[0].imshow(orig_crop, cmap=plt.cm.gray, interpolation='nearest')
    axes[0].contour(segmentation, [0.5], linewidths=1.2, colors='y')
    axes[1].imshow(image_label_overlay, interpolation='nearest')

    for a in axes:
        a.axis('off')

    plt.tight_layout()

    plt.show()