import os
import numpy as np
import skimage # v0.20.0
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation
import matplotlib
matplotlib.use('agg') 
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# takes a color image
# returns a list of bounding boxes and black_and_white image
def find_letters(image):
    # ocr processing here, one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes
    # https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_label.html
    sigma = skimage.restoration.estimate_sigma(image, average_sigmas=True, channel_axis=-1) # estimate noise
    image = skimage.filters.gaussian(image, sigma=sigma, channel_axis=-1) # Gaussian denoising
    image = skimage.color.rgb2gray(image) # Greyscale
    threshold = skimage.filters.threshold_otsu(image) # Threshold
    bw = skimage.morphology.closing(image < threshold, skimage.morphology.square(10)) # Morphology closing
    clear = skimage.segmentation.clear_border(bw) # Remove border artifacts
    label_image, _ = skimage.measure.label(clear, background=0, return_num=True, connectivity=2) # Label connected components
    # Filter regions and skip small boxes
    regions = skimage.measure.regionprops(label_image)
    areas = np.array([region.area for region in regions])
    area_thresh = np.sum(areas) / len(regions) / 3
    bboxes = []
    for region in regions:
        if region.area >= area_thresh:
            bboxes.append(region.bbox)
    # Bitwise NOT operation to invert black and white regions in binary image for visualization, Convert to float for visualization
    return bboxes, (~bw).astype(float)

if __name__ == "__main__":
    ARTIFACTS_DIR = os.getcwd() + "/artifacts"
    IMAGE_DIR = '../data/images'
    for img_filename in os.listdir(IMAGE_DIR):
        try:
            img_path = os.path.join(IMAGE_DIR, img_filename)
            if os.path.isfile(img_path) and img_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                im = skimage.img_as_float(skimage.io.imread(img_path))
                bboxes, bw = find_letters(im)
                plt.imshow(bw, cmap='gray')
                plt.axis('off')
                for bbox in bboxes:
                    minr, minc, maxr, maxc = bbox
                    rect = patches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='green', linewidth=1)
                    plt.gca().add_patch(rect)
                plt.savefig(ARTIFACTS_DIR + '/detected_' + img_filename)
                plt.close()
        except Exception as e:
            print(f"Failed with exception {e} for {img_path}")
