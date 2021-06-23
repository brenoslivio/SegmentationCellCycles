import numpy as np
import imageio
import cv2

def convertLuminance(img):
    """
    Convert to Grayscale using Luminance method.

    Parameters: img, image to be converted.

    Returns: imgGray, image converted to grayscale.
    """

    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    imgGray = np.floor((0.299 * R + 0.587 * G + 0.114 * B)).astype(np.uint8)

    return imgGray

def jaccardIndex(f, g):
    """
    Calculate Jaccard Index.

    Parameters: f, true mask;
                g, segmented mask.

    Returns: iou_score, Intersection over Union score.
    """

    intersection = np.logical_and(f, g)
    union = np.logical_or(f, g)
    iou_score = np.sum(intersection) / np.sum(union)

    return iou_score

def main():
    inputRefImg = "Data/TrueMask/M27/label.png"

    inputDegImg = "Data/Threshold/M27.jpg"

    f = imageio.imread(inputRefImg)

    g = imageio.imread(inputDegImg)

    f = convertLuminance(f)
    g = convertLuminance(g)

    g = cv2.resize(g, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)

    print("{:.4f}".format(jaccardIndex(f, g)))

if __name__ == '__main__':
    main()