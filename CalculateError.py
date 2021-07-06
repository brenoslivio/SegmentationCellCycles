# Name: Breno Lívio Silva de Almeida
# NUSP: 10276675

# SCC0251 - Image Processing
# Project: Segmentation of Cell Cycles Images
# 2021/1

import ImagePreprocessing
from scipy import stats
import numpy as np
import os
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

def calculateSegmentationError(regionTime, clustTime):
    """
    Read all segmentation masks to compare against true masks, calculating the Jaccard Index.
    A descriptive statistics about the scores is printed.
    """

    inputRefDir = "Data/TrueMask/"
    inputDegImgThreshold = "Data/Threshold/"
    inputDegImgKmeans = "Data/Kmeans/"

    thresholdScores = {}
    kMeansScores = {}

    # List all the folder in the directory
    for _, folder in enumerate(os.listdir(inputRefDir)):
        f = imageio.imread(inputRefDir + folder + "/label.png")

        g = imageio.imread(inputDegImgThreshold + folder + ".jpg")

        # Convert to grayscale
        f = ImagePreprocessing.convertLuminance(f)
        g = ImagePreprocessing.convertLuminance(g)

        # Resize because of size change in the segmentation step
        g = cv2.resize(g, dsize = (256, 256), interpolation = cv2.INTER_CUBIC)

        # Put in the dictionary
        thresholdScores[folder] = jaccardIndex(f, g)

    for _, folder in enumerate(os.listdir(inputRefDir)):
        f = imageio.imread(inputRefDir + folder + "/label.png")

        g = imageio.imread(inputDegImgKmeans + folder + ".jpg")

        f = ImagePreprocessing.convertLuminance(f)
        g = ImagePreprocessing.convertLuminance(g)

        g = cv2.resize(g, dsize = (256, 256), interpolation = cv2.INTER_CUBIC)

        kMeansScores[folder] = jaccardIndex(f, g)

    # Print descriptive statistics

    print("Region-Based Segmentation operation time: {:.4f} s".format(regionTime))
    print("Segmentation by Clustering operation time: {:.4f} s".format(clustTime))
    print("--------------------------------")
    print("Region-based segmentation statistics:\n")
    print("IoU mean score: {:.4f}".format(np.mean(list(thresholdScores.values()))))
    print("IoU median score: {:.4f}".format(np.median(list(thresholdScores.values()))))
    print("IoU variance score: {:.4f}".format(np.var(list(thresholdScores.values()))))
    print("Best image is", max(thresholdScores, key = thresholdScores.get), ", with IoU score: {:.4f}".format(thresholdScores[max(thresholdScores, key=thresholdScores.get)]))
    print("Worst image is", min(thresholdScores, key = thresholdScores.get), ", with IoU score: {:.4f}".format(thresholdScores[min(thresholdScores, key=thresholdScores.get)]))
    print("IoU score 1st Quartile: {:.4f}".format(np.percentile(list(thresholdScores.values()), 25)))
    print("IoU score 3rd Quartile: {:.4f}".format(np.percentile(list(thresholdScores.values()), 75)))
    print("--------------------------------")
    print("Clustering segmentation statistics:\n")
    print("IoU mean score: {:.4f}".format(np.mean(list(kMeansScores.values()))))
    print("IoU median score: {:.4f}".format(np.median(list(kMeansScores.values()))))
    print("IoU variance score: {:.4f}".format(np.var(list(kMeansScores.values()))))
    print("Best image is", max(kMeansScores, key = kMeansScores.get), ", with IoU score: {:.4f}".format(kMeansScores[max(kMeansScores, key=kMeansScores.get)]))
    print("Worst image is", min(kMeansScores, key = kMeansScores.get), ", with IoU score: {:.4f}".format(kMeansScores[min(kMeansScores, key=kMeansScores.get)]))
    print("IoU score 1st Quartile: {:.4f}".format(np.percentile(list(kMeansScores.values()), 25)))
    print("IoU score 3rd Quartile: {:.4f}".format(np.percentile(list(kMeansScores.values()), 75)))
    print("--------------------------------")
    print("Two-sample T-test, p-value: {:.4f}".format(stats.ttest_ind(list(thresholdScores.values()), list(kMeansScores.values()), equal_var = False, alternative = 'less')[1]))