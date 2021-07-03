# Name: Breno Lívio Silva de Almeida
# NUSP: 10276675

# SCC0251 - Image Processing
# Project: Segmentation of Cell Cycles Images
# 2021/1

import ImagePreprocessing, ConnectedComponent
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import imageio
import os

def saveSegmentedImg(img, name, dest):
    """
    Save segmented image.

    Parameters: img, image for saving.
    """

    plt.axis("off")
    plt.imshow(img, cmap = "gray")
    plt.savefig(dest + name, bbox_inches = 'tight', transparent = True, pad_inches = 0)

def thresholdRegion(img):
    """
    Region-based segmentation using local thresholds in the grayscale.

    Parameters: img, image to apply thresholding.

    Returns: imgThresh, image with threshold applied.
    """

    imgThresh = img.reshape(img.shape[0] * img.shape[1])

    for i in range(imgThresh.shape[0]):
        if imgThresh[i] > imgThresh.mean():
            imgThresh[i] = 3
        elif imgThresh[i] > 0.5:
            imgThresh[i] = 2
        elif imgThresh[i] > 0.2:
            imgThresh[i] = 1
        else: # The cells are within this range, being one of the darkest
            imgThresh[i] = 0

    imgThresh = imgThresh.reshape(img.shape[0], img.shape[1])

    return imgThresh

def regionBasedSegmentationMasks():
    """
    Generate segmentation masks with region-based segmentation.
    """

    src = "Data/Original/"

    # List the images in the directory
    for _, filename in enumerate(os.listdir(src)):
        cellOriginal = imageio.imread(src + filename, pilmode="RGB")

        # Convert to grayscale
        cellGray = ImagePreprocessing.convertLuminance(cellOriginal)

        # Histogram equalization
        cellEq = ImagePreprocessing.histogramEqualization(cellGray)

        # Gaussian filter
        cellGauss = ImagePreprocessing.gaussianFilter(cellEq, k = 15, sigma = 10)

        # Apply threshold
        cellThresh = thresholdRegion(cellGauss)

        # Create white mask to add the darkest regions in the threshold generated
        mask = np.ones(cellThresh.shape)
        mask[np.where(cellThresh == 0)] = 0
        mask = ~(mask.astype(np.uint8))

        mask = ImagePreprocessing.scalingImage(mask, 0, 1)

        # Apply connected-component labeling
        maskLabels = ConnectedComponent.connectedComponents(mask)

        labels = np.unique(maskLabels)

        # Calculate centers of mass for the blobs
        centers = ConnectedComponent.centerOfMass(maskLabels, labels)

        # Find nearest blob from the center, which is our nuclei
        nearestBlob = ConnectedComponent.nearDistance(maskLabels, centers)

        # Isolate blob
        segmented = np.zeros(mask.shape)
        segmented[maskLabels == nearestBlob] = 1

        dest = "Data/Threshold/"

        saveSegmentedImg(segmented, filename, dest)

def clusteringBasedSegmentationMasks():
    """
    Generate segmentation masks with region-based segmentation.
    """

    src = "Data/Original/"

    for _, filename in enumerate(os.listdir(src)):
        cellOriginal = imageio.imread(src + filename, pilmode="RGB")

        cellGray = ImagePreprocessing.convertLuminance(cellOriginal)

        cellEq = ImagePreprocessing.histogramEqualization(cellGray)

        cellGauss = ImagePreprocessing.gaussianFilter(cellEq, k = 15, sigma = 10)

        # Flatten for applying KMeans
        gaussFlatten = cellGauss.reshape(-1, 1)

        # Applying clustering segmentation with KMeans
        clusters = KMeans(n_clusters = 13, random_state = 0).fit(gaussFlatten)
        clusterImg = clusters.cluster_centers_[clusters.labels_]
        clusterImg = clusterImg.reshape(cellGauss.shape)

        clusterImg = ImagePreprocessing.scalingImage(clusterImg, 0, 1)

        mask = np.ones(clusterImg.shape)
        mask[np.where(clusterImg == 0)] = 0
        mask = ~(mask.astype(np.uint8))

        mask = ImagePreprocessing.scalingImage(mask, 0, 1)

        maskLabels = ConnectedComponent.connectedComponents(mask)

        labels = np.unique(maskLabels)

        centers = ConnectedComponent.centerOfMass(maskLabels, labels)

        nearestBlob = ConnectedComponent.nearDistance(maskLabels, centers)

        segmented = np.zeros(mask.shape)
        segmented[maskLabels == nearestBlob] = 1

        dest = "Data/Kmeans/"

        saveSegmentedImg(segmented, filename, dest)