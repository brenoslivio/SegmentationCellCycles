# Name: Breno Lívio Silva de Almeida
# NUSP: 10276675

# SCC0251 - Image Processing
# Project: Segmentation of Cell Cycles Images
# 2021/1

import imageio
import matplotlib.pyplot as plt
import numpy as np

def saveSegmentedImg(img):
    """
    Save segmented image.

    Parameters: img, image for saving.
    """

    plt.figure(figsize = (10, 10))
    plt.axis("off")
    plt.imshow(img, cmap = "gray")
    plt.savefig('Segmented.png', bbox_inches = 'tight', transparent = True, pad_inches = 0)

def convertLuminance(img):
    """
    Convert to Grayscale using Luminance method.

    Parameters: img, image to be converted.

    Returns: imgGray, image converted to grayscale.
    """

    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    imgGray = (0.299 * R + 0.587 * G + 0.114 * B).astype(np.uint8)

    return imgGray

def scalingImage(img, minVal, maxVal):
    """
    Scale image given a range.

    Parameters: img, image to be scaled;
                minVal, lower value for range;
                maxVal, upper value for range.

    Returns: imgScaled, image scaled.
    """

    imax = np.max(img)
    imin = np.min(img)

    std = (img - imin) / (imax - imin)

    imgScaled = std * (maxVal - minVal) + minVal

    return imgScaled

def histogramEqualization(img):
    """
    Histogram equalization for contrast adjustment using the image's histogram.

    Parameters: img, image to apply histogram equalization.

    Returns: imgEq, image with histogram equalization applied.
    """

    # Creation of image's cumulative histogram

    hist = np.zeros(256).astype(int)

    for i in range(256):
        pixels_value_i = np.sum(img == i)
        hist[i] = pixels_value_i

    histC = np.zeros(256).astype(int)

    histC[0] = hist[0]

    for i in range(1,  256):
        histC[i] = hist[i] + histC[i-1]

    N, M = img.shape
    
    imgEq = np.zeros([N,M]).astype(np.uint8)
    
    # For each intensity value, transforms in a new intensity
    for z in range(256):
        # Transformation function
        s = ((256 - 1)/float(M * N)) * histC[z]
        
        # Apply equalized value
        imgEq[np.where(img == z)] = s
        
    return imgEq

def gaussianFilter(img, k = 15, sigma = 10):
    """
    Gaussian filter applying for the image using symmetric padding.

    Parameters: img, image to apply gaussian filter;
                k, size of the filter;
                sigma, control the variation around the filter's mean value.

    Returns: imgGaussian, image with gaussian filter applied.
    """

    # Calculate gaussian filter

    arx = np.arange((-k // 2) + 1.0, (k // 2) + 1.0)
    x, y = np.meshgrid(arx, arx)
    filt = np.exp(-(1/2) * (np.square(x) + np.square(y)) / np.square(sigma))
    filter2d = filt / np.sum(filt)

    center = k // 2
    padImg = np.pad(img, pad_width = center, mode = "symmetric")
    N, M = img.shape

    # Convolution using gaussian filter

    imgGaussian = [[np.multiply(filter2d, padImg[(x - center):(x + center + 1), (y - center):(y + center + 1)]).sum() 
            for y in range(center, M + center)] for x in range(center, N + center)]

    imgGaussian = scalingImage(imgGaussian, 0, 1)

    return imgGaussian

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

def connectedComponents(img):
    """
    Connected-component labeling using 4-neighbors method.
    It was implemented using the Wikipedia article as a guide:
    https://en.wikipedia.org/wiki/Connected-component_labeling
    It's based on Hoshen–Kopelman algorithm.

    Parameters: img, image to apply labeling.

    Returns: labels, image with labeling applied.
    """

    equivList = {}
    N, M = img.shape
    labels = np.zeros(img.shape)
    label = 1

    def get4Neighbors(img, i, j):
        """
        Get the 4 neighbours for the pixel analysed.

        Parameters: img, image;
                    i, row number;
                    j, column number.

        Returns: neighbors, list of neighbors.
        """

        N, M = img.shape
        neighbors = []

        if i - 1 >= 0:
            neighbors.append(img[i-1][j])
        if j - 1 >= 0:
            neighbors.append(img[i][j-1])
        if j - 1 >= 0 and i - 1 >= 0:
            neighbors.append(img[i-1][j-1])
        if j + 1 < M and i - 1 >= 0:
            neighbors.append(img[i-1][j+1])

        return neighbors

    def find(label, equivList):
        """
        Find the neighbor with the smallest label and assign it to the current element.

        Parameters: label, label number;
                    equivList, equivalence relatioship list.

        Returns: minVal, smallest label.
        """

        minVal = min(equivList[label])
        while label != minVal:
            label = minVal
            minVal = min(equivList[label])

        return minVal

    # First pass

    for i in range(N):
        for j in range(M):
            if img[i][j] == 1:
                neighbors = get4Neighbors(labels, i, j)
                neighbors = list(filter(lambda a: a != 0, neighbors))

                if len(neighbors) == 0:
                    labels[i][j] = label
                    equivList[label] = set([label])
                    label += 1
                else:
                    minVal = min(neighbors)
                    labels[i][j] = minVal
                    for l in neighbors:
                        equivList[l] = set.union(equivList[l], neighbors)

    # Second pass

    finalLabels = {}
    newLabel = 1

    for i in range(N):
        for j in range(M):
            if labels[i][j] != 0:
                new = find(labels[i][j], equivList)
                labels[i][j] = new

                if new not in finalLabels:
                    finalLabels[new] = newLabel
                    newLabel += 1

    for i in range(N):
        for j in range(M):
            if labels[i][j] != 0:
                labels[i][j] = finalLabels[labels[i][j]]

    return labels

def centerOfMass(maskLabels, labels):
    """
    Get the center of mass for each label.

    Parameters: maskLabels, image with labels defined;
                labels, list of labels in the image.

    Returns: centers, list of label' centers of mass.
    """

    centers = []

    for i in labels:
        xMass, yMass = np.where(maskLabels == i)

        xCenter = int(np.average(xMass))
        yCenter = int(np.average(yMass))

        centers.append([xCenter, yCenter])

    return centers

def nearDistance(img, centers):
    """
    Get the blob nearest to the image center, which is probably the blob for cells, using euclidian distance.

    Parameters: img, image with labels defined;
                centers, list of label' centers of mass.

    Returns: nearestLabel, label nearest to the image center.
    """

    N, M = img.shape

    imgCenter = [N//2, M//2]
    distances = []

    for center in centers:
        distances.append(np.linalg.norm(np.array(imgCenter) - np.array(center)))

    nearestLabel = np.argmin(distances)

    return nearestLabel

def main():
    cellOriginal = imageio.imread('Data/mitosis/IMG_1684-4.jpg')

    cellGray = convertLuminance(cellOriginal)

    cellEq = histogramEqualization(cellGray)

    cellGauss = gaussianFilter(cellEq)

    cellThresh = thresholdRegion(cellGauss)

    mask = np.ones(cellThresh.shape)
    mask[np.where(cellThresh == 0)] = 0
    mask = ~(mask.astype(np.uint8))

    mask = scalingImage(mask, 0, 1)

    maskLabels = connectedComponents(mask)

    labels = np.unique(maskLabels)

    centers = centerOfMass(maskLabels, labels)

    nearestBlob = nearDistance(maskLabels, centers)

    segmented = np.zeros(mask.shape)
    segmented[np.where(maskLabels == nearestBlob)] = 1

    saveSegmentedImg(segmented)

if __name__ == '__main__':
    main()