# Name: Breno Lívio Silva de Almeida
# NUSP: 10276675

# SCC0251 - Image Processing
# Project: Segmentation of Cell Cycles Images
# 2021/1

import numpy as np

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

        xCenter = 0; yCenter = 0

        # Check if blob is of adequate size
        blobSize = len(np.where(maskLabels == i)[0])
        if blobSize > 500 and blobSize < 10000:
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