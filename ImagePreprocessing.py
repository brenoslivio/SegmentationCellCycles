# Name: Breno Lívio Silva de Almeida
# NUSP: 10276675

# SCC0251 - Image Processing
# Project: Segmentation of Cell Cycles Images
# 2021/1

import numpy as np

def convertLuminance(img):
    """
    Convert to Grayscale using Luminance method.

    Parameters: img, image to be converted.

    Returns: imgGray, image converted to grayscale.
    """

    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    imgGray = np.floor((0.299 * R + 0.587 * G + 0.114 * B)).astype(np.uint8)

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