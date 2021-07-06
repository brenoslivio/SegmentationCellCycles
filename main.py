# Name: Breno Lívio Silva de Almeida
# NUSP: 10276675

# SCC0251 - Image Processing
# Project: Segmentation of Cell Cycles Images
# 2021/1

import ImageSegmentation
import CalculateError
import time

def main():
    startRegion = time.time() # To count operation time
    # Generate segmentation masks with region-based segmentation
    ImageSegmentation.regionBasedSegmentationMasks()
    endRegion = time.time()
    regionTime = endRegion - startRegion

    startClust = time.time() # To count operation time
    # Generate segmentation masks with clustering segmentation
    ImageSegmentation.clusteringBasedSegmentationMasks()
    endClust = time.time()
    clustTime = endClust - startClust

    # Calculate error from segmentations
    CalculateError.calculateSegmentationError(regionTime, clustTime)

if __name__ == '__main__':
    main()