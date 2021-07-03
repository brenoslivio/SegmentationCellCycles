# Name: Breno Lívio Silva de Almeida
# NUSP: 10276675

# SCC0251 - Image Processing
# Project: Segmentation of Cell Cycles Images
# 2021/1

import ImageSegmentation
import CalculateError

def main():
    # Generate segmentation masks with region-based segmentation
    ImageSegmentation.regionBasedSegmentationMasks()
    # Generate segmentation masks with clustering segmentation
    ImageSegmentation.clusteringBasedSegmentationMasks()

    # Calculate error from segmentations
    CalculateError.calculateSegmentationError()

if __name__ == '__main__':
    main()