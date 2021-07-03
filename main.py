# Name: Breno Lívio Silva de Almeida
# NUSP: 10276675

# SCC0251 - Image Processing
# Project: Segmentation of Cell Cycles Images
# 2021/1

import ImageSegmentation

def main():
    # Generate segmentation masks with region-based segmentation
    # ImageSegmentation.regionBasedSegmentationMasks()
    ImageSegmentation.clusteringBasedSegmentationMasks()

if __name__ == '__main__':
    main()