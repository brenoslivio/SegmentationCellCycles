# Segmentation of Cell Cycles Images

SCC0251 - Image Processing

Name: Breno Lívio Silva de Almeida, NUSP: 10276675

---

## Introduction

It's known the association of differences in the rates of cellular division and differences in the amount of time spent in each stage of cellular division between healthy and cancer cells [1]. Therefore, it's essential to create methods to analyze images of a process as cell division.

We will explore the following dataset:
https://www.kaggle.com/paultimothymooney/cell-cycle-experiments

The input will be the nematode cells images in the dataset from Kaggle. The [images](https://github.com/brenoslivio/SegmentationCellCycles/tree/main/Data/Original) are divided in interphase and mitosis cycles. There are 90 images in total, being 57 of interphase and 33 of mitosis cycles.

![Inputs](https://raw.githubusercontent.com/brenoslivio/SegmentationCellCycles/main/Images/Inputs.png)

## Objective

With the images, it will be done the process of image segmentation for the Nuclei. Before proper segmentation, the image will be pre-processed using enhancement and filtering techniques. With this, two segmentation methods will be used and compared, Region-Based and Clustering. The segmentations methods will be evaluated using metrics as the Jaccard Index [2].

Note that all images will be segmented so we will have an average score of how accurate a segmentation method is to classify the nuclei in the images.

## Methodology

The project consists of the following pipeline:



### Image pre-processing

Before the process of segmentation, the input image needs essential adjustments. The order used for the project is:

- Convert to grayscale for proper processing. Color is not important here;
- Histogram Equalization for contrast adjustment;
- Gaussian filtering smoothing image for segmentation;

### Segmentation methods

It will be used the following segmentation methods.

#### Region-Based segmentation

Region-Based segmentation can separate the objects into different regions based on a threshold value. With the image converted to grayscale, for example, we can separate the darkest objects from the most enlightened ones. The nuclei are within the darkest objects in these images.

#### Segmentation by Clustering

Segmentation based on clustering can divide the pixels of the image into homogeneous clusters. For this method it will be used the k-means algorithm.

### Evaluating Segmentation methods

To evaluate the segmentations methods it will be created segmentations masks by hand using [labelme](https://github.com/wkentaro/labelme), a tool for Image Polygonal Annotation with Python. The masks are found [here](https://github.com/brenoslivio/SegmentationCellCycles/tree/main/Data/TrueMask).

The segmentations created by hand will be compared to the region-based and clustering methods, calculating the Intersection over Union (IoU) score, the Jaccard Index. 

![IoU](https://raw.githubusercontent.com/brenoslivio/SegmentationCellCycles/main/Images/iou_examples.png)

### Connected-component labeling

After generating segmentation masks we have to label the connected components. Connected-component labeling is used in computer vision to detect connected regions in binary digital images [3]. We can extract the blobs for the Nuclei using this method.

## Examples

For the partial project, it was done the Region-Based Segmentation for the [images](https://github.com/brenoslivio/SegmentationCellCycles/tree/main/Data/Original), generating [segmentation masks](https://github.com/brenoslivio/SegmentationCellCycles/tree/main/Data/Threshold).

We have the following original image (named I4.jpg):

![Original](https://raw.githubusercontent.com/brenoslivio/SegmentationCellCycles/main/Images/I4.jpg)

After pre-processing, using Region-Based Segmentation and applying Connect-component labeling for extracting the nuclei we have:

![Segmented](https://raw.githubusercontent.com/brenoslivio/SegmentationCellCycles/main/Images/I4_threshold.jpg)

The true segmentation mask would be:

![TrueMask](https://raw.githubusercontent.com/brenoslivio/SegmentationCellCycles/main/Images/I4_TrueMask.png)

The IoU score for this case is 0.7130.

## Next project steps

- It will be implemented the k-Means algorithms for Segmentation by Clustering;
- The average IoU score will be calculated for each Segmentation method considering the images;
- Jupyter Notebook with demonstrations showing step by step the segmentation methods;
- Better documented steps.

---

[1] Sherr, C. J. (1996). Cancer cell cycles. Science, 274(5293), 1672-1677.

[2] Wang, Z., Wang, E. & Zhu, Y. Image segmentation evaluation: a survey of methods. Artif Intell Rev 53, 5637–5674 (2020).

[3] He, L., Ren, X., Gao, Q., Zhao, X., Yao, B., & Chao, Y. (2017). The connected-component labeling problem: A review of state-of-the-art algorithms. Pattern Recognition, 70, 25-43.

Jaccard Index image from [pyimagesearch](https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/).