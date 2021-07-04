# Segmentation of Cell Cycles Images

SCC0251 - Image Processing

Name: Breno Lívio Silva de Almeida, NUSP: 10276675

---

You view the Jupyter Notebook for demonstrations here: [Jupyter Notebook](https://nbviewer.jupyter.org/github/brenoslivio/SegmentationCellCycles/blob/main/SegCellCycles.ipynb)

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

![pipeline](https://raw.githubusercontent.com/brenoslivio/SegmentationCellCycles/main/Images/Pipeline.png)

### Image pre-processing

Before the process of segmentation, the input image needs essential adjustments. The order used for the project is:

- Convert to grayscale for proper processing;
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

Considering the classification idea of True Positive (TP), True Negative (TN), False Positive (FP) and False Negative (FN), we have the following equation:

![latex]()

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

## Results and Discussion

After generating the segmentation masks and making the evaluation, descriptive statistics are informed. Considering the project reproducibility, it's expected the output to be similar to [this](https://github.com/brenoslivio/SegmentationCellCycles/blob/main/results.txt).

We can see both segmentation IoU mean scores are practically identical. There are some differences related to the median and other percentiles.

A way to analyze if there's some statistical difference to the methods is to use techniques as hypothesis tests. Taking a Two-sample T-test with the IoU scores we have a p-value of 0.98. With this, we fail to reject the null hypothesis and it would be really hard to choose a segmentation method as the best.

However, we can see that in some cases a method could be more suited to a task. KMeans method is really useful for dealing with nuclei around darker regions.

Threshold example:

![threshold](https://raw.githubusercontent.com/brenoslivio/SegmentationCellCycles/main/Data/Threshold/I3.jpg)

Clustering example:

![clustering](https://raw.githubusercontent.com/brenoslivio/SegmentationCellCycles/main/Data/Kmeans/I3.jpg)

And of course, the original image:

![originalimage](https://raw.githubusercontent.com/brenoslivio/SegmentationCellCycles/main/Data/Original/I3.jpg)

We could adjust the threshold value and the result would be possibly the same, but this would require an individual image approach. Clustering could be more generalist and better for dealing with more images.

Another important observation is that hyperparameter tuning for the segmentation methods could be used to have a better performance, but it would require an extensive amount of time for the case of this project.

Techniques as dilation from mathematical morphology could be used mainly with clustering segmentation, considering how this method disconsidered some nuclei borders.

Note that the Jaccard Index doesn't use True Negative calculation in the formula. Other metrics could be explored, and we could see a better difference between these two segmentation methods. 

---

## References

[1] Sherr, C. J. (1996). Cancer cell cycles. Science, 274(5293), 1672-1677.

[2] Wang, Z., Wang, E. & Zhu, Y. Image segmentation evaluation: a survey of methods. Artif Intell Rev 53, 5637–5674 (2020).

[3] He, L., Ren, X., Gao, Q., Zhao, X., Yao, B., & Chao, Y. (2017). The connected-component labeling problem: A review of state-of-the-art algorithms. Pattern Recognition, 70, 25-43.

Jaccard Index image from [pyimagesearch](https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/).