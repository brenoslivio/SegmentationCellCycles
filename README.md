# Segmentation of Cell Cycles Images

SCC0251 - Image Processing

Name: Breno Lívio Silva de Almeida, NUSP: 10276675

---

It's known the association of differences in the rates of cellular division and differences in the amount of time spent in each stage of cellular division between healthy and cancer cells [1]. Therefore, it's essential to create methods to analyze images of a process as cell division.

We will explore the following dataset:
https://www.kaggle.com/paultimothymooney/cell-cycle-experiments

The input will be the nematode cells images in the dataset from Kaggle.

## Objective

With the images, it will be done the process of image segmentation for the Nuclei and classification between interphase and mitosis based on the output of segmentation. 

Before proper segmentation, the image will be pre-processed using enhancement and filtering techniques. With this, two segmentation methods will be used and compared, Region-Based and Clustering.

### Image pre-processing

Before the process of segmentation, the input image needs essential adjustments. The order used for the project is:

- Convert to grayscale;
- Histogram Equalization;
- Gaussian filtering;

### Region-Based Segmentation

Region-Based Segmentation can separate the objects into different regions based on a threshold value. With the image converted to grayscale, we can, for example, separate the darkest objects from the most enlightened ones. The cell are within the darkest objects in these images.

### Segmentation by Clustering

Segmentation based on Clustering can divide the pixels of the image into homogeneous clusters. For this method it will be used the k-means algorithm.

### Classification with Convolutional Neural Network

After the generation of segmented images by the two methods, it will be used a Convolutional Neural Network (CNN) with the objective to analyse with method most constributed for the classification between interphase and mitosis cycles. It will be used 3 datasets for comparing:

- Original images;
- Segmented images produced by Region-Based Semgmentation;
- Segmented images produced by Segmentation based on Clustering;

The Image Segmentation method can be an important addition to image classification problems, considering how it can exclude background elements, possibly improving classification metrics [2].

[1] Sherr, C. J. (1996). Cancer cell cycles. Science, 274(5293), 1672-1677.

[2] Blaschke, T., Burnett, C., & Pekkarinen, A. (2004). Image segmentation methods for object-based analysis and classification. In Remote sensing image analysis: Including the spatial domain (pp. 211-236). Springer, Dordrecht.
