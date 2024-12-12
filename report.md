# SIFT Local Feature Matching Project Summary Report

## 1. Introduction
This project focuses on implementing a local feature matching algorithm using techniques related to the Scale-Invariant Feature Transform (SIFT). The goal is to develop a pipeline that can effectively match features between different views of the same physical scene. The report will detail the implementation and results of each part of the project, including interest point detection, local feature description, and feature matching.

## 2. Part 1: Interest Point Detection (part1_harris_corner.py)
### 2.1 Principle
The Harris corner detection algorithm is used to identify interest points in an image. It is based on the auto-correlation matrix \(A\) of the image, which is computed using the gradients \(I_{x}\) and \(I_{y}\) of the image. The auto-correlation matrix is calculated as \(A = w * [I_{x}^{2}, I_{x}I_{y}; I_{x}I_{y}, I_{y}^{2}]\), where \(w\) is a weighting kernel. The Harris corner score \(R\) is then derived from the auto-correlation matrix as \(R = det(A)-\alpha \cdot trace(A)^{2}\), with \(\alpha = 0.06\). Interest points are detected as local maxima of the Harris corner score above a certain threshold.

### 2.2 Process
1. **Compute Image Gradients**: The `compute_image_gradients()` function convolves the original image with a Sobel filter to obtain the horizontal and vertical derivatives \(I_{x}\) and \(I_{y}\).
2. **Create Gaussian Kernel**: The `get_gaussian_kernel_2D_pytorch()` function generates a 2D Gaussian kernel, similar to the method used in project 1.
3. **Compute Second Moments**: The `second_moments()` function calculates the second moments of the input image using the Gaussian kernel obtained in the previous step.
4. **Compute Harris Response Map**: The `compute_harris_response_map()` function computes the raw corner responses over the entire image using the previously calculated gradients and second moments.
5. **Perform Max Pooling (Manual and with PyTorch)**: The `maxpool_numpy()` function performs max pooling using NumPy, which helps in understanding the process. The `nms_maxpool_pytorch()` function then performs non-maximum suppression using PyTorch max-pooling operations to eliminate weaker responses.
6. **Remove Border Values**: The `remove_border_vals()` function removes values close to the border where it is not possible to create a useful SIFT window.
7. **Get Harris Interest Points**: The `get_harris_interest_points()` function finally obtains the interest points from the entire image using the previously implemented methods.

### 2.3 Results
The Harris corner detector successfully identified a set of interest points in the test images. The number of interest points detected depended on the image content and the chosen threshold. By visualizing the interest points, it was observed that they were concentrated in regions with significant changes in intensity, such as corners and edges.

### 2.4 Discussion
The Harris corner detector is a well-known and effective method for interest point detection. However, it has some limitations. It is sensitive to changes in image scale, and the detected interest points may not be truly scale-invariant. Additionally, the choice of threshold can significantly affect the number and quality of the detected interest points. A higher threshold may result in fewer, more stable interest points, while a lower threshold may lead to a larger number of potentially less reliable points. In future work, exploring more advanced interest point detectors that are truly scale-invariant could improve the overall performance of the feature matching pipeline.

## 3. Part 2: Local Feature Descriptors (part2_patch_descriptor.py)
### 3.1 Principle
A simple normalized patch feature is used to describe the local regions around the interest points. The feature is based on the normalized grayscale image intensity patches. The choice of the top-left option as the center of the square window is made to ensure consistency in the feature extraction process.

### 3.2 Process
The `compute_normalized_patch_descriptors()` function is implemented to compute the descriptors. Given an interest point, a square patch centered at the chosen top-left position is extracted from the image. The patch is then normalized to account for differences in illumination. The normalized patch serves as the local feature descriptor for the corresponding interest point.

### 3.3 Results
The normalized patch descriptors provided a basic representation of the local image regions. However, they were relatively simple and might not capture complex image structures as effectively as more advanced descriptors. In the matching process, these descriptors were able to find some correspondences between different views of the same scene, but the accuracy was limited compared to more sophisticated feature description methods.

### 3.4 Discussion
The simplicity of the normalized patch descriptors makes them computationally efficient but less discriminative. They lack the ability to capture the orientation and scale information of the features, which are important for accurate matching in the presence of geometric transformations. To improve the performance of the feature matching, more advanced descriptors that can handle these aspects, such as the SIFT descriptor implemented in part 4, are needed.

## 4. Part 3: Feature Matching (part3_feature_matching.py)
### 4.1 Principle
The "ratio test" (nearest neighbor distance ratio test) method is employed for matching local features. This method compares the distances between a feature in one image and its two nearest neighbors in the other image. If the ratio of the distances is below a certain threshold, the match is considered valid. The underlying idea is that correct matches should have a significantly closer nearest neighbor compared to the second nearest neighbor.

### 4.2 Process
1. **Compute Feature Distances**: The `compute_feature_distances()` function calculates the pairwise distances between the feature descriptors of two images. This is typically done using a distance metric such as the Euclidean distance.
2. **Perform Ratio Test**: The `match_features_ratio_test()` function then applies the ratio test to the computed distances. It compares the ratio of the distances to a predefined threshold (usually set based on empirical analysis). If the ratio is less than the threshold, the corresponding feature pair is considered a match.

### 4.3 Results
The feature matching using the ratio test was able to find a set of corresponding features between different images. The number of correct matches and incorrect matches varied depending on the complexity of the images and the quality of the feature descriptors. In some cases, a significant number of correct matches were obtained, especially for images with relatively small geometric and photometric differences. However, in more challenging scenarios with larger variations in viewpoint, scale, or illumination, the number of incorrect matches increased.

### 4.4 Discussion
The ratio test is a simple yet effective method for feature matching. However, its performance is highly dependent on the quality of the feature descriptors. If the descriptors are not discriminative enough, the ratio test may produce a large number of false positives. Additionally, the choice of the threshold is crucial. A too low threshold may result in missed correct matches, while a too high threshold may lead to an excessive number of incorrect matches. In future work, exploring more advanced matching strategies that can better handle the uncertainties in feature matching could improve the overall accuracy.

## 5. Part 4: SIFT Descriptor (part4_sift_descriptor.py)
### 5.1 Principle
The SIFT-like local feature is implemented to improve the feature description. It is based on histograms of the gradient magnitudes and orientations in the local regions around the interest points. The "Square-Root SIFT" modification from a 2012 CVPR paper is used to enhance the performance.

### 5.2 Process
1. **Get Gradient Magnitudes and Orientations**: The `get_magnitudes_and_orientations()` function computes the gradient magnitudes and orientations of the image. This is done using the image gradients calculated earlier.
2. **Compute Gradient Histogram Vec from Patch**: The `get_gradient_histogram_vec_from_patch()` function constructs the feature descriptor by computing histograms of the gradient magnitudes and orientations within a local patch around each interest point. The histograms are weighted by the gradient magnitudes, and the bins are defined based on the orientation angles.
3. **Get Adjusted Feature from a Single Point**: The `get_feat_vec()` function further processes the histogram-based feature to obtain the final adjusted feature vector for a single interest point.
4. **Get SIFT Descriptors**: The `get_SIFT_descriptors()` function collects all the feature vectors corresponding to the interest points in an image.

### 5.3 Results
The SIFT descriptors provided a more powerful representation of the local features compared to the simple normalized patch descriptors. They were better able to capture the orientation and scale information, resulting in improved matching accuracy. In the experiments, the number of correct matches increased, and the overall performance of the feature matching pipeline was enhanced. The SIFT descriptors were more robust to changes in viewpoint, scale, and illumination, as demonstrated by testing on more challenging image pairs.

### 5.4 Discussion
The SIFT descriptor is a state-of-the-art method for local feature description, and its implementation in this project shows its effectiveness. However, it is computationally more expensive than the simple patch descriptors. The "Square-Root SIFT" modification helps to some extent in improving the performance without significantly increasing the computational cost. Future work could explore further optimizations of the SIFT descriptor to make it more efficient while maintaining its accuracy. Additionally, combining SIFT with other complementary features or using deep learning-based methods could potentially lead to even better results in feature matching.

## 6. Conclusion
In this project, a local feature matching pipeline was implemented, consisting of Harris corner detection for interest point identification, simple normalized patch descriptors and SIFT-like descriptors for feature description, and the ratio test for feature matching. The results showed that while the simple methods provided a basic level of performance, the SIFT-like descriptor significantly improved the matching accuracy, especially in handling geometric and photometric variations. However, there is still room for improvement in terms of computational efficiency and handling more complex scenarios. Future research could focus on exploring more advanced techniques in each step of the pipeline to further enhance the performance of local feature matching algorithms.