#!/usr/bin/python3

import copy
import matplotlib.pyplot as plt
import numpy as np
import pdb
import time
import torch

from part1_harris_corner import compute_image_gradients
from torch import nn
from typing import Tuple

"""
Implement SIFT  (See Szeliski 7.1.2 or the original publications here:
    https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf

Your implementation will not exactly match the SIFT reference. For example,
we will be excluding scale and rotation invariance.

You do not need to perform the interpolation in which each gradient
measurement contributes to multiple orientation bins in multiple cells. 
"""


def get_magnitudes_and_orientations(
        Ix: np.ndarray,
        Iy: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function will return the magnitudes and orientations of the
    gradients at each pixel location.

    Args:
        Ix: array of shape (m,n), representing x gradients in the image
        Iy: array of shape (m,n), representing y gradients in the image
    Returns:
        magnitudes: A numpy array of shape (m,n), representing magnitudes of
            the gradients at each pixel location
        orientations: A numpy array of shape (m,n), representing angles of
            the gradients at each pixel location. angles should range from
            -PI to PI.
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    magnitudes = np.sqrt(Ix ** 2 + Iy ** 2)
    orientations = np.arctan2(Iy, Ix)

    # raise NotImplementedError('`get_magnitudes_and_orientations()` function ' +
    #     'in `part4_sift_descriptor.py` needs to be implemented')

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return magnitudes, orientations


def get_gradient_histogram_vec_from_patch(
        window_magnitudes: np.ndarray,
        window_orientations: np.ndarray
) -> np.ndarray:
    """ Given 16x16 patch, form a 128-d vector of gradient histograms.

    Key properties to implement:
    (1) a 4x4 grid of cells, each feature_width/4. It is simply the terminology
        used in the feature literature to describe the spatial bins where
        gradient distributions will be described. The grid will extend
        feature_width/2 - 1 to the left of the "center", and feature_width/2 to
        the right. The same applies to above and below, respectively.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions. The bin centers for the histogram
        should be at -7pi/8,-5pi/8,...5pi/8,7pi/8. The histograms should be
        added to the feature vector left to right then row by row (reading
        order).

    Do not normalize the histogram here to unit norm -- preserve the histogram
    values. A useful function to look at would be np.histogram.

    Args:
        window_magnitudes: (16,16) array representing gradient magnitudes of the
            patch
        window_orientations: (16,16) array representing gradient orientations of
            the patch

    Returns:
        wgh: (128,1) representing weighted gradient histograms for all 16
            neighborhoods of size 4x4 px
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    n = window_magnitudes.shape[0]
    splitted_mags = []
    v_splitted_mags = np.vsplit(window_magnitudes, n // 4)
    for v_splitted_magsi in v_splitted_mags:
        splitted_mags.append(np.hsplit(v_splitted_magsi, n // 4))
    splitted_mags = np.array(splitted_mags)
    flattened_splitted_mags = splitted_mags.reshape((n // 4, n // 4, 16))

    splitted_oris = []
    v_splitted_oris = np.vsplit(window_orientations, n // 4)
    for v_splitted_orisi in v_splitted_oris:
        splitted_oris.append(np.hsplit(v_splitted_orisi, n // 4))
    splitted_oris = np.array(splitted_oris)
    flattened_splitted_oris = splitted_oris.reshape((n // 4, n // 4, 16))
    wgh = []
    for i in range(n // 4):
        for j in range(n // 4):
            histogram = np.histogram(flattened_splitted_oris[i][j], 8,
                                     (-np.pi, np.pi), False, flattened_splitted_mags[i][j])
            wgh.append(histogram[0])
    wgh = np.array(wgh).reshape(8 * (n // 4) ** 2, 1)

    # raise NotImplementedError('`get_gradient_histogram_vec_from_patch` ' +
    #     'function in `part4_sift_descriptor.py` needs to be implemented')

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return wgh


def get_feat_vec(
        c: float,
        r: float,
        magnitudes,
        orientations,
        feature_width: int = 16
) -> np.ndarray:
    """
    This function returns the feature vector for a specific interest point.
    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT descriptor
    (See Szeliski 7.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)
    Your implementation does not need to exactly match the SIFT reference.


    Your (baseline) descriptor should have:
    (1) Each feature should be normalized to unit length.
    (2) Each feature should be raised to the 1/2 power, i.e. square-root SIFT
        (read https://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf)

    For our tests, you do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions.
    The autograder will only check for each gradient contributing to a single bin.

    Args:
        c: a float, the column (x-coordinate) of the interest point
        r: A float, the row (y-coordinate) of the interest point
        magnitudes: A numpy array of shape (m,n), representing image gradients
            at each pixel location
        orientations: A numpy array of shape (m,n), representing gradient
            orientations at each pixel location
        feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e. every
                cell of your local SIFT-like feature will have an integer width
                and height). This is the initial window size we examine around
                each keypoint.
    Returns:
        fv: A numpy array of shape (feat_dim,1) representing a feature vector.
            "feat_dim" is the feature_dimensionality (e.g. 128 for standard SIFT).
            These are the computed features.
    """

    fv = []  # placeholder
    #############################################################################
    # TODO: YOUR CODE HERE                                                      #                                          #
    #############################################################################
    r = int(r)
    c = int(c)
    window_mags = magnitudes[r - (feature_width // 2 - 1):r + (feature_width // 2 + 1),
                  c - (feature_width // 2 - 1):c + (feature_width // 2 + 1)]
    window_oris = orientations[r - (feature_width // 2 - 1):r + (feature_width // 2 + 1),
                  c - (feature_width // 2 - 1):c + (feature_width // 2 + 1)]
    fv = get_gradient_histogram_vec_from_patch(window_mags, window_oris)
    fv = fv / np.linalg.norm(fv)
    fv = np.sqrt(fv)
    # raise NotImplementedError('`get_feat_vec` function in ' +
    #     '`student_sift.py` needs to be implemented')

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return fv


def get_SIFT_descriptors(
        image_bw: np.ndarray,
        X: np.ndarray,
        Y: np.ndarray,
        feature_width: int = 16
) -> np.ndarray:
    """
    This function returns the 128-d SIFT features computed at each of the input
    points. Implement the more effective SIFT descriptor (see Szeliski 7.1.2 or
    the original publications at http://www.cs.ubc.ca/~lowe/keypoints/)

    Args:
        image: A numpy array of shape (m,n), the image
        X: A numpy array of shape (k,), the x-coordinates of interest points
        Y: A numpy array of shape (k,), the y-coordinates of interest points
        feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e.,
            every cell of your local SIFT-like feature will have an integer
            width and height). This is the initial window size we examine
            around each keypoint.
    Returns:
        fvs: A numpy array of shape (k, feat_dim) representing all feature
            vectors. "feat_dim" is the feature_dimensionality (e.g., 128 for
            standard SIFT). These are the computed features.
    """
    assert image_bw.ndim == 2, 'Image must be grayscale'

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    # raise NotImplementedError('`get_SIFT_descriptors` function in ' +
    #     '`part4_sift_descriptor.py` needs to be implemented')
    Ix, Iy = compute_image_gradients(image_bw)
    magnitudes, orientations = get_magnitudes_and_orientations(Ix, Iy)
    (k,) = X.shape
    fvs = np.empty((k, 128, 1))
    for i in range(k):
        fvs[i] = (get_feat_vec(X[i], Y[i], magnitudes, orientations, feature_width))
    (k, feat_dim, _) = fvs.shape
    fvs = fvs.reshape((k, feat_dim))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return fvs


### ----------------- OPTIONAL (below) ------------------------------------

## Implementation of the function below is  optional (extra credit)

def get_sift_features_vectorized(
        image_bw: np.ndarray,
        X: np.ndarray,
        Y: np.ndarray,
        Window_size=16
) -> np.ndarray:
    """
    This function is a vectorized version of `get_SIFT_descriptors`.

    As before, start by computing the image gradients, as done before. Then
    using PyTorch convolution with the appropriate weights, create an output
    with 10 channels, where the first 8 represent cosine values of angles
    between unit circle basis vectors and image gradient vectors at every
    pixel. The last two channels will represent the (dx, dy) coordinates of the
    image gradient at this pixel. The gradient at each pixel can be projected
    onto 8 basis vectors around the unit circle

    Next, the weighted histogram can be created by element-wise multiplication
    of a 4d gradient magnitude tensor, and a 4d gradient binary occupancy
    tensor, where a tensor cell is activated if its value represents the
    maximum channel value within a "fibre" (see
    http://cs231n.github.io/convolutional-networks/ for an explanation of a
    "fibre"). There will be a fibre (consisting of all channels) at each of the
    (M,N) pixels of the "feature map".

    The four dimensions represent (N,C,H,W) for batch dim, channel dim, height
    dim, and weight dim, respectively. Our batch size will be 1.

    In order to create the 4d binary occupancy tensor, you may wish to index in
    at many values simultaneously in the 4d tensor, and read or write to each
    of them simultaneously. This can be done by passing a 1D PyTorch Tensor for
    every dimension, e.g., by following the syntax:
        My4dTensor[dim0_idxs, dim1_idxs, dim2_idxs, dim3_idxs] = 1d_tensor.

    Finally, given 8d feature vectors at each pixel, the features should be
    accumulated over 4x4 subgrids using PyTorch convolution.

    You may find torch.argmax(), torch.zeros_like(), torch.meshgrid(),
    flatten(), torch.arange(), torch.unsqueeze(), torch.mul(), and
    torch.norm() helpful.

    Returns:
        fvs
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    Ix, Iy = compute_image_gradients(image_bw)
    magnitudes, orientations = get_magnitudes_and_orientations(Ix, Iy)
    (k,) = X.shape
    fvs = np.empty((k, (Window_size // 4) ** 2 * 8, 1))
    for i in range(k):
        fvs[i] = (get_feat_vec(X[i], Y[i], magnitudes, orientations, Window_size))
    (k, feat_dim, _) = fvs.shape
    fvs = fvs.reshape((k, feat_dim))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return fvs