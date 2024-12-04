#!/usr/bin/python3

import numpy as np
import torch
import scipy.ndimage

from torch import nn
from typing import Tuple


SOBEL_X_KERNEL = np.array(
    [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]).astype(np.float32)
SOBEL_Y_KERNEL = np.array(
    [
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ]).astype(np.float32)


def compute_image_gradients(image_bw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Use convolution with Sobel filters to compute the image gradient at each
    pixel.

    Args:
        image_bw: A numpy array of shape (M,N) containing the grayscale image

    Returns:
        Ix: Array of shape (M,N) representing partial derivatives of image
            w.r.t. x-direction
        Iy: Array of shape (M,N) representing partial derivative of image
            w.r.t. y-direction
    """
    # Convolution with Sobel_X_KERNEL to get Ix
    Ix = np.zeros_like(image_bw)
    for i in range(0, image_bw.shape[0]):
        for j in range(0, image_bw.shape[1]):
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if 0 <= i + dx < image_bw.shape[0] and 0 <= j + dy < image_bw.shape[1]:
                        Ix[i, j] += image_bw[i + dx, j + dy] * SOBEL_X_KERNEL[dx + 1, dy + 1]
            # Ix[i, j] = np.sum(image_bw[i - 1:i + 2, j - 1:j + 2] * SOBEL_X_KERNEL)

    # Convolution with Sobel_Y_KERNEL to get Iy
    Iy = np.zeros_like(image_bw)
    for i in range(0, image_bw.shape[0]):
        for j in range(0, image_bw.shape[1]):
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if 0 <= i + dx < image_bw.shape[0] and 0 <= j + dy < image_bw.shape[1]:
                        Iy[i, j] += image_bw[i + dx, j + dy] * SOBEL_Y_KERNEL[dx + 1, dy + 1]
            # Iy[i, j] = np.sum(image_bw[i - 1:i + 2, j - 1:j + 2] * SOBEL_Y_KERNEL)

    return Ix, Iy


def get_gaussian_kernel_2D_pytorch(ksize: int, sigma: float) -> torch.Tensor:
    """Create a Pytorch Tensor representing a 2d Gaussian kernel

    Args:
        ksize: dimension of square kernel
        sigma: standard deviation of Gaussian

    Returns:
        kernel: Tensor of shape (ksize,ksize) representing 2d Gaussian kernel

    You should be able to reuse your project 1 Code here.
    """
    x = torch.arange(ksize) - ksize // 2
    y = torch.arange(ksize) - ksize // 2
    xx, yy = torch.meshgrid(x, y)
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel


def second_moments(
    image_bw: np.ndarray,
    ksize: int = 7,
    sigma: float = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Compute second moments from image.

    Compute image gradients Ix and Iy at each pixel, the mixed derivatives,
    then the second moments (sx2, sxsy, sy2) at each pixel, using convolution
    with a Gaussian filter.

    Args:
        image_bw: array of shape (M,N) containing the grayscale image
        ksize: size of 2d Gaussian filter
        sigma: standard deviation of Gaussian filter

    Returns:
        sx2: array of shape (M,N) containing the second moment in x direction
        sy2: array of shape (M,N) containing the second moment in y direction
        sxsy: array of dim (M,N) containing the second moment in the x then the
            y direction
    """
    # copilot ver.
    Ix, Iy = compute_image_gradients(image_bw)
    kernel = get_gaussian_kernel_2D_pytorch(ksize, sigma).numpy()
    kernel = np.flip(kernel)
    sx2 = scipy.ndimage.convolve(Ix**2, kernel, mode="constant", cval=0.0)
    sy2 = scipy.ndimage.convolve(Iy**2, kernel, mode="constant", cval=0.0)
    sxsy = scipy.ndimage.convolve(Ix*Iy, kernel, mode="constant", cval=0.0)
    return sx2, sy2, sxsy

    # doubao ver.
    # Ix, Iy = compute_image_gradients(image_bw)
    # kernel = get_gaussian_kernel_2D_pytorch(ksize, sigma).numpy()
    # conv = nn.Conv2d(1, 1, kernel_size=ksize, padding=ksize // 2, bias=False)
    # conv.weight.data = torch.from_numpy(kernel).view(1, 1, ksize, ksize)
    
    # sx2 = conv(torch.from_numpy(Ix.reshape(1, 1, *Ix.shape)).float()).detach().numpy().reshape(Ix.shape)
    # sy2 = conv(torch.from_numpy(Iy.reshape(1, 1, *Iy.shape)).float()).detach().numpy().reshape(Iy.shape)
    # sxsy = conv(torch.from_numpy((Ix * Iy).reshape(1, 1, *Ix.shape)).float()).detach().numpy().reshape(Ix.shape)
    
    # return sx2, sy2, sxsy

def compute_harris_response_map(
    image_bw: np.ndarray,
    ksize: int = 7,
    sigma: float = 5,
    alpha: float = 0.05
) -> np.ndarray:
    """Compute the Harris cornerness score at each pixel (See Szeliski 7.1.1)

    Recall that R = det(M) - alpha * (trace(M))^2
    where M = [S_xx S_xy;
               S_xy  S_yy],
          S_xx = Gk * I_xx
          S_yy = Gk * I_yy
          S_xy  = Gk * I_xy,
    and * is a convolutional operation over a Gaussian kernel of size (k, k).
    (You can verify that this is equivalent to taking a (Gaussian) weighted sum
    over the window of size (k, k), see how convolutional operation works here:
        http://cs231n.github.io/convolutional-networks/)

    Ix, Iy are simply image derivatives in x and y directions, respectively.
    You may find the Pytorch function nn.Conv2d() helpful here.

    Args:
        image_bw: array of shape (M,N) containing the grayscale image
            ksize: size of 2d Gaussian filter
        sigma: standard deviation of gaussian filter
        alpha: scalar term in Harris response score

    Returns:
        R: array of shape (M,N), indicating the corner score of each pixel.
    """
    sx2, sy2, sxsy = second_moments(image_bw, ksize, sigma)
    det = sx2 * sy2 - sxsy ** 2
    trace = sx2 + sy2
    R = det - alpha * (trace ** 2)
    return R


def maxpool_numpy(R: np.ndarray, ksize: int) -> np.ndarray:
    """ Implement the 2d maxpool operator with (ksize,ksize) kernel size.

    Note: the implementation is identical to my_conv2d_numpy(), except we
    replace the dot product with a max() operator.

    Args:
        R: array of shape (M,N) representing a 2d score/response map

    Returns:
        maxpooled_R: array of shape (M,N) representing the maxpooled 2d
            score/response map
    """
    # my based on doubao
    M, N = R.shape
    maxpooled_R = np.zeros_like(R)
    for i in range(M):
        for j in range(N):
            for dx in range(3):
                for dy in range(3):
                    if 0 <= i + dx - 1 < M and 0 <= j + dy - 1 < N:
                        maxpooled_R[i, j] = max(maxpooled_R[i, j], R[i + dx - 1, j + dy - 1])
    return maxpooled_R

    # copilot
    # output = np.zeros_like(R)
    # for i in range(0, R.shape[0], ksize):
    #     for j in range(0, R.shape[1], ksize):
    #         i_end = min(i + ksize, R.shape[0])
    #         j_end = min(j + ksize, R.shape[1])
    #         output[i:i_end, j:j_end] = np.max(R[i:i_end, j:j_end])
    # return output

    # # doubao
    # M, N = R.shape
    # maxpooled_R = np.zeros((M - ksize + 1, N - ksize + 1))
    # for i in range(M - ksize + 1):
    #     for j in range(N - ksize + 1):
    #         maxpooled_R[i, j] = np.max(R[i:i + ksize, j:j + ksize])
    # return maxpooled_R


def nms_maxpool_pytorch(
    R: np.ndarray,
    k: int,
    ksize: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Get top k interest points that are local maxima over (ksize,ksize)
    neighborhood.

    HINT: One simple way to do non-maximum suppression is to simply pick a
    local maximum over some window size (u, v). This can be achieved using
    nn.MaxPool2d. Note that this would give us all local maxima even when they
    have a really low score compare to other local maxima. It might be useful
    to threshold out low value score before doing the pooling (torch.median
    might be useful here).

    You will definitely need to understand how nn.MaxPool2d works in order to
    utilize it, see https://pytorch.org/docs/stable/nn.html#maxpool2d

    Threshold globally everything below the median to zero, and then
    MaxPool over a 7x7 kernel. This will fill every entry in the subgrids
    with the maximum nearby value. Binarize the image according to
    locations that are equal to their maximum. Multiply this binary
    image, multiplied with the cornerness response values. We'll be testing
    only 1 image at a time.

    Args:
        R: score response map of shape (M,N)
        k: number of interest points (take top k by confidence)
        ksize: kernel size of max-pooling operator

    Returns:
        x: array of shape (k,) containing x-coordinates of interest points
        y: array of shape (k,) containing y-coordinates of interest points
        c: array of shape (k,) containing confidences of interest points
    """
    # copilot
    R = torch.from_numpy(R)
    median_val = torch.median(R)
    R[R < median_val] = 0
    pool = nn.MaxPool2d(ksize, stride=1, padding=ksize//2)
    pooled = pool(R.unsqueeze(0).unsqueeze(0)).squeeze()
    maxima = (pooled == R) * R
    y, x = torch.nonzero(maxima, as_tuple=True)
    confidences = maxima[y, x]
    sorted_confidences, idx = torch.sort(confidences, descending=True)
    return x[idx[:k]].numpy(), y[idx[:k]].numpy(), sorted_confidences[:k].numpy()

    # # doubao
    # R_thresh = np.where(R < np.median(R), 0, R)
    # R_maxpooled = maxpool_numpy(R_thresh, ksize)
    # max_indices = np.argwhere(R_maxpooled == R)
    # top_k_indices = max_indices[np.argsort(R[max_indices[:, 0], max_indices[:, 1]])[-k:]]
    # x = top_k_indices[:, 1]
    # y = top_k_indices[:, 0]
    # c = R[y, x]
    # return x, y, c


def remove_border_vals(
    img: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    c: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove interest points that are too close to a border to allow SIFT feature
    extraction. Make sure you remove all points where a 16x16 window around
    that point cannot be formed.

    Args:
        img: array of shape (M,N) containing the grayscale image
        x: array of shape (k,) representing x coord of interest points
        y: array of shape (k,) representing y coord of interest points
        c: array of shape (k,) representing confidences of interest points

    Returns:
        x: array of shape (p,), where p <= k (less than or equal after pruning)
        y: array of shape (p,)
        c: array of shape (p,)
    """
    M, N = img.shape
    valid_indices = []
    for i in range(len(x)):
        if 16 <= x[i] < N - 16 and 16 <= y[i] < M - 16:
            valid_indices.append(i)
    x = x[valid_indices]
    y = y[valid_indices]
    c = c[valid_indices]
    return x, y, c


def get_harris_interest_points(
    image_bw: np.ndarray,
    k: int = 2500
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Implement the Harris Corner detector. You will find
    compute_harris_response_map(), nms_maxpool_pytorch(), and
    remove_border_vals() useful. Make sure to sort the interest points in
    order of confidence!

    Args:
        image_bw: array of shape (M,N) containing the grayscale image
        k: maximum number of interest points to retrieve

    Returns:
        x: array of shape (p,) containing x-coordinates of interest points
        y: array of shape (p,) containing y-coordinates of interest points
        c: array of dim (p,) containing the strength(confidence) of each
            interest point where p <= k.
    """
    R = compute_harris_response_map(image_bw)
    x, y, c = nms_maxpool_pytorch(R, k, 7)
    x, y, c = remove_border_vals(image_bw, x, y, c)
    sorted_indices = np.argsort(c)[::-1]
    x = x[sorted_indices]
    y = y[sorted_indices]
    c = c[sorted_indices]
    return x[:k], y[:k], c[:k]