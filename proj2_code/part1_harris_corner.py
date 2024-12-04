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
    Ix = scipy.ndimage.convolve(image_bw, SOBEL_X_KERNEL)
    Iy = scipy.ndimage.convolve(image_bw, SOBEL_Y_KERNEL)
    return Ix, Iy


def get_gaussian_kernel_2D_pytorch(ksize: int, sigma: float) -> torch.Tensor:
    ax = torch.arange(-ksize // 2 + 1., ksize // 2 + 1.)
    xx, yy = torch.meshgrid(ax, ax)
    kernel = torch.exp(-0.5 * (xx**2 + yy**2) / sigma**2)
    kernel = kernel / torch.sum(kernel)
    return kernel


def second_moments(
    image_bw: np.ndarray,
    ksize: int = 7,
    sigma: float = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Ix, Iy = compute_image_gradients(image_bw)
    kernel = get_gaussian_kernel_2D_pytorch(ksize, sigma).numpy()
    sx2 = scipy.ndimage.convolve(Ix**2, kernel)
    sy2 = scipy.ndimage.convolve(Iy**2, kernel)
    sxsy = scipy.ndimage.convolve(Ix*Iy, kernel)
    return sx2, sy2, sxsy


def compute_harris_response_map(
    image_bw: np.ndarray,
    ksize: int = 7,
    sigma: float = 5,
    alpha: float = 0.05
) -> np.ndarray:
    sx2, sy2, sxsy = second_moments(image_bw, ksize, sigma)
    det_M = sx2 * sy2 - sxsy**2
    trace_M = sx2 + sy2
    R = det_M - alpha * trace_M**2
    return R


def maxpool_numpy(R: np.ndarray, ksize: int) -> np.ndarray:
    output = np.zeros_like(R)
    for i in range(0, R.shape[0] - ksize + 1, ksize):
        for j in range(0, R.shape[1] - ksize + 1, ksize):
            output[i:i+ksize, j:j+ksize] = np.max(R[i:i+ksize, j:j+ksize])
    return output


def nms_maxpool_pytorch(
    R: np.ndarray,
    k: int,
    ksize: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def remove_border_vals(
    img: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    c: np.ndarray
) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    mask = (x >= 8) & (x < img.shape[1] - 8) & (y >= 8) & (y < img.shape[0] - 8)
    return x[mask], y[mask], c[mask]


def get_harris_interest_points(
    image_bw: np.ndarray,
    k: int = 2500
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    R = compute_harris_response_map(image_bw)
    x, y, confidences = nms_maxpool_pytorch(R, k, 7)
    x, y, confidences = remove_border_vals(image_bw, x, y, confidences)
    return x, y, confidences
