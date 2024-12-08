#!/usr/bin/python3

import numpy as np

def compute_normalized_patch_descriptors(
    image_bw: np.ndarray, X: np.ndarray, Y: np.ndarray, feature_width: int
) -> np.ndarray:

    K = len(X)
    D = feature_width * feature_width
    fvs = []

    # 调整 half_width
    if feature_width % 2 == 0:
        half_width = feature_width // 2 - 1
    else:
        half_width = feature_width // 2

    for i in range(K):
        x, y = X[i], Y[i]

        left = int(x) - half_width
        top = int(y) - half_width

        # 检查边界条件，确保索引在图像范围内
        if (left < 0 or top < 0 or
            left + feature_width > image_bw.shape[1] or
            top + feature_width > image_bw.shape[0]):
            continue  # 跳过无法提取完整窗口的关键点

        # 提取窗口
        patch = image_bw[top : top + feature_width, left : left + feature_width]

        # 展平并归一化
        fv = patch.flatten()
        norm = np.linalg.norm(fv)
        if norm != 0:
            fv = fv / norm

        fvs.append(fv)

    fvs = np.array(fvs)

    return fvs

def compute_normalized_patch_descriptors_copilot(
    image_bw: np.ndarray, X: float, Y: float, feature_width: int
) -> np.ndarray:
    """Create local features using normalized patches.

    Normalize image intensities in a local window centered at keypoint to a
    feature vector with unit norm. This local feature is simple to code and
    works OK.

    Choose the top-left option of the 4 possible choices for center of a square
    window.

    Args:
        image_bw: array of shape (M,N) representing grayscale image
        X: array of shape (K,) representing x-coordinate of keypoints
        Y: array of shape (K,) representing y-coordinate of keypoints
        feature_width: size of the square window

    Returns:
        fvs: array of shape (K,D) representing feature descriptors
    """
    K = len(X)
    D = feature_width * feature_width
    fvs = np.zeros((K, D))

    half_width = feature_width // 2

    for i in range(K):
        x = int(X[i])
        y = int(Y[i])
        
        # Ensure the patch is within the image boundaries
        if x - half_width < 0 or x + half_width >= image_bw.shape[1] or \
           y - half_width < 0 or y + half_width >= image_bw.shape[0]:
            continue
        
        patch = image_bw[y - half_width:y + half_width, x - half_width:x + half_width]
        patch = patch.flatten()
        
        # Normalize the patch to have unit norm
        norm = np.linalg.norm(patch)
        if norm != 0:
            patch = patch / norm
        
        fvs[i, :] = patch

    return fvs
