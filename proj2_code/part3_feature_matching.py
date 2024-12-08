#!/usr/bin/python3

import numpy as np

from typing import Tuple


def compute_feature_distances(
    features1: np.ndarray,
    features2: np.ndarray
) -> np.ndarray:
    """
    This function computes a list of distances from every feature in one array
    to every feature in another.

    Using Numpy broadcasting is required to keep memory requirements low.

    Note: Using a double for-loop is going to be too slow. One for-loop is the
    maximum possible. Vectorization is needed.
    See numpy broadcasting details here:
        https://cs231n.github.io/python-numpy-tutorial/#broadcasting

    Args:
        features1: A numpy array of shape (n1,feat_dim) representing one set of
            features, where feat_dim denotes the feature dimensionality
        features2: A numpy array of shape (n2,feat_dim) representing a second
            set of features (n1 not necessarily equal to n2)

    Returns:
        dists: A numpy array of shape (n1,n2) which holds the distances (in
            feature space) from each feature in features1 to each feature in
            features2
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    sum1 = np.sum(features1**2, axis=1).reshape(-1, 1)  # 形状 (n1, 1)
    # 计算 features2 的平方和并在行方向扩展
    sum2 = np.sum(features2**2, axis=1).reshape(1, -1)  # 形状 (1, n2)
    # 计算点积
    inner_product = np.dot(features1, features2.T)  # 形状 (n1, n2)
    # 使用欧氏距离公式的变形
    dists = np.sqrt(sum1 - 2 * inner_product + sum2)  # 形状 (n1, n2)
    return dists
    # raise NotImplementedError('`compute_feature_distances` function in ' +
    #     '`part3_feature_matching.py` needs to be implemented')

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dists


def match_features_ratio_test(
    features1: np.ndarray,
    features2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """ Nearest-neighbor distance ratio feature matching.

    This function does not need to be symmetric (e.g. it can produce different
    numbers of matches depending on the order of the arguments).

    To start with, simply implement the "ratio test", equation 7.18 in section
    7.1.3 of Szeliski. There are a lot of repetitive features in these images,
    and all of their descriptors will look similar. The ratio test helps us
    resolve this issue (also see Figure 11 of David Lowe's IJCV paper).

    You should call `compute_feature_distances()` in this function, and then
    process the output.

    Args:
        features1: A numpy array of shape (n1,feat_dim) representing one set of
            features, where feat_dim denotes the feature dimensionality
        features2: A numpy array of shape (n2,feat_dim) representing a second
            set of features (n1 not necessarily equal to n2)

    Returns:
        matches: A numpy array of shape (k,2), where k is the number of matches.
            The first column is an index in features1, and the second column is
            an index in features2
        confidences: A numpy array of shape (k,) with the real valued confidence
            for every match

    'matches' and 'confidences' can be empty, e.g., (0x2) and (0x1)
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    distances = compute_feature_distances(features1, features2)  # shape: (n1, n2)

    matches = []
    confidences = []

    for i in range(distances.shape[0]):
        # 获取第 i 个特征与所有特征的距离
        dists = distances[i, :]

        # 找到最近邻和次近邻的索引
        idx_sorted = np.argsort(dists)
        nearest = idx_sorted[0]
        second_nearest = idx_sorted[1]

        # 计算距离比率
        ratio = dists[nearest] / dists[second_nearest]

        # 设定阈值，通常为 0.8
        if ratio < 0.8:
            matches.append([i, nearest])
            confidences.append(1 - ratio)  # 置信度可以设为 1 - ratio

    matches = np.array(matches)
    confidences = np.array(confidences)

    # 按置信度从高到低排序
    sorted_indices = np.argsort(-confidences)
    matches = matches[sorted_indices]
    confidences = confidences[sorted_indices]
    # raise NotImplementedError('`match_features_ratio_test` function in ' +
    #     '`part3_feature_matching.py` needs to be implemented')

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return matches, confidences
