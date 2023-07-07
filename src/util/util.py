# Author: Aria Wang
import json
import pickle
import re
from math import sqrt

import numpy as np
import torch


def zero_strip(s):
    if s[0] == "0":
        s = s[1:]
        return zero_strip(s)
    else:
        return s


def r2_score(Real, Pred):
    # print(Real.shape)
    # print(Pred.shape)
    SSres = np.mean((Real - Pred) ** 2, 0)
    # print(SSres.shape)
    SStot = np.var(Real, 0)
    # print(SStot.shape)
    return np.nan_to_num(1 - SSres / SStot)


def zscore(mat, axis=None):
    if axis is None:
        return (mat - np.mean(mat)) / np.std(mat)
    else:
        return (mat - np.mean(mat, axis=axis, keepdims=True)) / np.std(
            mat, axis=axis, keepdims=True
        )


def fdr_correct_p(var):
    from statsmodels.stats.multitest import fdrcorrection

    n = var.shape[0]
    p_vals = np.sum(var < 0, axis=0) / n  # proportions of permutation below 0
    fdr_p = fdrcorrection(p_vals)  # corrected p
    return fdr_p


def ztransform(val):
    val = np.clip(val, a_min=1e-4, a_max=0.999)
    val = np.log((1 + val) / (1 - val)) / 2.0
    return val


def pearson_corr(X, Y, rowvar=True):
    if rowvar:
        return np.mean(zscore(X, axis=1) * zscore(Y, axis=1), axis=1)
    else:
        return np.mean(zscore(X, axis=0) * zscore(Y, axis=0), axis=0)


def empirical_p(acc, dist, dim=2):
    # dist is permute times x num_voxels
    # acc is of length num_voxels
    if dim == 1:
        return np.sum(dist > acc) / dist.shape[0]
    elif dim == 2:
        assert len(acc) == dist.shape[1]
        ps = list()
        for i, r in enumerate(acc):
            ps.append(np.sum(dist[:, i] > r) / dist.shape[0])
        return ps


def pool_size(fm, dim):
    """
    pool_size() calculates what size avgpool needs to do to reduce the 2d feature into
    desired dimension.
    :param fm: 2D feature/data matrix
    :param dim:
    :param adaptive:
    :return:
    """

    k = 1
    tot = torch.numel(torch.Tensor(fm.view(-1).shape))
    print(tot)
    ctot = tot
    while ctot > dim:
        k += 1
        ctot = tot / k / k
    return k


def check_nans(data, clean=False):
    if np.sum(np.isnan(data)) > 0:
        print("NaNs in the data")
        if clean:
            nan_sum = np.sum(np.isnan(data), axis=1)
            new_data = data[nan_sum < 1, :]
            print("Original data shape is " + data.shape)
            print("NaN free data shape is " + new_data.shape)
            return new_data
    else:
        return data


def pytorch_pca(x):
    # TODO: check this again
    x_mu = x.mean(dim=0, keepdim=True)
    x = x - x_mu

    _, s, v = x.svd()
    s = s.unsqueeze(0)
    nsqrt = sqrt(x.shape[0] - 1)
    xp = x @ (v / s * nsqrt)

    return xp


def pca_test(x):
    from sklearn.decomposition import PCA

    pca = PCA()
    pca.fit(x)
    xp = pca.transform(x)
    return xp


def sum_squared_error(x1, x2):
    return np.sum((x1 - x2) ** 2, axis=0)


def ev(data, biascorr=True):
    """
    Computes the amount of variance in a voxel's response that can be explained by the
    mean response of that voxel over multiple repetitions of the same stimulus.

    If [biascorr], the explainable variance is corrected for bias, and will have mean zero
    for random datasets.

    Data is assumed to be a 2D matrix: time x repeats.
    """
    ev = 1 - np.nanvar(data.T - np.nanmean(data, axis=1)) / np.nanvar(data)
    if biascorr:
        return ev - ((1 - ev) / (data.shape[1] - 1.0))
    else:
        return ev


def generate_rdm(mat, idx=None, avg=False):
    """
    Generate rdm based on data selected by the idx
    idx: lists of index if averaging is not needed; list of list of index if averaging is needed
    """
    from scipy.spatial.distance import pdist, squareform

    if idx is None:
        idx = np.arange(mat.shape[0])

    if type(mat) == list:
        data = np.array(mat)[idx]
        return np.corrcoef(data)
    if avg:
        data = np.zeros((len(idx), mat.shape[1]))
        for i in range(len(idx)):
            data[i] = np.mean(mat[idx[i], :], axis=0)
    else:
        data = mat[idx, :]

    dist = squareform(pdist(data, "cosine"))
    return dist


def negative_tail_fdr_threshold(x, chance_level, alpha=0.05, axis=-1):
    """
    The idea of this is to assume that the noise distribution around the known chance level is symmetric. We can then
    estimate how many of the values at a given level above the chance level are due to noise based on how many values
    there are at the symmetric below chance level.
    Args:
        x: The data
        chance_level: The known chance level for this metric.
            For example, if the metric is correlation, this could be 0.
        alpha: Significance level
        axis: Which axis contains the distribution of values
    Returns:
        The threshold at which only alpha of the values are due to noise, according to this estimation method
    """
    noise_values = np.where(x <= chance_level, x, np.inf)
    # sort ascending, i.e. from most extreme to least extreme
    noise_values = np.sort(noise_values, axis=axis)
    noise_values = np.where(np.isfinite(noise_values), noise_values, np.nan)

    mixed_values = np.where(x > chance_level, x, -np.inf)
    # sort descending, i.e. from most extreme to least extreme
    mixed_values = np.sort(-mixed_values, axis=axis)
    mixed_values = np.where(np.isfinite(mixed_values), mixed_values, np.nan)

    # arange gives the number of values which are more extreme in a sorted array
    num_more_extreme = np.arange(x.shape[axis])
    # if we take these to be the mixed counts, then multiplying by alpha (after including the value itself)
    # gives us the maximum noise counts, which we can use as an index
    # we also add 1 at the end to include the item at that level
    noise_counts = np.ceil(alpha * (num_more_extreme + 1)).astype(np.intp) + 1

    # filter out illegal indexes
    indicator_valid = noise_counts < noise_values.shape[axis]

    noise_values_at_counts = np.take(
        noise_values, noise_counts[indicator_valid], axis=axis
    )
    mixed_values_at_counts = np.take(
        mixed_values, np.arange(mixed_values.shape[axis])[indicator_valid], axis=axis
    )

    # if the (abs) mixed value is greater than the (abs) noise value, we would have to move to the left on the noise
    # counts to get to the mixed value (i.e. the threshold), which is in the direction of decreasing counts. Therefore
    # at this threshold, the fdr is less than alpha
    noise_values_at_counts = np.abs(noise_values_at_counts - chance_level)
    mixed_values_at_counts = np.abs(mixed_values_at_counts - chance_level)
    thresholds = np.where(
        mixed_values_at_counts >= noise_values_at_counts, mixed_values_at_counts, np.nan
    )
    # take the minimum value where this holds
    thresholds = np.nanmin(thresholds, axis=axis)
    return thresholds


if __name__ == "__main__":
    import torch
    from scipy.stats import pearsonr

    # PCA test
    x = np.array([[12.0, -51, 4, 99], [6, 167, -68, -129], [-4, 24, -41, 77]])
    x = torch.from_numpy(x).to(dtype=torch.float64)
    xp1 = pytorch_pca(x)

    xp2 = pca_test(x)
    assert np.sum(abs(xp1.numpy() - xp2) > 0.5) == 0

    # correlation test
    a = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
    b = np.array([[12.0, -51, 4], [99, 6, 167], [-68, -129, -4], [24, -41, 77]])

    corr_row_1 = pearson_corr(a, b, rowvar=True).astype(np.float32)
    corr_row_2 = []
    for i in range(a.shape[0]):
        corr_row_2.append(pearsonr(a[i, :], b[i, :])[0].astype(np.float32))
    assert [corr_row_1[i] == corr_row_2[i] for i in range(len(corr_row_2))]

    corr_col_1 = pearson_corr(a, b, rowvar=False).astype(np.float32)
    corr_col_2 = []
    for i in range(a.shape[1]):
        corr_col_2.append(pearsonr(a[:, i], b[:, i])[0].astype(np.float32))
    assert [corr_col_1[i] == corr_col_2[i] for i in range(len(corr_col_2))]
