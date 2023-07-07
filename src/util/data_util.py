import pickle
from matplotlib.pyplot import fill
import numpy as np
import pandas as pd
from torch import neg
from util.model_config import COCO_cat, COCO_super_cat


def fill_in_nan_voxels(vals, subj, output_root, fill_in=0):
    try:  # some subject has zeros voxels masked out
        nonzero_mask = np.load(
            "%s/output/voxels_masks/subj%d/nonzero_voxels_subj%02d.npy"
            % (output_root, subj, subj)
        )
        if type(vals) is list:
            tmp = np.zeros(nonzero_mask.shape) + fill_in
            tmp[nonzero_mask] = vals
            return tmp
        elif len(vals.shape) == 1:
            tmp = np.zeros(nonzero_mask.shape) + fill_in
            tmp[nonzero_mask] = vals
            return tmp
        elif len(vals.shape) == 2:
            tmp = np.zeros((vals.shape[0], len(nonzero_mask))) + fill_in
            tmp[:, nonzero_mask] = vals
            return tmp
    except FileNotFoundError:
        return vals


def load_model_performance(model, output_root=".", subj=1, measure="corr"):
    if measure == "pvalue":
        measure = "corr"
        pvalue = True
    else:
        pvalue = False

    if type(model) == list:
        # to accomodate different naming of the same model
        for m in model:
            try:
                out = np.load(
                    "%s/output/encoding_results/subj%d/%s_%s_whole_brain.p"
                    % (output_root, subj, measure, m),
                    allow_pickle=True,
                )
            except FileNotFoundError:
                continue
    else:
        out = np.load(
            "%s/output/encoding_results/subj%d/%s_%s_whole_brain.p"
            % (output_root, subj, measure, model),
            allow_pickle=True,
        )

    if measure == "corr":
        if pvalue:
            out = np.array(out)[:, 1]
            out = fill_in_nan_voxels(out, subj, output_root, fill_in=1)
            return out
        else:
            out = np.array(out)[:, 0]

    out = fill_in_nan_voxels(out, subj, output_root)

    return np.array(out)


def load_top1_objects_in_COCO(cid):
    try:
        stim = pd.read_pickle(
            "/lab_data/tarrlab/common/datasets/NSD/nsddata/experiments/nsd/nsd_stim_info_merged.pkl"
        )
    except FileNotFoundError:
        stim = pd.read_pickle("nsddata/nsd_stim_info_merged.pkl")
    try:
        cat = np.load("/lab_data/tarrlab/common/datasets/features/NSD/COCO_Cat/cat.npy")
    except FileNotFoundError:
        cat = np.load("features/cat.npy")

    # extract the nsd ID corresponding to the coco ID in the stimulus list
    stim_ind = stim["nsdId"][stim["cocoId"] == cid]
    # extract the respective features for that nsd ID
    catID_of_trial = cat[stim_ind, :]
    catnm = COCO_cat[np.argmax(catID_of_trial)]
    return catnm


def load_objects_in_COCO(cid):
    try:
        stim = pd.read_pickle(
            "/lab_data/tarrlab/common/datasets/NSD/nsddata/experiments/nsd/nsd_stim_info_merged.pkl"
        )
    except FileNotFoundError:
        stim = pd.read_pickle("nsddata/nsd_stim_info_merged.pkl")
    try:
        cat = np.load("/lab_data/tarrlab/common/datasets/features/NSD/COCO_Cat/cat.npy")
        supcat = np.load(
            "/lab_data/tarrlab/common/datasets/features/NSD/COCO_Cat/supcat.npy"
        )
    except FileNotFoundError:
        cat = np.load("features/cat.npy")
        supcat = np.load("features/supcat.npy")

    # extract the nsd ID corresponding to the coco ID in the stimulus list
    stim_ind = stim["nsdId"][stim["cocoId"] == cid]
    # extract the repective features for that nsd ID
    catID_of_trial = cat[stim_ind, :].squeeze()
    supcatID_of_trial = supcat[stim_ind, :].squeeze()
    catnms = []

    assert len(catID_of_trial) == len(COCO_cat)
    assert len(supcatID_of_trial) == len(COCO_super_cat)

    catnms += list(COCO_cat[catID_of_trial > 0])
    catnms += list(COCO_super_cat[supcatID_of_trial > 0])
    return catnms


def load_subset_trials(coco_id_by_trial, cat, negcat=False):
    """
    Returns a list of idx to apply on the 10,000 trials for each subject. These are not trials ID themselves but
    indexs for trials IDS.
    """
    subset_idx, negsubset_idx = [], []
    for i, id in enumerate(coco_id_by_trial):
        catnms = load_objects_in_COCO(id)
        if cat in catnms:
            subset_idx.append(i)
        else:
            negsubset_idx.append(i)
    if negcat:
        return negsubset_idx
    else:
        return subset_idx


def find_trial_indexes(
    subj, cat="person", output_dir="/user_data/yuanw3/project_outputs/NSD/output"
):
    coco_id = np.load("%s/coco_ID_of_repeats_subj%02d.npy" % (output_dir, subj))

    idx1, idx2 = [], []
    for i, id in enumerate(coco_id):
        catnms = load_objects_in_COCO(id)
        if cat in catnms:
            idx1.append(i)
        else:
            idx2.append(i)
    return idx1, idx2


def extract_test_image_ids(
    subj=1, output_dir="/user_data/yuanw3/project_outputs/NSD/output"
):
    from sklearn.model_selection import train_test_split

    _, test_idx = train_test_split(range(10000), test_size=0.15, random_state=42)
    coco_id = np.load("%s/coco_ID_of_repeats_subj%02d.npy" % (output_dir, subj))
    test_image_id = coco_id[test_idx]
    return test_image_id, test_idx


def extract_single_roi(roi_name, output_dir, subj):
    from util.model_config import roi_name_dict
    from extract_cortical_voxel import extract_cortical_mask

    output_masks, roi_labels = list(), list()
    try:
        roi_mask = np.load(
            "%s/voxels_masks/subj%01d/roi_1d_mask_subj%02d_%s.npy"
            % (output_dir, subj, subj, roi_name)
        )
    except FileNotFoundError:
        roi_mask = extract_cortical_mask(subj, roi=roi_name, output_dir=output_dir)
        roi_mask = np.load(
            "%s/voxels_masks/subj%01d/roi_1d_mask_subj%02d_%s.npy"
            % (output_dir, subj, subj, roi_name)
        )

    roi_dict = roi_name_dict[roi_name]
    for k, v in roi_dict.items():
        if int(k) > 0:
            if np.sum(roi_mask == int(k)) > 0:
                output_masks.append(roi_mask == int(k))
                roi_labels.append(v)
    return output_masks, roi_labels


def compute_sample_performance(model, subj, output_dir, masking="sig", measure="corrs"):
    """
    Returns sample-wise performances for encoding model.
    """
    if measure == "corrs":
        from scipy.stats import pearsonr

        metric = pearsonr
    elif measure == "rsq":
        from sklearn.metrics import r2_score

        metric = r2_score

    try:
        sample_corrs = np.load(
            "%s/output/clip/%s_sample_%s_%s.npy" % (output_dir, model, measure, masking)
        )
        if len(sample_corrs.shape) == 2:
            sample_corrs = np.array(sample_corrs)[:, 0]
            np.save(
                "%s/output/clip/%s_sample_corrs_%s.npy" % (output_dir, model, masking),
                sample_corrs,
            )
    except FileNotFoundError:
        yhat, ytest = load_model_performance(
            model, output_root=output_dir, measure="pred"
        )
        if masking == "sig":
            pvalues = load_model_performance(
                model, output_root=output_dir, measure="pvalue"
            )
            sig_mask = pvalues <= 0.05

            sample_corrs = [
                metric(ytest[:, sig_mask][i, :], yhat[:, sig_mask][i, :])
                for i in range(ytest.shape[0])
            ]

        else:
            roi = np.load(
                "%s/output/voxels_masks/subj%01d/roi_1d_mask_subj%02d_%s.npy"
                % (output_dir, subj, subj, masking)
            )
            roi_mask = roi > 0
            sample_corrs = [
                metric(ytest[:, roi_mask][i, :], yhat[:, roi_mask][i, :])
                for i in range(ytest.shape[0])
            ]

        if measure == "corr":
            sample_corrs = np.array(sample_corrs)[:, 0]
        np.save(
            "%s/output/clip/%s_sample_%s_%s.npy"
            % (output_dir, model, measure, masking),
            sample_corrs,
        )

    return sample_corrs
