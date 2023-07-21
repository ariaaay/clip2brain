import argparse
import pickle
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from util.util import ev
from util.model_config import roi_name_dict

# extract each subject's index for 1000 images
def extract_subject_trials_index_shared1000(stim, subj):
    index = list()
    for i in range(3):
        col = "subject%01d_rep%01d" % (subj, i)
        assert len(stim[col][stim["shared1000"]]) == 1000
        index.append(list(stim[col][stim["shared1000"]]))
    assert len(index) == 3
    return index


def compute_ev(subj, roi="", biascorr=False, zscored_input=False):
    l = np.load(
        "%s/trials_subj%02d.npy" % (args.output_dir, subj)
    )  # size should be 10000 by 3 for subj 1,2,5,7; ordered by image id. entries are trial numbers

    repeat_n = l.shape[0]
    # print("The number of images with 3 repetitions are: " + str(repeat_n))

    try:
        assert l.shape == (repeat_n, 3)
    except AssertionError:
        print("Irregular trial shape:")
        print(l.shape)

    if zscored_input:
        data = np.load(
            "%s/cortical_voxels/cortical_voxel_across_sessions_zscored_by_run_subj%02d%s.npy"
            % (args.output_dir, subj, roi)
        )
    else:
        data = np.load(
            "%s/cortical_voxels/cortical_voxel_across_sessions_subj%02d%s.npy"
            % (args.output_dir, subj, roi)
        )

    # data size is # of total trials X # of voxels
    ev_list = []
    avg_mat = np.zeros(
        (repeat_n, data.shape[1])
    )  # size = number of repeated images by number of voxels

    print("Brain data shape is:")
    print(data.shape)
    # import pdb; pdb.set_trace()
    # fill in 0s for nonexisting trials
    if data.shape[0] < 30000:
        tmp = np.zeros((30000, data.shape[1]))
        tmp[:] = np.nan
        tmp[: data.shape[0], :] = data.copy()
        data = tmp

    for v in tqdm(range(data.shape[1])):  # loop over voxels
        repeat = list()
        for r in range(3):
            try:
                repeat.append(data[l[:, r], v])  # all repeated trials for each voxels
            except IndexError:
                print("Index Error")
                print(r, v)

        # repeat size: 3 x # repeated images
        repeat = np.array(repeat).T
        try:
            assert repeat.shape == (repeat_n, 3)
            avg_mat[:, v] = np.nanmean(repeat, axis=1)
            print("NaNs:")
            print(np.sum(np.isnan(avg_mat[:, v])))
        except AssertionError:
            print(repeat.shape)

        ev_list.append(ev(repeat, biascorr=biascorr))
    np.save(
        "%s/cortical_voxels/averaged_cortical_responses_zscored_by_run_subj%02d%s.npy"
        % (args.output_dir, subj, roi),
        avg_mat,
    )
    return np.array(ev_list)


def compute_sample_wise_ev(
    subj,
    mask,
    biascorr=False,
    zscored_input=False,
    output_dir="output",
):
    l = np.load(
        "%s/trials_subj%02d.npy" % (output_dir, subj)
    )  # size should be 10000 by 3 for subj 1,2,5,7; ordered by image id

    repeat_n = l.shape[0]
    print("The number of images with 3 repetitions are: " + str(repeat_n))

    try:
        assert l.shape == (repeat_n, 3)
    except AssertionError:
        print(l.shape)

    if zscored_input:
        data = np.load(
            "%s/cortical_voxels/cortical_voxel_across_sessions_zscored_by_run_subj%02d.npy"
            % (output_dir, subj)
        )
    else:
        data = np.load(
            "%s/cortical_voxels/cortical_voxel_across_sessions_subj%02d.npy"
            % (output_dir, subj)
        )

    # index by roi
    data = data[:, mask]
    print("Brain data shape is:")
    print(data.shape)

    ev_list = []
    for i in tqdm(range(l.shape[0])):  # loop over images
        repeat = data[l[i, :], :].T  # all repeated trials for each voxels
        assert repeat.shape == (data.shape[1], 3)
        ev_list.append(ev(repeat, biascorr=biascorr))

    return ev_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--roi_only", action="store_true")
    parser.add_argument("--subj", type=int, default=1)
    parser.add_argument("--biascorr", action="store_true")
    parser.add_argument("--zscored_input", default=True, action="store_true")
    parser.add_argument("--compute_ev", action="store_true", default=False)
    parser.add_argument("--compute_sample_ev", action="store_true")
    parser.add_argument("--roi_for_sample_ev", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="output")

    args = parser.parse_args()

    tag = ""

    if args.roi_only:
        roi = "_roi_only"
        tag += roi
    else:
        roi = ""

    if args.biascorr:
        tag += "_biascorr"

    if args.zscored_input:
        tag += "_zscored"

    if args.compute_ev:
        print("computing EVs")
        all_evs = compute_ev(args.subj, roi, args.biascorr, args.zscored_input)
        np.save("%s/evs_subj%02d%s.npy" % (args.output_dir, args.subj, tag), all_evs)

        plt.figure()
        plt.hist(all_evs)
        plt.title("Explainable Variance across Voxels (subj%02d%s)" % (args.subj, tag))
        plt.savefig("figures/evs_subj%02d%s.png" % (args.subj, tag))

    elif args.compute_sample_ev:
        sample_ev_by_roi = {}
        roi = args.roi_for_sample_ev
        if roi is not None:
            roi_mask = np.load(
                "%s/voxels_masks/subj%01d/roi_1d_mask_subj%02d_%s.npy"
                % (args.output_dir, args.subj, args.subj, roi)
            )
            roi_dict = roi_name_dict[roi]
            for k, v in roi_dict.items():
                if k > 0:
                    mask = roi_mask == k
                    sample_ev = compute_sample_wise_ev(
                        args.subj,
                        mask,
                        args.biascorr,
                        args.zscored_input,
                        output_dir=args.output_dir,
                    )
                    sample_ev_by_roi[v] = sample_ev
            json.dump(
                sample_ev_by_roi,
                open(
                    "%s/sample_snr/sample_snr_subj%02d_%s.json"
                    % (args.output_dir, args.subj, roi),
                    "w",
                ),
            )

        else:
            sample_evs = compute_sample_wise_ev(
                args.subj,
                roi,
                args.biascorr,
                args.zscored_input,
                output_dir=args.output_dir,
            )
            np.save(
                "%s/sample_snr/sample_snr_subj%02d_%s.npy"
                % (args.output_dir, args.subj, roi),
                sample_evs,
            )
