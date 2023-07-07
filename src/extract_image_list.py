"""
extract_image_list.py extracts coco ID
"""

import pickle
import numpy as np
import pandas as pd
import argparse


def extract_repeat_img_list(stim, subj):
    """
    :param stim: stimulus info file
    :param subj: subject number
    :return: the COCO IDs of images that are shown to a subject.
    """
    # using the first rep b/c three repetitions should have the same order of IDs.
    col_name = "subject%d_rep0" % subj
    image_id_list = list(stim.cocoId[stim[col_name] != 0])
    return np.array(image_id_list)


def extract_repeat_trials_list(stim, subj):
    """
    :param stim: stimulus info file
    :param subj: subject number
    :return: a n-by-3 matrix where n is the number of unique images this subject sees. # size should be 10000 by 3 for
    subj 1,2,5,7; the matrix rows are ordered by image id. The first column is the trial indexes of the first
    repetition of the images; the second columns are the trials index for the second repetitions, etc. This is supposed
     to be used with the coco ID output from extract_repeat_img_list().
    """
    all_rep_trials_list = list()
    for rep in range(3):
        col_name = "subject%d_rep%01d" % (subj, rep)
        trial_id_list = list(stim[col_name][stim[col_name] != 0])
        all_rep_trials_list.append(trial_id_list)
    all_rep_trials_list = (
        np.array(all_rep_trials_list).T - 1
    )  # change from 1 based to 0 based
    return all_rep_trials_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subj", type=int)
    parser.add_argument("--type", type=str)
    parser.add_argument("--output_dir", type=str, default=".")
    # parser.add_argument("--rep", type=int, default="0", help="Choose which repeats (0-2)")

    args = parser.parse_args()

    stim = pd.read_pickle(
        "/lab_data/tarrlab/common/datasets/NSD/nsddata/experiments/nsd/nsd_stim_info_merged.pkl"
    )

    if args.type == "cocoId":
        image_list = extract_repeat_img_list(stim, args.subj)
        np.save(
            "%s/output/coco_ID_of_repeats_subj%02d.npy" % (args.output_dir, args.subj),
            image_list,
        )

    elif args.type == "trial":
        trial_list = extract_repeat_trials_list(stim, args.subj)
        np.save(
            "%s/output/trials_subj%02d.npy" % (args.output_dir, args.subj), trial_list
        )

    else:
        raise TypeError("please correct return type to cocoId or trial")
