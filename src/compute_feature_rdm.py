import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform


from util.model_config import *


def computeRSM(model, feature_dir, subj=1):
    feature_path = "%s/subj%d/%s.npy" % (feature_dir, subj, model)
    feature = np.load(feature_path).squeeze()
    rsm = np.corrcoef(feature)
    return rsm


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--subj", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--feature_dir", default="features")
    parser.add_argument("--feature", type=str)
    parser.add_argument("--ID", action="store_true")
    parser.add_argument("--RDM", action="store_true")
    args = parser.parse_args()

    if args.RSM:
        rsm = computeRSM(args.feature, args.feature_dir, args.subj)
        np.save(
            "%s/rdms/subj%02d_%s.npy" % (args.output_dir, args.subj, args.feature),
            rsm,
        )
