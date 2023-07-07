import os
import json
import argparse
import numpy as np
from tqdm import tqdm

from util.model_config import *

from IDNN.intrinsic_dimension import estimate, block_analysis
from scipy.spatial.distance import pdist, squareform


def computeID(r, nres=20, fraction=0.9, method="euclidean", verbose=False):
    ID = []
    n = int(np.round(r.shape[0] * fraction))
    dist = squareform(pdist(r, method))
    for i in tqdm(range(nres), leave=False):
        dist_s = dist
        perm = np.random.permutation(dist.shape[0])[0:n]
        dist_s = dist_s[perm, :]
        dist_s = dist_s[:, perm]
        ID.append(estimate(dist_s, verbose=verbose)[2])
    mean = np.mean(ID)
    error = np.std(ID)
    return mean, error


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
            "%s/rdms/subj%02d_%s.npy" % (args.output_dir, args.subj, args.feature), rsm,
        )

    if args.ID:
        subsample_idx = np.random.choice(np.arange(10000), size=1000, replace=False)
        ID_dict = {}
        feat_names = os.listdir("%s/subj%d" % (args.feature_dir, args.subj))
        for fname in tqdm(feat_names):
            if ("avgpool" in fname) or ("layer" in fname):
                print(fname)
                feature = np.load("%s/subj%d/%s" % (args.feature_dir, args.subj, fname))
                # if feature.shape[-1] < 6000:
                r = feature.squeeze()[subsample_idx, :]
                mean, error = computeID(feature.squeeze())
                print(fname)
                print("Mean ID is: %d Error of ID is: %d" % (mean, error))
                ID_dict[fname] = (mean, error)
        with open("../Cats/outputs/ID_dict.json", "w") as f:
            json.dump(ID_dict, f)
