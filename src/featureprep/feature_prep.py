"This script loads feature spaces and prepares it for encoding model"
import numpy as np
from tqdm import tqdm
import pandas as pd

import configparser

config = configparser.ConfigParser()
config.read("config.cfg")
stim_path = config["DATA"]["StimuliInfo"]
STIM = pd.read_pickle(stim_path)


def get_preloaded_features(
    subj,
    stim_list,
    model,
    layer=None,
    features_dir="features",
):
    """
    :param subj: subject ID
    :param stim_list: a list of COCO IDs for the stimuli images
    :param model: models to extract features from
    :param cache: to load/save a cache of the feature matrix or not
    :return featmat: a matrix of features that matches with the order of brain data
    """
    subj = int(subj)
    if layer is None:
        layer_modifier = ""
    else:
        layer_modifier = "_" + layer

    print("Getting features for %s%s, for subject %d" % (model, layer_modifier, subj))

    try:
        if subj == 0:
            featmat = np.load("%s/%s%s.npy" % (features_dir, model, layer_modifier))
        else:
            featmat = np.load(
                "%s/subj%d/%s%s.npy" % (features_dir, subj, model, layer_modifier)
            )

    except FileNotFoundError:
        featmat = extract_feature_by_imgs(stim_list, model, layer=layer)
        if subj == 0:  # meaning it is features for all subjects
            np.save("%s/%s%s.npy" % (features_dir, model, layer_modifier), featmat)
        else:
            np.save(
                "%s/subj%d/%s%s.npy" % (features_dir, subj, model, layer_modifier),
                featmat,
            )

    print("feature shape is " + str(featmat.shape[0]))
    return featmat.squeeze()


def extract_feature_by_imgs(
    stim_list,
    model,
    features_dir="features",
):

    if "resnet50" in model:
        print("Loading convnet model %s..." % model)
        # this extracted feature is order based on nsd ID (order in the stimulus info file)
        all_feat = np.load("%s/%s.npy" % (features_dir, model))

        featmat = []
        for img_id in tqdm(stim_list):
            try:
                # extract the nsd ID corresponding to the coco ID in the stimulus list
                stim_ind = STIM["nsdId"][STIM["cocoId"] == img_id]
                # extract the repective features for that nsd ID
                featmat.append(all_feat[stim_ind, :])
            except IndexError:
                print("COCO Id Not Found: " + str(img_id))
        featmat = np.array(featmat).squeeze()

    elif "clip" in model:
        all_feat = np.load("%s/%s.npy" % (features_dir, model))
        featmat = []
        for img_id in tqdm(stim_list):
            try:
                # extract the nsd ID corresponding to the coco ID in the stimulus list
                stim_ind = STIM["nsdId"][STIM["cocoId"] == img_id]
                # extract the repective features for that nsd ID
                featmat.append(all_feat[stim_ind, :])
            except IndexError:
                print("COCO Id Not Found: " + str(img_id))
        featmat = np.array(featmat).squeeze()

    elif "cat" in model:
        all_feat = np.load("%s/%s.npy" % (features_dir, model))
        featmat = []
        for img_id in tqdm(stim_list):
            try:
                # extract the nsd ID corresponding to the coco ID in the stimulus list
                stim_ind = STIM["nsdId"][STIM["cocoId"] == img_id]
                # extract the repective features for that nsd ID
                featmat.append(all_feat[stim_ind, :])
            except IndexError:
                print("COCO Id Not Found: " + str(img_id))
        featmat = np.array(featmat).squeeze()

    else:
        try:
            all_feat = np.load("%s/%s.npy" % (features_dir, model))
        except FileNotFoundError:
            print("ERROR: Feature spaces unknown...")

    return featmat.squeeze()


def extract_feature_with_image_order(stim_list, feature_matrix, image_order):
    try:
        assert len(image_order) == feature_matrix.shape[0]
    except AssertionError:
        print(
            "Image order should have the same length as the 1st dimension of the feature matrix."
        )
    image_order = list(image_order)
    idxes = [image_order.index(cid) for cid in stim_list]
    # extract the respective features for that coco ID
    featmat = feature_matrix[np.array(idxes), :]
    return featmat.squeeze()
