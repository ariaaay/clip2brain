"This script loads feature spaces and prepares it for encoding model"
import numpy as np
from tqdm import tqdm
import pandas as pd


def get_preloaded_features(
    subj,
    stim_list,
    model,
    layer=None,
    features_dir="/user_data/yuanw3/project_outputs/NSD/features",
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
    layer=None,
    features_dir="/user_data/yuanw3/project_outputs/NSD/features",
):
    if "taskrepr" in model:
        # latent space in taskonomy, model should be in the format of "taskrepr_X", e.g. taskrep_curvature
        task = "_".join(model.split("_")[1:])

        if layer is None:
            repr_dir = (
                "/lab_data/tarrlab/yuanw3/taskonomy_features/genStimuli/%s" % task
            )
        else:
            repr_dir = (
                "/lab_data/tarrlab/yuanw3/taskonomy_features/genStimuli_layers/%s"
                % task
            )

        featmat = []
        print("stimulus length is: " + str(len(stim_list)))
        for img_id in tqdm(stim_list):
            if layer is None:
                try:
                    fpath = "%s/%d.npy" % (repr_dir, img_id)
                    repr = np.load(fpath).flatten()
                except FileNotFoundError:
                    fpath = "%s/COCO_train2014_%012d.npy" % (repr_dir, img_id)
                    repr = np.load(fpath).flatten()
            else:
                fpath = "%s/%d_%s.npy" % (repr_dir, img_id, layer)
                repr = np.load(fpath).flatten()
            featmat.append(repr)
        featmat = np.array(featmat)

        if featmat.shape[1] > 6000:
            from sklearn.decomposition import PCA

            pca = PCA(n_components=500)  # TODO:test this dimension later 500d --> 44%
            featmat = pca.fit_transform(featmat.astype(np.float16))
            print("PCA explained variance" + str(np.sum(pca.explained_variance_ratio_)))

    elif "convnet" in model or "resnet50" in model:
        # model should be named "convnet_vgg16" to load "feat_vgg16.npy"
        model_folder, layer = model.split("_")
        print("Loading convnet model %s..." % model)
        # this extracted feature is order based on nsd ID (order in the stimulus info file)
        try:
            all_feat = np.load("%s/%s.npy" % (features_dir, model))
        except FileNotFoundError:
            all_feat = np.load(
                "/lab_data/tarrlab/common/datasets/features/NSD/%s/%s_%s.npy"
                % (model_folder, model_folder, layer)
            )  # on clsuter

        stim = pd.read_pickle(
            "/lab_data/tarrlab/common/datasets/NSD/nsddata/experiments/nsd/nsd_stim_info_merged.pkl"
        )

        featmat = []
        for img_id in tqdm(stim_list):
            try:
                # extract the nsd ID corresponding to the coco ID in the stimulus list
                stim_ind = stim["nsdId"][stim["cocoId"] == img_id]
                # extract the repective features for that nsd ID
                featmat.append(all_feat[stim_ind, :])
            except IndexError:
                print("COCO Id Not Found: " + str(img_id))
        featmat = np.array(featmat).squeeze()

    elif "clip" in model:
        try:
            all_feat = np.load("%s/%s.npy" % (features_dir, model))
        except FileNotFoundError:
            all_feat = np.load(
                "/lab_data/tarrlab/common/datasets/features/NSD/CLIP/%s.npy" % model
            )
        stim = pd.read_pickle(
            "/lab_data/tarrlab/common/datasets/NSD/nsddata/experiments/nsd/nsd_stim_info_merged.pkl"
        )
        featmat = []
        for img_id in tqdm(stim_list):
            try:
                # extract the nsd ID corresponding to the coco ID in the stimulus list
                stim_ind = stim["nsdId"][stim["cocoId"] == img_id]
                # extract the repective features for that nsd ID
                featmat.append(all_feat[stim_ind, :])
            except IndexError:
                print("COCO Id Not Found: " + str(img_id))
        featmat = np.array(featmat).squeeze()

    elif "cat" in model:
        try:
            all_feat = np.load("%s/%s.npy" % (features_dir, model))
        except FileNotFoundError:
            all_feat = np.load(
                "/lab_data/tarrlab/common/datasets/features/NSD/COCO_Cat/%s.npy" % model
            )
        stim = pd.read_pickle(
            "/lab_data/tarrlab/common/datasets/NSD/nsddata/experiments/nsd/nsd_stim_info_merged.pkl"
        )
        featmat = []
        for img_id in tqdm(stim_list):
            try:
                # extract the nsd ID corresponding to the coco ID in the stimulus list
                stim_ind = stim["nsdId"][stim["cocoId"] == img_id]
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
