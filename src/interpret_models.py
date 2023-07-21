import os
import argparse

import pickle

import pandas as pd
import seaborn as sns
import nibabel as nib
import numpy as np

# from shinyutils.matwrap import sns, plt, MatWrap as mw
import matplotlib.pyplot as plt
import skimage.io as io

from tqdm import tqdm
from PIL import Image

from sklearn.decomposition import PCA

# import torch
import clip

from util.util import fdr_correct_p
from util.data_util import (
    load_model_performance,
    extract_test_image_ids,
    fill_in_nan_voxels,
)
from util.model_config import *

# device = "cuda" if torch.cuda.is_available() else "cpu"

from pycocotools.coco import COCO


def extract_text_activations(model, word_lists):
    activations = []
    for word in word_lists:
        text = clip.tokenize([word]).to(device)
        with torch.no_grad():
            activations.append(model.encode_text(text).data.numpy())
    return np.array(activations)


def extract_keywords_for_roi(
    w, activations, common_words, roi_name=None, roi_vals=None, mask=None
):
    if mask is None:
        roi_mask = np.load(
            "%s/output/voxels_masks/subj%01d/roi_1d_mask_subj%02d_%s.npy"
            % (args.output_root, args.subj, args.subj, roi_name)
        )
        roi_selected_vox = np.zeros((roi_mask.shape))
        for v in roi_vals:
            roi_selected_vox += roi_mask == v
        roi_selected_vox = roi_selected_vox > 0
    else:
        roi_selected_vox = mask

    roi_w = w[:, roi_selected_vox]

    scores = np.mean(activations.squeeze() @ roi_w, axis=1)
    print(roi_name)
    best_list = list(np.array(common_words)[np.argsort(scores)[::-1][:30]])
    worst_list = list(np.array(common_words)[np.argsort(scores)[:30]])
    print(best_list)
    print(worst_list)
    pickle.dump(
        best_list,
        open(
            "%s/output/clip/roi_maximization/best_in_%s.json"
            % (args.output_root, roi_name),
            "wb",
        ),
    )
    pickle.dump(
        worst_list,
        open(
            "%s/output/clip/roi_maximization/worst_in_%s.json"
            % (args.output_root, roi_name),
            "wb",
        ),
    )


def extract_captions_for_voxel(roi=None, mask=None, n=3):
    """
    voxel that are selected by the mask will be assigned integer values
    """
    save_path = "%s/output/clip/roi_maximization" % args.output_root

    try:
        activations = np.load(
            "%s/output/clip/roi_maximization/subj1_caption_activation.npy"
            % args.output_root
        )
        all_captions = pickle.load(
            open(
                "%s/output/clip/roi_maximization/all_captions_subj1.pkl"
                % args.output_root,
                "rb",
            )
        )
    except FileNotFoundError:
        import clip
        import torch
        from util.coco_utils import load_captions

        device = "cuda" if torch.cuda.is_available() else "cpu"

        all_captions = []
        all_coco_ids = np.load(
            "%s/output/coco_ID_of_repeats_subj%02d.npy" % (args.output_root, args.subj)
        )
        model, _ = clip.load("ViT-B/32", device=device)
        activations = []
        for cid in tqdm(all_coco_ids):
            with torch.no_grad():
                captions = load_captions(cid)
                all_captions += captions
                for caption in captions:
                    text = clip.tokenize(caption).to(device)
                    activations.append(model.encode_text(text).cpu().data.numpy())

        np.save(
            "%s/output/clip/roi_maximization/subj1_caption_activation.npy"
            % args.output_root,
            activations,
        )
        pickle.dump(
            all_captions,
            open(
                "%s/output/clip/roi_maximization/all_captions_subj1.pkl"
                % args.output_root,
                "wb",
            ),
        )

    activations = np.array(activations)
    print(activations.shape)

    w = np.load(
        "%s/output/encoding_results/subj%d/weights_clip_whole_brain.npy"
        % (args.output_root, args.subj)
    )

    best_caption_dict = dict()

    if mask is None:
        if roi is not None:
            roi_mask = np.load(
                "%s/output/voxels_masks/subj%01d/roi_1d_mask_subj%02d_%s.npy"
                % (args.output_root, args.subj, args.subj, roi)
            )

            try:  # take out zero voxels
                non_zero_mask = np.load(
                    "%s/output/voxels_masks/subj%d/nonzero_voxels_subj%02d.npy"
                    % (args.output_root, args.subj, args.subj)
                )
                print("Masking zero voxels...")
                roi_mask = roi_mask[non_zero_mask]
            except FileNotFoundError:
                pass

            if args.roi_value != 0:
                mask = roi_mask == args.roi_value
            else:
                mask = roi_mask > 0
        else:
            raise ValueError("Mask or roi cannot both be None.")

    print(str(sum(mask)) + " voxels for optimization")
    vox_w = w[:, mask]

    vindx = np.arange(sum(mask))

    scores = activations.squeeze() @ vox_w  # of captions x # voxels

    for i in vindx:
        best_caption_dict[str(i)] = list(
            np.array(all_captions)[np.argsort(scores[:, i])[::-1][:n]]
        )

    import json

    with open("%s/max_caption_per_voxel_in_%s.json" % (save_path, roi), "w") as f:
        json.dump(best_caption_dict, f)
    return best_caption_dict


def extract_emb_keywords(embedding, activations, common_words, n=15):
    scores = activations.squeeze() @ embedding
    if len(embedding.shape) > 1:
        scores = np.mean(scores, axis=1)

    best_list = list(np.array(common_words)[np.argsort(scores)[::-1][:n]])
    worst_list = list(np.array(common_words)[np.argsort(scores)[:n]])
    best_list_word_only = [w.split(" ")[-1] for w in best_list]
    worst_list_word_only = [w.split(" ")[-1] for w in worst_list]
    return best_list_word_only, worst_list_word_only


def compute_average_pairwise_brain_corr(subj=1, sample_n=5000):
    print("computing average corr")
    from scipy.stats import pearsonr

    bdata = np.load(
        "%s/output/cortical_voxels/averaged_cortical_responses_zscored_by_run_subj%02d.npy"
        % (args.output_root, subj)
    )
    try:
        non_zero_mask = np.load(
            "%s/output/voxels_masks/subj%d/nonzero_voxels_subj%02d.npy"
            % (args.output_root, subj, subj)
        )
        print("Masking zero voxels...")
        bdata = bdata[:, non_zero_mask]
    except FileNotFoundError:
        pass

    select_ind = np.random.choice(bdata.shape[0], size=sample_n * 2, replace=True)

    b1 = bdata[select_ind[:sample_n], :]
    b2 = bdata[select_ind[sample_n:], :]
    b_corr = np.array([pearsonr(b1[:, v], b2[:, v])[0] for v in range(b1.shape[1])])
    print(b_corr.shape)
    np.save(
        "%s/output/rdm_based_analysis/subj%d/voxel_corr_baseline_n%d.npy"
        % (args.output_root, subj, sample_n),
        b_corr,
    )


def plot_max_diff_images(
    rsm1,
    rsm2,
    cocoId_subj,
    model_name_for_fig,
    print_caption=False,
    print_distance=False,
):
    from util.coco_utils import get_coco_image, get_coco_caps

    diff1 = rsm1 - rsm2  # close in 1, far in 2
    # diff2 = rsm2 - rsm1
    ind_1 = np.unravel_index(np.argsort(diff1, axis=None), diff1.shape)
    # ind_2 = np.unravel_index(np.argsort(diff2, axis=None), diff2.shape)
    # b/c symmetry of RDM, every two pairs are the same
    trial_id_pair_1 = [(ind_1[0][::-1][i], ind_1[1][::-1][i]) for i in range(0, 30, 2)]
    # trial_id_pair_2 = [(ind_2[0][::-1][i], ind_2[1][::-1][i]) for i in range(0, 30, 2)]
    trial_id_pair_2 = [(ind_1[0][i], ind_1[1][i]) for i in range(0, 30, 2)]

    plt.figure(figsize=(20, 30))
    for m, trial_id_pair in enumerate([trial_id_pair_1, trial_id_pair_2]):
        for i in range(10):
            if trial_id_pair[i][0] == trial_id_pair[i][1]:  # a pair of the same image
                continue

            plt.subplot(10, 2, i * 2 + 1)
            id = cocoId_subj[trial_id_pair[i][0]]
            I = get_coco_image(id, coco_train, coco_val)
            t = str(id)
            plt.imshow(I)
            if print_distance:
                t += "Diff:%.2f; Sim1:%.2f; Sim2:%.2f" % (
                    diff1[trial_id_pair[i]],
                    rsm1[trial_id_pair[i]],
                    rsm2[trial_id_pair[i]],
                )
            if print_caption:
                caption = get_coco_caps(id, coco_train_caps, coco_val_caps)
                try:
                    caption = caption[0]
                except IndexError:
                    print("cant find caption for " + str(id))
                    caption = " "
                t += caption
            plt.title(t)
            plt.axis("off")

            plt.subplot(10, 2, i * 2 + 2)
            id = cocoId_subj[trial_id_pair[i][1]]
            I = get_coco_image(id, coco_train, coco_val)
            plt.imshow(I)
            if print_caption:
                # import pdb; pdb.set_trace()
                caption = get_coco_caps(id, coco_train_caps, coco_val_caps)
                try:
                    caption = caption[0]
                except IndexError:
                    print("cant find caption for " + str(id))
                    caption = " "
                plt.title(str(id) + caption)
            plt.axis("off")

        plt.tight_layout()
        plt.savefig(
            "figures/CLIP/RDM_max/RDM_max_images_close_in_%s_far_in_%s.png"
            % model_name_for_fig[m]
        )


def sample_level_semantic_analysis(
    subj=1,
    model1="clip",
    model2="resnet50_bottleneck",
    print_caption=False,
    print_distance=False,
    threshold=None,
    normalize="sub_baseline",
):
    """
    Find images that have maximum representational distances in models space.
    """

    from util.coco_utils import get_coco_image, get_coco_caps
    from sklearn.model_selection import train_test_split

    if not os.path.exists(
        "%s/output/rdm_based_analysis/subj%d/" % (args.output_root, subj)
    ):
        os.makedirs("%s/output/rdm_based_analysis/subj%d/" % (args.output_root, subj))

    # load coco id in presentation to this subject
    # load model prediction accuracy
    # load brain response to compute voxel similarity
    # load caption, and bert(caption)

    cocoId_subj = np.load(
        "%s/output/coco_ID_of_repeats_subj%02d.npy" % (args.output_root, subj)
    )
    try:
        rsm1 = np.load(
            "%s/output/rdms/subj%02d_%s.npy" % (args.output_root, subj, model1)
        )
    except FileNotFoundError:
        from compute_feature_rdm import computeRSM

        rsm1 = computeRSM(model1, args.feature_dir)
        np.save(
            "%s/output/rdms/subj%02d_%s.npy" % (args.output_root, subj, model1),
            rsm1,
        )

    try:
        rsm2 = np.load(
            "%s/output/rdms/subj%02d_%s.npy" % (args.output_root, subj, model2)
        )
    except FileNotFoundError:
        from compute_feature_rdm import computeRSM

        rsm2 = computeRSM(model2, args.feature_dir)
        np.save(
            "%s/output/rdms/subj%02d_%s.npy" % (args.output_root, subj, model2),
            rsm2,
        )

    # plot max diff images
    model_name_for_fig = [(model1, model2), (model2, model1)]
    plot_max_diff_images(
        rsm1,
        rsm2,
        cocoId_subj,
        model_name_for_fig,
        print_caption=print_caption,
        print_distance=print_distance,
    )

    if threshold is not None:
        triu_flag = np.triu(np.ones(rsm1.shape), 1).astype(bool)
        perc = 97.5
        if threshold[2] == "br":
            # import pdb; pdb.set_trace()
            t1 = np.percentile(rsm1[triu_flag], perc)
            t2 = np.percentile(rsm2[triu_flag], (100 - perc))
            select_pair = (rsm1 > t1) * (rsm2 < t2)
            print(t1, t2, threshold[2])
        elif threshold[2] == "tl":
            t1 = np.percentile(rsm1[triu_flag], (100 - perc))
            t2 = np.percentile(rsm2[triu_flag], perc)
            print(t1, t2, threshold[2])
            select_pair = (rsm1 < t1) * (rsm2 > t2)

        num_pairs = np.sum(select_pair)
        print("number of pair of images selected: " + str(num_pairs / 2))
        ind = np.unravel_index(np.argsort(select_pair, axis=None), select_pair.shape)
        # b/c symmetry of RDM, every two pairs are the same
        trial_id_pair = [
            [ind[0][::-1][i], ind[1][::-1][i]] for i in range(0, num_pairs, 2)
        ]
        trial_id_pair = np.array(trial_id_pair)

        print(trial_id_pair.shape)

        # compare brain image
        print("Computing brain correlation...")
        from scipy.stats import pearsonr

        bdata = np.load(
            "%s/output/cortical_voxels/averaged_cortical_responses_zscored_by_run_subj%02d.npy"
            % (args.output_root, subj)
        )

        try:
            non_zero_mask = np.load(
                "%s/output/voxels_masks/subj%d/nonzero_voxels_subj%02d.npy"
                % (args.output_root, subj, subj)
            )
            print("Masking zero voxels...")
            bdata = bdata[:, non_zero_mask]
        except FileNotFoundError:
            pass

        print("NaNs? Finite?:")
        print(np.any(np.isnan(bdata)))
        print(np.all(np.isfinite(bdata)))
        print("Brain response size is: " + str(bdata.shape))

        b1 = bdata[trial_id_pair[:, 0], :]
        b2 = bdata[trial_id_pair[:, 1], :]
        b_corr = np.array([pearsonr(b1[:, v], b2[:, v])[0] for v in range(b1.shape[1])])
        print(b_corr.shape)
        normalize_flag = ""
        if normalize == "sub_baseline":
            baseline = np.load(
                "%s/output/rdm_based_analysis/subj%d/voxel_corr_baseline_n%d.npy"
                % (args.output_root, subj, 5000)
            )
            b_corr = b_corr - baseline
            normalize_flag = "_sub_baseline"

        np.save(
            "%s/output/rdm_based_analysis/subj%d/voxel_corr_%s_vs_%s_%.2f_%.2f_%s%s.npy"
            % (
                args.output_root,
                subj,
                model1,
                model2,
                t1,
                t2,
                threshold[2],
                normalize_flag,
            ),
            b_corr,
        )

        ## visualize the image
        print("visualizing images...")
        plt.figure(figsize=(20, 30))
        for i, pair in enumerate(trial_id_pair[:30]):
            for j in range(2):
                plt.subplot(30, 2, i * 2 + j + 1)
                id = cocoId_subj[pair[j]]
                I = get_coco_image(id, coco_train, coco_val)
                C = get_coco_caps(id, coco_train_caps, coco_val_caps)
                try:
                    cap = C[0]
                except IndexError:
                    print(C)
                    cap = " "
                plt.imshow(I)
                plt.title(cap)
                plt.axis("off")
        plt.tight_layout()

        if not os.path.exists("figures/CLIP/RDM_based_analysis/subj%d/" % subj):
            os.makedirs("figures/CLIP/RDM_based_analysis/subj%d/" % subj)

        plt.savefig(
            "figures/CLIP/RDM_based_analysis/subj%d/%s_vs_%s_%.2f_%.2f_%s.png"
            % (subj, model1, model2, t1, t2, threshold[2])
        )


def compare_model_and_brain_performance_on_COCO():
    """
    Analyze patterns in CLIP performances on COCO (in terms of matching captions) versus
    CLIP encoding models performances sample-wise.
    """
    import torch
    from scipy.stats import pearsonr
    from utils.coco_utils import load_captions
    from utils.data_util import compute_sample_performance

    config = configparser.ConfigParser()
    config.read("config.cfg")
    stimuli_dir = config["DATA"]["StimuliDir"]

    corrs_v, corrs_t = [], []
    for subj in np.arange(1, 9):
        test_image_id, _ = extract_test_image_ids(subj)
        all_images_paths = list()
        all_images_paths += ["%s/%s.jpg" % (stimuli_dir, id) for id in test_image_id]

        print("Number of Images: {}".format(len(all_images_paths)))

        captions = [
            load_captions(cid)[0] for cid in test_image_id
        ]  # pick the first caption
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)

        preds = list()
        for i, p in enumerate(all_images_paths):
            image = preprocess(Image.open(p)).unsqueeze(0).to(device)
            text = clip.tokenize(captions).to(device)
            with torch.no_grad():
                logits_per_image, logits_per_text = model(image, text)
                # print(logits_per_image.shape)
                probs = logits_per_image.squeeze().softmax(dim=-1).cpu().numpy()
                # print(probs.shape)
                preds.append(probs[i])

        sample_corr_clip = compute_sample_performance("clip", i, args.output_root)
        sample_corr_clip_text = compute_sample_performance(
            "clip_text", i, args.output_root
        )

        corrs_v.append(pearsonr(sample_corr_clip, preds)[0])
        corrs_t.append(pearsonr(sample_corr_clip_text, preds)[0])

    fig = plt.figure()
    plt.plot(corrs_v, color="red", label="clip visual")
    plt.plot(corrs_t, color="blue", label="clip text")
    plt.legend()

    plt.savefig("figures/CLIP/model_brain_comparison.png")


def coarse_level_semantic_analysis(subj=1):
    """
    RDM analysis on model representations based on COCO super categories
    """

    from sklearn.metrics.pairwise import cosine_similarity

    image_supercat = np.load("data/NSD_supcat_feat.npy")
    # image_cat = np.load("data/NSD_cat_feat.npy")
    cocoId_subj = np.load(
        "%s/output/coco_ID_of_repeats_subj%02d.npy" % (args.output_root, subj)
    )
    nsd2coco = np.load("%s/output/NSD2cocoId.npy" % args.output_root)
    img_ind = [list(nsd2coco).index(i) for i in cocoId_subj]
    image_supercat_subsample = image_supercat[img_ind, :]
    max_cat = np.argmax(image_supercat_subsample, axis=1)
    max_cat_order = np.argsort(max_cat)
    sorted_image_supercat = image_supercat_subsample[max_cat_order, :]
    sorted_image_supercat_sim_by_image = cosine_similarity(sorted_image_supercat)
    # image_cat_subsample = image_cat[img_ind,:]
    # sorted_image_cat = image_cat_subsample[np.argsort(max_cat),:]
    models = [
        "clip",
        "clip_text",
        "convnet_res50",
        "bert_layer_13",
        "clip_visual_resnet",
    ]
    plt.figure(figsize=(30, 30))
    plt.subplot(2, 3, 1)
    plt.imshow(sorted_image_supercat_sim_by_image)
    plt.colorbar()
    for i, m in enumerate(models):
        plt.subplot(2, 3, i + 2)
        rdm = np.load("%s/output/rdms/subj%02d_%s.npy" % (args.output_root, subj, m))
        plt.imshow(rdm[max_cat_order, :][:, max_cat_order])
        r = np.corrcoef(rdm.flatten(), sorted_image_supercat_sim_by_image.flatten())[
            1, 1
        ]
        plt.title("%s (r=%.2g)" % (m, r))
        plt.colorbar()
    plt.tight_layout()
    plt.savefig("figures/CLIP/coarse_category_RDM_comparison.png")


def image_level_scatter_plot(
    model1="clip",
    model2="resnet50_bottleneck",
    subj=1,
    i=1,
    subplot=False,
    do_zscore=False,
    # threshold=(0, 0, 0),
):
    """
    Compare model presentations across pairs of sampled images.
    """

    from compute_feature_rdm import computeRSM
    from scipy.stats import pearsonr
    from util.util import zscore

    # cocoId_subj = np.load(
    #     "%s/output/coco_ID_of_repeats_subj%02d.npy" % (args.output_root, subj)
    # )
    try:
        rsm1 = np.load(
            "%s/output/rdms/subj%02d_%s.npy" % (args.output_root, subj, model1)
        )
    except FileNotFoundError:
        rsm1 = computeRSM(model1, args.feature_dir)
        np.save(
            "%s/output/rdms/subj%02d_%s.npy" % (args.output_root, subj, model1),
            rsm1,
        )

    try:
        rsm2 = np.load(
            "%s/output/rdms/subj%02d_%s.npy" % (args.output_root, subj, model2)
        )
    except FileNotFoundError:
        rsm2 = computeRSM(model2, args.feature_dir)
        np.save(
            "%s/output/rdms/subj%02d_%s.npy" % (args.output_root, subj, model2),
            rsm2,
        )

    tmp = np.ones(rsm1.shape)
    triu_flag = np.triu(tmp, k=1).astype(bool)
    # subsample 1000 point for plotting
    sampling_idx = np.random.choice(len(rsm1[triu_flag]), size=10000, replace=False)
    # sampling_idx = np.arange(rsm1.shape[0])

    if do_zscore:
        x = zscore(rsm1[triu_flag][sampling_idx])
        y = zscore(rsm2[triu_flag][sampling_idx])
    else:
        x = rsm1[triu_flag][sampling_idx]
        y = rsm2[triu_flag][sampling_idx]
    if subplot:
        plt.subplot(2, 3, i)

    plt.scatter(x, y, alpha=0.2, s=2, label=model2)
    plt.box(False)

    perc = 95
    corner = "br"
    t1 = np.percentile(rsm1[triu_flag], perc)
    t2 = np.percentile(rsm2[triu_flag], (100 - perc))
    br_select = (x > t1) * (y < t2)
    print(t1, t2, corner)

    corner = "tl"
    t1 = np.percentile(rsm1[triu_flag], (100 - perc))
    t2 = np.percentile(rsm2[triu_flag], perc)
    print(t1, t2, corner)
    tl_select = (x < t1) * (y > t2)
    select = br_select + tl_select

    print(np.sum(select))
    plt.scatter(x[select], y[select], c="r", s=2)

    b, a = np.polyfit(x, y, deg=1)
    xseq = np.linspace(min(x), np.max(x), num=100)
    plt.plot(xseq, a + b * xseq, lw=1, color="k", label=model2 + "_fit")

    # r = pearsonr(rsm1[triu_flag][sampling_idx], rsm2[triu_flag][sampling_idx])
    plt.xlabel(model1)
    plt.ylabel(model2)
    # plt.xlim(0, 1)
    # plt.ylim(-0.25, 1)
    # ax = plt.gca()
    # ax.spines['left'].set_position('center')
    # ax.spines['bottom'].set_position('center')
    # plt.axis("off")
    # plt.legend()
    plt.savefig(
        "figures/CLIP/manifold_distance/subj%d/%s_vs_%s.png" % (subj, model1, model2),
        dpi=400,
    )


def category_based_similarity_analysis(model, threshold, subj=1):
    """
    Compute similarity of model representations based on samples containing one category of objects.
    In this implementation, samples all contained human.
    """

    from compute_feature_rdm import computeRSM
    from scipy.stats import pearsonr
    from util.util import zscore

    try:
        rsm = np.load(
            "%s/output/rdms/subj%02d_%s.npy" % (args.output_root, subj, model)
        )
    except FileNotFoundError:
        rsm = computeRSM(model1, args.feature_dir)
        np.save(
            "%s/output/rdms/subj%02d_%s.npy" % (args.output_root, subj, model),
            rsm,
        )

    tmp = np.ones(rsm.shape)
    triu_flag = np.triu(tmp, k=1).astype(bool)

    from featureprep.feature_prep import get_preloaded_features

    stimulus_list = np.load(
        "%s/output/coco_ID_of_repeats_subj%02d.npy" % (args.output_root, 1)
    )
    COCO_cat_feat = get_preloaded_features(
        1,
        stimulus_list,
        "cat",
        features_dir="%s/features" % args.output_root,
    )
    print(COCO_cat_feat.shape)
    person_flag = COCO_cat_feat[:, 0] > threshold
    person_n = np.sum(person_flag)
    cluster_flag = np.outer(person_flag, person_flag).astype(bool)
    within_flag = cluster_flag * triu_flag
    cross_flag = ~cluster_flag * triu_flag

    within_cluster_score = np.mean(rsm[within_flag])
    cross_cluster_score = np.mean(rsm[cross_flag])

    # print(within_cluster_score)
    # print(cross_cluster_score)

    return within_cluster_score / (cross_cluster_score + within_cluster_score), person_n


def max_img4vox(weight, subj, model, save_name):
    from util.coco_utils import (
        get_coco_anns,
        get_coco_image,
        get_coco_caps,
    )

    from featureprep.feature_prep import get_preloaded_features

    subj = 1  # using images that subj 1 saw as a random "sample"

    stimulus_list = np.load(
        "%s/output/coco_ID_of_repeats_subj%02d.npy" % (args.output_root, subj)
    )

    activations = get_preloaded_features(
        subj,
        stimulus_list,
        "%s" % model,
        features_dir="%s/features" % args.output_root,
    )
    # getting scores and plotting

    # plot sampled images
    plt.figure(figsize=(30, 30))

    print(weight.shape)

    for w in tqdm(range(weight.shape[1])):
        # scores = np.mean(activations.squeeze() @ weight, axis=1)
        scores = activations.squeeze() @ weight[:, w]
        sampled_img_ids = stimulus_list[np.argsort(scores)[::-1][:20]]

        # plot images
        for j, id in enumerate(sampled_img_ids):
            plt.subplot(4, 5, j + 1)
            I = get_coco_image(id, coco_train, coco_val)
            plt.axis("off")
            plt.imshow(I)
        plt.tight_layout()

        # plt.savefig(
        #     "figures/CLIP/max_img4vox/%s_max_images_%s.png"
        #     % (args.model, save_name)
        # )
        plt.savefig(
            "figures/CLIP/max_img4vox/single_voxel/%s_max_images_%s_%d.png"
            % (args.model, save_name, w)
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subj",
        type=int,
        default=1,
        help="Specify which subject to build model on. Currently it supports subject 1, 2, 7",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=".",
        help="Specify the path to the output directory",
    )
    parser.add_argument("--feature_dir", default="features")
    parser.add_argument("--roi", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--roi_value", default=0, type=int)
    parser.add_argument(
        "--coarse_level_semantic_analysis", default=False, action="store_true"
    )
    parser.add_argument(
        "--sample_level_semantic_analysis", default=False, action="store_true"
    )
    parser.add_argument(
        "--compare_brain_and_clip_performance", default=False, action="store_true"
    )
    parser.add_argument(
        "--compare_to_human_judgement", default=False, action="store_true"
    )
    parser.add_argument(
        "--image_level_scatter_plot", default=False, action="store_true"
    )
    parser.add_argument(
        "--category_based_similarity_analysis", default=False, action="store_true"
    )
    parser.add_argument("--weight_analysis", default=False, action="store_true")
    parser.add_argument(
        "--extract_keywords_for_roi", default=False, action="store_true"
    )
    parser.add_argument("--extract_captions_for_roi", default=None, type=str)
    parser.add_argument("--mask", default=False, action="store_true")
    parser.add_argument("--vox_img_maximization", default=False, action="store_true")
    args = parser.parse_args()

    if args.compare_brain_and_clip_performance:
        compare_model_and_brain_performance_on_COCO()

    if args.coarse_level_semantic_analysis:
        coarse_level_semantic_analysis(subj=1)

    if args.image_level_scatter_plot:
        models = [
            "clip",
            "YFCC_clip",
            "YFCC_slip",
            "YFCC_simclr",
            "resnet50_bottleneck",
        ]
        # plt.figure(figsize=(10, 10))
        # for m, model in enumerate(models):
        #     image_level_scatter_plot(model1="bert_layer_13", model2=model, i=m + 1, subplot=True, do_zscore=True)
        # # plt.legend()
        # plt.savefig("figures/CLIP/manifold_distance/bert_vs_others.png", dpi=400)

        # model1, model2 = "clip", "resnet50_bottleneck"
        # # threshold = (0.8, 0.2, "br")
        # plt.figure()
        # image_level_scatter_plot(model1=model1, model2=model2, do_zscore=False)

        model1, model2 = "YFCC_clip", "YFCC_simclr"
        # threshold = (0.6, 0.8, "br")
        plt.figure()
        image_level_scatter_plot(
            model1,
            model2,
            do_zscore=False,
            # threshold=threshold,
        )

        # model1, model2 = "YFCC_slip", "YFCC_simclr"
        # # threshold = (0.6, 0.8, "br")
        # plt.figure()
        # image_level_scatter_plot(
        #     model1,
        #     model2,
        #     do_zscore=False,
        #     # threshold=threshold,
        # )

    if args.extract_keywords_for_roi:
        with open(
            "%s/output/clip/roi_maximization/1000eng.txt" % args.output_root
        ) as f:
            out = f.readlines()
        common_words = ["photo of " + w[:-1] for w in out]
        try:
            activations = np.load(
                "%s/output/clip/roi_maximization/1000eng_activation.npy"
                % args.output_root
            )
        except FileNotFoundError:
            from nltk.corpus import wordnet
            import clip
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, _ = clip.load("ViT-B/32", device=device)
            activations = extract_text_activations(model, common_words)
            np.save(
                "%s/output/clip/roi_maximization/1000eng_activation.npy"
                % args.output_root,
                activations,
            )

        w = np.load(
            "%s/output/encoding_results/subj%d/weights_clip_whole_brain.npy"
            % (args.output_root, args.subj)
        )

        # extract_keywords_for_roi(w, activations, common_words, "floc-faces", [2, 3])
        # extract_keywords_for_roi(w, activations, common_words, "floc-bodies", [1, 2, 3])
        # extract_keywords_for_roi(
        #     w, activations, common_words, "floc-places", [1, 2, 3, 4]
        # )
        model = "YFCC_slip"
        fdr_p = np.load(
            "%s/output/ci_threshold/%s_unique_var_fdr_p_subj%01d.npy"
            % (args.output_root, model, args.subj)
        )
        weight_mask = fdr_p[1] < 0.05
        extract_keywords_for_roi(
            w,
            activations,
            common_words,
            roi_name="YFCC_slip_unique_var",
            mask=weight_mask,
        )

    if args.extract_captions_for_roi is not None:
        from util.model_config import roi_name_dict

        if args.extract_captions_for_roi in roi_name_dict.keys():
            extract_captions_for_voxel(args.extract_captions_for_roi)
        else:
            model = args.extract_captions_for_roi
            fdr_p = np.load(
                "%s/output/ci_threshold/%s_unique_var_fdr_p_subj%01d.npy"
                % (args.output_root, model, args.subj)
            )
            weight_mask = fdr_p[1] < 0.05
            extract_captions_for_voxel(mask=weight_mask, roi="%s_unique_var" % model)

    if args.sample_level_semantic_analysis:
        import skimage.io as io

        from pycocotools.coco import COCO
        import configparser

        config = configparser.ConfigParser()
        config.read("config.cfg")
        annFile_train = config["COCO"]["AnnFileTrain"]
        annFile_val = config["COCO"]["AnnFileVal"]
        annFile_train_caps = config["COCO"]["AnnFileTrainCaps"]
        annFile_val_caps = config["COCO"]["AnnFileValCaps"]

        coco_train = COCO(annFile_train)
        coco_val = COCO(annFile_val)
        coco_train_caps = COCO(annFile_train_caps)
        coco_val_caps = COCO(annFile_val_caps)

        compute_average_pairwise_brain_corr(subj=args.subj, sample_n=5000)

        for normalize in ["sub_baseline", "None"]:

            # sample_level_semantic_analysis(
            #     subj=args.subj,
            #     model1="clip",
            #     model2="resnet50_bottleneck",
            #     print_caption=True,
            #     print_distance=False,
            #     threshold=(0.8, 0.2, "br"),
            #     normalize=normalize,
            # )

            # sample_level_semantic_analysis(
            #     subj=args.subj,
            #     model1="YFCC_clip",
            #     model2="YFCC_simclr",
            #     print_caption=True,
            #     print_distance=False,
            #     threshold=(0.65, 0.8, "br"),
            #     normalize=normalize,
            # )

            # sample_level_semantic_analysis(
            #     subj=args.subj,
            #     model1="YFCC_slip",
            #     model2="YFCC_simclr",
            #     print_caption=True,
            #     print_distance=False,
            #     threshold=(0.65, 0.8, "br"),
            #     normalize=normalize,
            # )

            sample_level_semantic_analysis(
                subj=args.subj,
                model1="clip",
                model2="visual_layer_11",
                print_caption=True,
                print_distance=False,
                normalize=normalize,
            )

            sample_level_semantic_analysis(
                subj=args.subj,
                model1="clip",
                model2="laion400m_clip",
                print_caption=True,
                print_distance=False,
                normalize=normalize,
            )

            sample_level_semantic_analysis(
                subj=args.subj,
                model1="laion2b_clip",
                model2="laion400m_clip",
                print_caption=True,
                print_distance=False,
                normalize=normalize,
            )

            # sample_level_semantic_analysis(
            #     subj=args.subj,
            #     model1="clip",
            #     model2="resnet50_bottleneck",
            #     print_caption=True,
            #     print_distance=False,
            #     threshold=(0.3, 0.4, "tl"),
            #     normalize=normalize,
            # )

            # sample_level_semantic_analysis(
            #     subj=args.subj,
            #     model1="YFCC_clip",
            #     model2="YFCC_simclr",
            #     print_caption=True,
            #     print_distance=False,
            #     threshold=(0.25, 0.9, "tl"),
            #     normalize=normalize,
            # )

            # sample_level_semantic_analysis(
            #     subj=args.subj,
            #     model1="YFCC_slip",
            #     model2="YFCC_simclr",
            #     print_caption=True,
            #     print_distance=False,
            #     threshold=(0.35, 0.9, "tl"),
            #     normalize=normalize,
            # )

            # sample_level_semantic_analysis(
            #     subj=args.subj,
            #     model1="YFCC_slip",
            #     model2="YFCC_simclr",
            #     print_caption=True,
            #     print_distance=False,
            #     threshold=(0.35, 0.9, "tl"),
            #     normalize=None,
            # )

        # sample_level_semantic_analysis(
        #     subj=args.subj, model1="clip", model2="bert_layer_13"
        # )
        # sample_level_semantic_analysis(
        #     subj=args.subj, model1="visual_layer_11", model2="resnet50_bottleneck"
        # )
        # sample_level_semantic_analysis(
        #     subj=args.subj, model1="clip", model2="visual_layer_1"
        # )
        # sample_level_semantic_analysis(
        #     subj=args.subj, model1="clip", model2="clip_text"
        # )

    if args.category_based_similarity_analysis:
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        models = [
            "clip",
            "YFCC_clip",
            "YFCC_slip",
            "YFCC_simclr",
            "resnet50_bottleneck",
        ]
        for threshold in thresholds:
            scores = []
            for model in models:
                score, person_n = category_based_similarity_analysis(
                    model, threshold=threshold
                )
                # print(model)
                # print(score)
                scores.append(score)
            plt.figure()
            plt.bar(np.arange(len(scores)), scores)
            plt.xticks(ticks=np.arange(len(scores)), labels=models, rotation="45")
            plt.ylabel("within/(within+cross)")
            plt.title("# of pics with person: %d/10000" % person_n)
            plt.subplots_adjust(bottom=0.25)
            plt.savefig("figures/CLIP/category_based_sim/person_%.1f.png" % threshold)

    if args.compare_to_human_judgement:
        human_emb_path = (
            "data/human_similarity_judgement/spose_embedding_49d_sorted.txt"
        )
        word_path = "data/human_similarity_judgement/unique_id.txt"

        human_emb = np.loadtxt(human_emb_path)
        emb_label = np.loadtxt(word_path, dtype="S")
        emb_label = [w.decode("utf-8") for w in emb_label]

        # checked that the label and emb matches
        from util.model_config import COCO_cat, COCO_super_cat

        print(len(COCO_cat))
        count = 0
        COCO_HJ_overlap = []
        # all_coco = COCO_cat + COCO_super_cat
        for w in COCO_cat:
            if "_" in w:
                word = "_".join(w.split(" "))
            else:
                word = w

            if word in emb_label:
                count += 1
                COCO_HJ_overlap.append(w)

        print(count)
        # print(COCO_HJ_overlap)
        # ['car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        # 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
        # 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'tie', 'suitcase',
        # 'frisbee', 'snowboard', 'kite', 'skateboard', 'surfboard', 'bottle',
        # 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        # 'orange', 'broccoli', 'carrot', 'pizza', 'donut', 'cake', 'chair',
        # 'couch', 'bed', 'toilet', 'laptop', 'keyboard', 'microwave', 'oven',
        # 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'toothbrush']

        # compare clip, bert, and human judgement
        # clip
        clip_model, _ = clip.load("ViT-B/32", device=device)
        clip_features = []
        for word in COCO_HJ_overlap:
            with torch.no_grad():
                expression = "a photo of " + word
                text = clip.tokenize(expression).to(device)
                emb = clip_model.encode_text(text).cpu().data.numpy()
                clip_features.append(emb)
        clip_features = np.array(clip_features).squeeze()
        rsm_clip = np.corrcoef(clip_features)

        # bert
        bert_features = []
        from transformers import BertTokenizer, BertModel

        bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        bert_model = BertModel.from_pretrained(
            "bert-base-uncased", output_hidden_states=True
        )

        bert_model.eval()
        for word in COCO_HJ_overlap:
            text = "[CLS] a photo of " + word + " [SEP]"
            tokenized_text = bert_tokenizer.tokenize(text)
            indexed_tokens = bert_tokenizer.convert_tokens_to_ids(tokenized_text)
            segments_ids = [1] * len(tokenized_text)
            tokens_tensor = torch.tensor([indexed_tokens])
            segments_tensors = torch.tensor([segments_ids])
            with torch.no_grad():
                outputs = bert_model(tokens_tensor, segments_tensors)
                hidden_states = outputs[2]
                # print(len(hidden_states[-1][0][4]))
                # print(hidden_states[-1][0][4])
                # print(hidden_states)
                # print(hidden_states.shape)
                bert_features.append(
                    hidden_states[-1][0][4].numpy()
                )  # embedding of the word, from first layer of bert
        rsm_bert = np.corrcoef(bert_features)

        # human judgement model
        hj_features = []
        for word in COCO_HJ_overlap:
            ind = list(emb_label).index(word)
            hj_features.append(human_emb[ind])
        hj_features = np.array(hj_features)
        rsm_hj = np.corrcoef(hj_features)

        print(np.corrcoef(rsm_hj.flatten(), rsm_clip.flatten()))
        print(np.corrcoef(rsm_hj.flatten(), rsm_bert.flatten()))
        print(np.corrcoef(rsm_bert.flatten(), rsm_clip.flatten()))

        plt.figure()

        plt.subplot(1, 3, 1)
        plt.imshow(rsm_clip)
        plt.colorbar()
        plt.title("CLIP")

        plt.subplot(1, 3, 2)
        plt.imshow(rsm_bert)
        plt.colorbar()
        plt.title("BERT")

        plt.subplot(1, 3, 3)
        plt.imshow(rsm_hj)
        plt.colorbar()
        plt.title("Human Behavior")

        plt.tight_layout()

        plt.savefig("figures/CLIP/human_judgement_rsm_comparison_bert13.png")

    if args.vox_img_maximization:
        from pycocotools.coco import COCO
        import configparser

        config = configparser.ConfigParser()
        config.read("config.cfg")
        annFile_train = config["COCO"]["AnnFileTrain"]
        annFile_val = config["COCO"]["AnnFileVal"]

        coco_train = COCO(annFile_train)
        coco_val = COCO(annFile_val)

        w = np.load(
            "%s/output/encoding_results/subj%d/weights_%s_whole_brain.npy"
            % (args.output_root, args.subj, args.model)
        )
        w = fill_in_nan_voxels(w, args.subj, args.output_root)

        # visualize slip uni var vox
        fdr_p = np.load(
            "%s/output/ci_threshold/%s_unique_var_fdr_p_subj%01d.npy"
            % (args.output_root, "YFCC_slip", args.subj)
        )
        weight_mask_slipu = fdr_p[1] < 0.0001
        import pdb

        pdb.set_trace()
        print(np.sum(weight_mask_slipu))
        weight_mask_slipu = fill_in_nan_voxels(
            weight_mask_slipu, subj=args.subj, output_root=args.output_root
        )
        max_img4vox(
            w[:, weight_mask_slipu],
            args.subj,
            args.model,
            save_name="YFCC_slip_unique_var",
        )

        # visualize general EBA voxel
        roi_mask = np.load(
            "%s/output/voxels_masks/subj%d/roi_1d_mask_subj%02d_%s.npy"
            % (args.output_root, args.subj, args.subj, "floc-bodies")
        )
        # weight_mask_eba = roi_mask == 1
        # weight_mask_eba = fill_in_nan_voxels(
        #     weight_mask_eba, subj=args.subj, output_root=args.output_root
        # )
        # max_img4vox(w[:, weight_mask_eba], args.subj, args.model, save_name="EBA")

        # visualize EBA - SLIP unique var
        # weight_mask_diff1 = np.clip((weight_mask_eba.astype(int) - weight_mask_slipu.astype(int)), 0, 1).astype(bool)
        # max_img4vox(w[:, weight_mask_diff1], args.subj, args.model, save_name="EBA-slipu")

        # weight_mask_diff2 = np.clip((weight_mask_slipu.astype(int) - weight_mask_eba.astype(int)), 0, 1).astype(bool)
        # max_img4vox(w[:, weight_mask_diff2], args.subj, args.model, save_name="slipu-EBA")
