from cProfile import label
import os
import argparse

import pickle
from unicodedata import name

import seaborn as sns
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.decomposition import PCA


# import torch
import clip

from util.data_util import (
    load_model_performance,
    extract_test_image_ids,
    fill_in_nan_voxels,
)
from util.model_config import *


def make_word_cloud(text, saving_fname):
    from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

    text = " ".join(t for t in text)
    # print(text)
    wordcloud = WordCloud(background_color="white").generate(text)

    # Display the generated image:
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    wordcloud.to_file(saving_fname)


def make_name_modifier(args):
    if (
        (args.threshold == 0)
        and (args.best_voxel_n == 0)
        and (args.roi_only is None)
        and (args.model_unique_var is None)
    ):
        raise NameError("One of the selection criteria has to be used.")

    if args.roi_only is not None:
        if args.sub_roi is not None:
            name_modifier = "%s_only" % args.sub_roi
        else:
            regions = "_".join(args.roi_only)
            name_modifier = "%s_only" % regions

    if args.threshold != 0:
        name_modifier = "acc_%.1f" % args.threshold
    elif args.best_voxel_n != 0:
        name_modifier = "best_%d" % args.best_voxel_n
    elif args.model_unique_var is not None:
        name_modifier = "%s_unique_var" % args.model_unique_var
        return name_modifier

    if args.mask_out_roi is not None:
        name_modifier += "_minus_%s" % args.mask_out_roi

    if args.nc_corrected:
        name_modifier += "_nc"

    return name_modifier


def extract_single_subject_weight(subj, args):
    name_modifier = make_name_modifier(args)
    w = np.load(
        "%s/output/encoding_results/subj%d/weights_%s_whole_brain.npy"
        % (args.output_root, subj, args.model)
    )

    # w = np.load(
    #     "%s/output/encoding_results/subj%d/weights_%s_whole_brain.npy"
    #     % (args.output_root, subj, model)
    # )
    w = fill_in_nan_voxels(w, subj, args.output_root)
    if args.roi_only is not None:
        roi_mask = np.load(
            "%s/output/voxels_masks/subj%d/roi_1d_mask_subj%02d_%s.npy"
            % (args.output_root, subj, subj, args.roi_only[0])
        )
        if len(args.roi_only) > 1:
            print("Loading multiple ROIs")
            for region in args.roi_only[1:]:
                more_mask = np.load(
                    "%s/output/voxels_masks/subj%d/roi_1d_mask_subj%02d_%s.npy"
                    % (args.output_root, subj, subj, region)
                )
                roi_mask = roi_mask + more_mask
        elif args.sub_roi is not None:
            from util.model_config import roi_name_dict

            roi_dict = roi_name_dict[args.roi_only[0]]
            roi_int = [k for k, v in roi_dict.items() if v == args.sub_roi][0]
            roi_mask = roi_mask == roi_int

        weight_mask = roi_mask > 0

    elif args.mask_out_roi is not None:
        roi_mask = np.load(
            "%s/output/voxels_masks/subj%d/roi_1d_mask_subj%02d_%s.npy"
            % (args.output_root, subj, subj, args.mask_out_roi)
        )

        weight_mask = ~roi_mask
        print("masking out %d voxels..." % sum(roi_mask))

    elif args.model_unique_var is not None:
        fdr_p = np.load(
            "%s/output/ci_threshold/%s_unique_var_fdr_p_subj%01d.npy"
            % (args.output_root, args.model_unique_var, subj)
        )
        weight_mask = fdr_p[1] < 0.05
        # print(weight_mask.shape)
        weight_mask = fill_in_nan_voxels(
            weight_mask, subj=subj, output_root=args.output_root
        )
    else:
        weight_mask = np.ones(w.shape[1])

    rsq = load_model_performance(
        args.model, output_root=args.output_root, subj=subj, measure="rsq"
    )
    if args.nc_corrected:
        nc = np.load(
            "%s/output/noise_ceiling/subj%01d/noise_ceiling_1d_subj%02d.npy"
            % (args.output_root, subj, subj)
        )
        rsq = rsq / (nc / 100)
    if args.threshold == 0:  # then selecting voxels based on number of accuracy
        if args.best_voxel_n != 0:
            threshold = rsq[
                np.argsort(rsq)[-args.best_voxel_n]
            ]  # get the threshold for the best n voxels
            print("The threshold for best 20000 voxels is: " + str())
        else:
            threshold = 0  # select all voxels
    else:
        threshold = args.threshold

    acc_mask = rsq >= threshold
    weight_mask = (weight_mask * acc_mask).astype(bool)
    print("Total voxels left: %d" % sum(weight_mask))
    if not os.path.exists(
        "%s/output/pca/%s/%s/pca_voxels" % (args.output_root, args.model, name_modifier)
    ):
        os.makedirs(
            "%s/output/pca/%s/%s/pca_voxels"
            % (args.output_root, args.model, name_modifier)
        )

    np.save(
        "%s/output/pca/%s/%s/pca_voxels/pca_voxels_subj%02d.npy"
        % (args.output_root, args.model, name_modifier, subj),
        weight_mask,
    )
    return w[:, weight_mask]


def load_weight_matrix_from_subjs_for_pca(args):
    # subjs = np.arange(1, 9)
    subjs = [1, 2, 5, 7]
    name_modifier = make_name_modifier(args)
    group_w_path = "%s/output/pca/%s/%s/weight_matrix.npy" % (
        args.output_root,
        args.model,
        name_modifier,
    )

    if not os.path.exists("%s/output/pca/%s" % (args.output_root, args.model)):
        os.makedirs("%s/output/pca/%s" % (args.output_root, args.model))

    try:
        group_w = np.load(group_w_path)

    except FileNotFoundError:
        print("Generating new weight matrix")
        group_w = []
        for subj in subjs:
            subj_w = extract_single_subject_weight(subj, args)
            group_w.append(subj_w)
        group_w = np.hstack(group_w)
        np.save(group_w_path, group_w)
    return group_w


def get_PCs(args, data=None, return_pca_object=False):
    name_modifier = make_name_modifier(args)
    try:
        fpath = "%s/output/pca/%s/%s/pca_group_components.npy" % (
            args.output_root,
            args.model,
            name_modifier,
        )
        print("Loading PCA from: " + fpath)
        if return_pca_object:
            with open(
                "%s/output/pca/%s/%s/pca_group_components.pkl"
                % (args.output_root, args.model, name_modifier),
                "rb",
            ) as f:
                pca = pickle.load(f)
        else:
            PCs = np.load(fpath)

    except FileNotFoundError:
        output_dir = "%s/output/pca/%s/%s/" % (
            args.output_root,
            args.model,
            name_modifier,
        )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            os.makedirs(output_dir + "pca_voxels/")

        print("Running PCA of: " + name_modifier)
        if data is None:
            data = load_weight_matrix_from_subjs_for_pca(args).T

        print("Shape of weight is:")
        print(data.shape)

        # remove outlier voxels
        print("Outlier:")
        print(data[np.where(data > 30)])

        data[np.where(data > 30)] = 0  # TODO: check why this weight is so high

        pca = PCA(n_components=args.num_pc, svd_solver="full")
        pca.fit(data)
        PCs = pca.components_
        np.save(fpath, PCs)

        with open(
            "%s/output/pca/%s/%s/pca_group_components.pkl"
            % (args.output_root, args.model, name_modifier),
            "wb",
        ) as f:
            pickle.dump(pca, f)

        plt.plot(pca.explained_variance_ratio_)
        plt.xlabel("Principal Components")
        plt.ylabel("Explained Variance")
        plt.xticks(ticks=np.arange(20) + 1)
        plt.savefig("figures/PCA/ev/%s_pca_group_%s.png" % (args.model, name_modifier))

    if return_pca_object:
        return pca, name_modifier
    else:
        return PCs, name_modifier


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subj", type=int, default=0, help="Specify which subject to build model on.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=".",
        help="Specify the path to the output directory",
    )
    parser.add_argument("--model", default="clip")
    parser.add_argument("--threshold", default=0)
    parser.add_argument("--num_pc", default=20)
    parser.add_argument("--best_voxel_n", type=int, default=0)
    parser.add_argument("--roi_only", default=None, nargs="+")
    parser.add_argument("--sub_roi", default=None)
    parser.add_argument("--model_unique_var", default=None)
    parser.add_argument("--mask_out_roi", default=None)
    parser.add_argument("--nc_corrected", default=False, action="store_true")
    parser.add_argument("--plotting", default=True)
    parser.add_argument("--group_pca_analysis", default=False, action="store_true")
    parser.add_argument("--pc_text_visualization", default=False, action="store_true")
    parser.add_argument("--pc_image_visualization", default=False, action="store_true")
    parser.add_argument("--proj_feature_pc_to_subj", default=False, action="store_true")
    parser.add_argument(
        "--analyze_PCproj_consistency", default=False, action="store_true"
    )
    parser.add_argument("--image2pc", default=False, action="store_true")
    parser.add_argument(
        "--load_and_show_all_word_clouds", default=False, action="store_true"
    )
    parser.add_argument(
        "--kmean_clustering_on_pc_proj", default=False, action="store_true"
    )
    parser.add_argument("--hclustering_on_pc_proj", default=False, action="store_true")

    parser.add_argument(
        "--maximize_input_for_cluster", default=False, action="store_true"
    )
    parser.add_argument("--analyze_PC_images", default=False, action="store_true")
    parser.add_argument("--uv_vs_pc", default=False, action="store_true")

    args = parser.parse_args()

    pca, name_modifier = get_PCs(args, return_pca_object=True)
    plt.plot(pca.explained_variance_ratio_)
    plt.xlabel("Principal Components")
    plt.ylabel("Explained Variance")
    plt.xticks(ticks=np.arange(20) + 1)
    plt.savefig("figures/PCA/ev/%s_pca_group_%s.png" % (args.model, name_modifier))

    if args.group_pca_analysis:
        from util.coco_utils import (
            get_coco_anns,
            get_coco_image,
            get_coco_caps,
        )

        from interpret_models import extract_text_activations, extract_emb_keywords
        from featureprep.feature_prep import get_preloaded_features

        PCs, name_modifier = get_PCs(args)

        ############## pc_image_visualization ##############
        if not os.path.exists(
            "figures/PCA/image_vis/%s/%s/" % (args.model, name_modifier)
        ):
            os.makedirs("figures/PCA/image_vis/%s/%s/" % (args.model, name_modifier))

        stimulus_list = np.load(
            "%s/output/coco_ID_of_repeats_subj%02d.npy" % (args.output_root, 1)
        )

        activations = get_preloaded_features(
            1,
            stimulus_list,
            "%s" % args.model,
            features_dir="%s/features" % args.output_root,
        )  # using images that subj 1 saw as a random "sample"

        # getting scores and plotting
        from pycocotools.coco import COCO

        annFile_train = "/lab_data/tarrlab/common/datasets/coco_annotations/instances_train2017.json"
        annFile_val = (
            "/lab_data/tarrlab/common/datasets/coco_annotations/instances_val2017.json"
        )
        coco_train = COCO(annFile_train)
        coco_val = COCO(annFile_val)

        cats = coco_train.loadCats(coco_train.getCatIds())
        id2cat = {}
        for cat in cats:
            id2cat[cat["id"]] = cat["name"]

        # compute label embedding correlation
        best_label_corrs, worst_label_corrs = [], []
        COCO_cat_feat = get_preloaded_features(
            1, stimulus_list, "cat", features_dir="%s/features" % args.output_root,
        )

        # plot sampled images
        plt.figure(figsize=(30, 30))
        for i in tqdm(range(PCs.shape[0])):
            n_samples = int(len(stimulus_list) / 20)
            sample_idx = np.arange(0, len(stimulus_list), n_samples)

            scores = activations.squeeze() @ PCs[i, :]
            sampled_img_ids = stimulus_list[np.argsort(scores)[::-1][sample_idx]]

            # plot images
            for j, id in enumerate(sampled_img_ids):
                plt.subplot(20, 20, i * 20 + j + 1)
                I = get_coco_image(id, coco_train, coco_val)
                plt.axis("off")
                plt.imshow(I)
        plt.tight_layout()

        plt.savefig(
            "figures/PCA/image_vis/%s/%s/pc_sampled_images.png"
            % (args.model, name_modifier)
        )

        for i in tqdm(range(PCs.shape[0])):
            scores = activations.squeeze() @ PCs[i, :]
            best_img_ids = stimulus_list[np.argsort(scores)[::-1][:20]]
            worst_img_ids = stimulus_list[np.argsort(scores)[:20]]

            if args.plotting:
                # plot images
                plt.figure()
                for j, id in enumerate(best_img_ids):
                    plt.subplot(4, 5, j + 1)
                    I = get_coco_image(id, coco_train, coco_val)
                    plt.axis("off")
                    plt.imshow(I)
                plt.tight_layout()
                plt.savefig(
                    "figures/PCA/image_vis/%s/%s/pc%d_best_images.png"
                    % (args.model, name_modifier, i)
                )
                plt.close()

                plt.figure()
                for j, id in enumerate(worst_img_ids):
                    plt.subplot(4, 5, j + 1)
                    I = get_coco_image(id, coco_train, coco_val)
                    plt.axis("off")
                    plt.imshow(I)
                plt.tight_layout()
                plt.savefig(
                    "figures/PCA/image_vis/%s/%s/pc%d_worst_images.png"
                    % (args.model, name_modifier, i)
                )
                plt.close()

            # #find corresponding captions of best image
            # best_caps, worst_caps = [], []
            # for j, id in enumerate(best_img_ids):
            #     captions = get_coco_caps(id)
            #     best_caps += captions

            # for j, id in enumerate(worst_img_ids):
            #     captions = get_coco_caps(id)
            #     worst_caps += captions

            # # print(best_caps)
            # # print(worst_caps)

            # make_word_cloud(best_caps, saving_fname="./figures/PCA/image_vis/word_clouds/PC%d_best_captions.png" % i)
            # make_word_cloud(worst_caps, saving_fname="./figures/PCA/image_vis/word_clouds/PC%d_worst_captions.png" % i)

            # calculate label consistency
            cat_feats = []
            for j, id in enumerate(best_img_ids):
                idx = np.where(stimulus_list == id)[0]
                cat_feats.append(COCO_cat_feat[idx, :])

            cat_feats = np.array(cat_feats).squeeze()
            # corr = (np.sum(np.corrcoef(cat_feats)) - num_pc) / (num_pc^2-num_pc)
            corr = np.mean(np.corrcoef(cat_feats))
            best_label_corrs.append(corr)

            cat_feats = []
            for j, id in enumerate(worst_img_ids):
                idx = np.where(stimulus_list == id)[0]
                cat_feats.append(COCO_cat_feat[idx, :])

            cat_feats = np.array(cat_feats).squeeze()
            # print(cat_feats.shape)
            # corr = (np.sum(np.corrcoef(cat_feats)) - num_pc) / (num_pc^2-num_pc)
            corr = np.mean(np.corrcoef(cat_feats))
            worst_label_corrs.append(corr)

        plt.figure()
        plt.plot(np.arange(20), worst_label_corrs, label="Worst")
        plt.plot(np.arange(20), best_label_corrs, label="Best")

        plt.ylabel("Mean Pairwise Correlation")
        plt.xlabel("PCs")
        plt.legend()
        plt.savefig(
            "figures/PCA/image_vis/%s/%s/pc_label_corr.png"
            % (args.model, name_modifier)
        )

    if args.proj_feature_pc_to_subj:
        # Calculate weight projection onto PC space
        pca, name_modifier = get_PCs(args, return_pca_object=True)
        print("Projecting PC to subject: %s" % name_modifier)

        group_w = load_weight_matrix_from_subjs_for_pca(args).T  # (80,000 x 512)
        group_w -= np.mean(group_w, axis=0)  # each feature should have mean 0

        # method 1
        w_transformed = pca.transform(group_w)
        # w_transformed = np.dot(group_w, PCs.T)  # (80,000x512 x 512x20)

        proj = w_transformed.T  # should be (# of PCs) x (# of voxels)
        from util.util import zscore

        print("check norm and correlation")
        print((proj[0, :] ** 2).mean())
        print((proj[1, :] ** 2).mean())
        print(np.corrcoef(zscore(proj, axis=1)))

        print(name_modifier)

        subjs = np.arange(1, 9)
        idx = 0
        for subj in subjs:
            subj_mask = np.load(
                "%s/output/pca/%s/%s/pca_voxels/pca_voxels_subj%02d.npy"
                % (args.output_root, args.model, name_modifier, subj)
            )
            subj_proj = np.zeros((args.num_pc, len(subj_mask)))
            subj_proj[:, subj_mask] = proj[:, idx : idx + np.sum(subj_mask)]
            print(subj)
            print((proj[0, idx : idx + np.sum(subj_mask)] ** 2).mean())
            print((proj[1, idx : idx + np.sum(subj_mask)] ** 2).mean())
            save_dir = "%s/output/pca/%s/%s/subj%02d" % (
                args.output_root,
                args.model,
                name_modifier,
                subj,
            )
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            np.save(
                "%s/pca_projections.npy" % (save_dir), subj_proj,
            )
            idx += np.sum(subj_mask)

    if args.analyze_PCproj_consistency:
        from analyze_in_mni import analyze_data_correlation_in_mni

        subjs = np.arange(1, 9)
        name_modifier = make_name_modifier(args)
        # load all PC projection from all 8 subjs
        all_PC_projs = []
        for subj in subjs:
            all_PC_projs.append(
                np.load(
                    "%s/output/pca/%s/%s/subj%02d/pca_projections.npy"
                    % (args.output_root, args.model, name_modifier, subj)
                )
            )

        # remember to run `module load fsl-6.0.3` on cluster
        analyze_data_correlation_in_mni(
            all_PC_projs,
            args.model,
            save_name="PC_proj_%s" % name_modifier,
            subjs=subjs,
            dim=20,
        )

    # if args.image2pc:
    #     from featureprep.feature_prep import get_preloaded_features
    #     from analyze_clip_results import extract_text_activations, extract_emb_keywords, get_coco_anns, get_coco_image, get_coco_caps

    #     model = "clip"
    #     stimulus_list = np.load(
    #         "%s/output/coco_ID_of_repeats_subj%02d.npy" % (args.output_root, 1)
    #     )

    #     activations = get_preloaded_features(
    #         1,
    #         stimulus_list,
    #         "%s" % model.replace("_rep_only", ""),
    #         features_dir="%s/features" % args.output_root,
    #     )

    #     PCs, name_modifier = get_PCs(model=model)
    #     nPC = PCs.shape[0]
    #     pc_proj = np.dot(activations, PCs.T) # returns a 10000 by 20 matrix
    #     top2_pcs = np.argsort(np.abs(pc_proj), axis=1)[:, -2:] # returns a 10000 by 2 matrix
    #     pc_counter = np.zeros((nPC, nPC))
    #     for i in range(top2_pcs.shape[0]):
    #         j, k = top2_pcs[i, :]
    #         pc_counter[j, k] += 1
    #         pc_counter[k, j] += 1

    #     plt.imshow(pc_counter, cmap="Blues")
    #     plt.xlabel("PCs")
    #     plt.ylabel("PCs")
    #     plt.xticks(np.arange(nPC))
    #     plt.yticks(np.arange(nPC))
    #     plt.colorbar()
    #     plt.savefig("figures/PCA/top2pc/top2pc.png")

    #     ind = np.unravel_index(np.argsort(pc_counter, axis=None), pc_counter.shape)
    #     best_2_pcs = [(ind[0][::-1][i], ind[1][::-1][i]) for i in np.arange(0, 10, 2)]
    #     print(best_2_pcs)
    #     for p in best_2_pcs:
    #         proj = np.vstack((pc_proj[:, p[0]], pc_proj[:, p[1]])).T
    #         proj_norm = np.linalg.norm(proj, axis=1)
    #         img_rank = np.argsort(proj_norm)[::-1][:20]
    #         plt.figure(figsize=(20,20))
    #         for i, idx in enumerate(img_rank):
    #             coco_id = stimulus_list[idx]
    #             I = get_coco_image(coco_id)
    #             plt.subplot(4, 5, i+1)
    #             plt.imshow(I)
    #             plt.title("proj: %.2f, %.2f" % (pc_proj[idx, p[0]], pc_proj[idx, p[1]]))
    #         plt.savefig("figures/PCA/top2pc/top_images_for_PC%d&%d_%s.png" % (p[0], p[1], name_modifier))

    # proj_norm = np.linalg.norm(pc_proj, axis=1)
    # img_rank = np.argsort(proj_norm)[::-1][:20]
    # plt.figure(figsize=(30,10))
    # for i, idx in enumerate(img_rank):
    #     coco_id = stimulus_list[idx]
    #     I = get_coco_image(coco_id)
    #     pref_pc = np.argsort(pc_proj[idx,:])[::-1][:2]
    #     first2 = ["%d:%.2f" % (pc, pc_proj[idx, pc]) for pc in pref_pc]
    #     plt.subplot(4, 5, i + 1)
    #     plt.axis("off")
    #     plt.imshow(I)
    #     plt.title(first2)
    # plt.tight_layout()
    # plt.savefig("figures/PCA/image_vis/image2PC/image_PC_proj_%s_l2max.png" % model)

    # img_rank = np.argsort(proj_norm)[:20]
    # plt.figure(figsize=(30,10))
    # for i, idx in enumerate(img_rank):
    #     coco_id = stimulus_list[idx]
    #     I = get_coco_image(coco_id)
    #     pref_pc = np.argsort(pc_proj[idx,:])[::-1][:3]
    #     first3 = ["%d:%.2f" % (pc, pc_proj[idx, pc]) for pc in pref_pc]
    #     plt.subplot(4, 5, i + 1)
    #     plt.axis("off")
    #     plt.imshow(I)
    #     plt.title(first3)
    # plt.tight_layout()
    # plt.savefig("figures/PCA/image_vis/image2PC/image_PC_proj_%s_l2min.png" % model)
    # plt.close()

    # proj_norm = np.linalg.norm(pc_proj, ord=-np.inf, axis=1)
    # img_rank = np.argsort(proj_norm)[:20]
    # plt.figure(figsize=(30,10))
    # for i, idx in enumerate(img_rank):
    #     coco_id = stimulus_list[idx]
    #     I = get_coco_image(coco_id)
    #     pref_pc = np.argsort(pc_proj[idx,:])[::-1][:3]
    #     first3 = ["%d:%.2f" % (pc, pc_proj[idx, pc]) for pc in pref_pc]
    #     plt.subplot(4, 5, i + 1)
    #     plt.axis("off")
    #     plt.imshow(I)
    #     plt.title(first3)
    # plt.tight_layout()
    # plt.savefig("figures/PCA/image_vis/image2PC/image_PC_proj_%s_-inf.png" % model)
    # plt.close()

    # if args.load_and_show_all_word_clouds:
    #     import matplotlib.image as img
    #     plt.figure(figsize=(10, 50))
    #     for i in range(20):
    #         plt.subplot(20, 2, i*2+1)
    #         im = img.imread("./figures/PCA/image_vis/word_clouds/PC%d_best_captions.png" % i)
    #         plt.imshow(im)
    #         plt.title("PC %d" % i)
    #         plt.subplot(20, 2, i*2+2)
    #         im = img.imread("./figures/PCA/image_vis/word_clouds/PC%d_worst_captions.png" % i)
    #         plt.imshow(im)
    #     plt.tight_layout()
    #     plt.savefig("./figures/PCA/image_vis/word_clouds/all_word_clouds.png")

    if args.hclustering_on_pc_proj:
        from scipy.cluster.hierarchy import (
            dendrogram,
            linkage,
            fcluster,
            maxinconsts,
            inconsistent,
        )

        name_modifier = make_name_modifier(args)
        method = "single"
        metric = "euclidean"
        subj = np.arange(1, 9)
        for s in subj:
            p = np.load(
                "%s/output/pca/%s/%s/subj%02d/pca_projections.npy"
                % (args.output_root, args.model, name_modifier, s)
            ).T
            subj_mask = np.load(
                "%s/output/pca/%s/%s/pca_voxels/pca_voxels_subj%02d.npy"
                % (args.output_root, args.model, name_modifier, s)
            )
            print(sum(subj_mask))
            projs = p[subj_mask, :]

            print(projs.shape)
            Z = linkage(projs, method=method, metric=metric, optimal_ordering=False)
            # labelList=range(1, subj_w.shape[1]+1)
            plt.figure()
            dendrogram(
                Z,
                orientation="top",
                distance_sort="descending",
                show_leaf_counts=True,
                truncate_mode="level",
                p=10,
            )
            plt.savefig(
                "figures/PCA/clustering/hclustering_%s_%s_%s_subj%01d.png"
                % (method, metric, name_modifier, s)
            )

    if args.analyze_PC_images:
        # PC 0 positive images distribution (compared to full 10000 samples): probably have people in it
        from analyze_clip_results import (
            extract_text_activations,
            extract_emb_keywords,
            get_coco_anns,
            get_coco_image,
            get_coco_caps,
        )
        from featureprep.feature_prep import get_preloaded_features
        from pycocotools.coco import COCO

        PCs, name_modifier = get_PCs(args)
        stimulus_list = np.load(
            "%s/output/coco_ID_of_repeats_subj%02d.npy" % (args.output_root, 1)
        )
        activations = get_preloaded_features(
            1,
            stimulus_list,
            "%s" % args.model,
            features_dir="%s/features" % args.output_root,
        )
        annFile_train = "/lab_data/tarrlab/common/datasets/coco_annotations/instances_train2017.json"
        coco_train = COCO(annFile_train)
        # cats = coco_train.loadCats(coco_train.getCatIds())
        # id2cat = {}
        # for cat in cats:
        #     id2cat[cat["id"]] = cat["name"]
        # print(len(id2cat))
        # cat_list = list(id2cat.values())
        from util.model_config import COCO_super_cat as cat_list

        COCO_cat_feat = get_preloaded_features(
            1, stimulus_list, "supcat", features_dir="%s/features" % args.output_root,
        )
        print(COCO_cat_feat.shape)

        # get positive images
        scores = activations.squeeze() @ PCs[0, :]
        # positive_images = stimulus_list[scores>0]
        all_cat_dist = np.sum(COCO_cat_feat, axis=0)
        negative_cat_dist = np.sum(COCO_cat_feat[scores < 0, :], axis=0)

        # plt.figure(figsize=(12, 5))
        # plt.bar(np.arange(len(cat_list)), negative_cat_dist/np.sum(negative_cat_dist), label="PC0 Negatives", alpha=0.5)
        # plt.bar(np.arange(len(cat_list)), all_cat_dist/np.sum(all_cat_dist), label="All", alpha=0.5)
        # plt.xlabel("category")
        # plt.ylabel("proportion")
        # plt.xticks(np.arange(len(cat_list)), cat_list, rotation=90)
        # plt.legend()
        # plt.savefig("figures/PCA/PC0_analysis/cat_dist.png")

        # all_cat_dist_int = np.sum(COCO_cat_feat>0, axis=0)
        negative_cat_dist_int = np.sum(COCO_cat_feat[scores < 0, :] > 0, axis=0)
        positive_cat_dist_int = np.sum(COCO_cat_feat[scores > 0, :] > 0, axis=0)

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        width = 0.3
        plt.gcf().subplots_adjust(bottom=0.25)
        plt.bar(
            np.arange(len(cat_list)),
            positive_cat_dist_int / np.sum(positive_cat_dist_int),
            width=width,
            label="+",
            alpha=0.9,
            color="forestgreen",
        )
        plt.bar(
            np.arange(len(cat_list)) + width,
            negative_cat_dist_int / np.sum(negative_cat_dist_int),
            width=width,
            label="--",
            alpha=0.9,
            color="gold",
        )
        plt.xlabel("Categories", fontsize=18)
        plt.ylabel("Proportions", fontsize=18)
        plt.xticks(
            np.arange(len(cat_list)) + width / 2, cat_list, rotation=90, fontsize=15
        )
        plt.legend(fontsize=20)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)

        plt.savefig("figures/PCA/PC0_analysis/cat_dist_integer.png")

        # # num of labeled objects in two sets of images: basically the same
        # neg_count = np.mean(np.sum(COCO_cat_feat[scores<0, :]>0, axis=1))
        # all_count = np.mean(np.sum(COCO_cat_feat>0, axis=1))

        # print(neg_count)
        # print(all_count)

        # # 1. Images with animate things (or people) that are not high on this PC.
        # # 2. Number of people
        # # 3. Images with people and with or without verbs
        # # 4. Social scene/interaction ROIs/rebacca saxe/find TPOJ?
        # person_idx = np.where(cat_list == "person")[0][0]
        # print(person_idx)

        # pos_image = np.argsort(scores)[::-1]
        # pos_people_idx = list()
        # for idx in pos_image:
        #     if COCO_cat_feat[idx, person_idx] > 0.2:
        #         pos_people_idx.append(idx)
        #     if len(pos_people_idx) == 20:
        #         break

        # pos_people_img_id = stimulus_list[pos_people_idx]

        # # plot images
        # plt.figure()
        # for j, id in enumerate(pos_people_img_id):
        #     plt.subplot(4, 5, j + 1)
        #     I = get_coco_image(id)
        #     plt.axis("off")
        #     plt.imshow(I)
        # plt.tight_layout()
        # plt.savefig("figures/PCA/PC0_analysis/positive_imgs_with_people.png")
        # plt.close()

        # # caption analysis
        # def count_verb_in_captions(img_ids):
        #     import nltk
        #     from nltk.tokenize import word_tokenize
        #     verb_count = 0
        #     num_of_caps = 0
        #     for id in img_ids:
        #         output = get_coco_caps(id)
        #         num_of_caps += len(output)
        #         for caption in output:
        #             words = word_tokenize(caption)
        #             pos_tags = nltk.pos_tag(words)
        #             for tag in pos_tags:
        #                 if "VB" in tag[1]:
        #                     verb_count += 1
        #     return verb_count/num_of_caps

        # negative_img_ids = stimulus_list[np.argsort(scores)[:100]]
        # positive_img_ids = stimulus_list[np.argsort(scores)[::-1][:100]]

        # negative_verb_cnt = count_verb_in_captions(negative_img_ids)
        # positive_verb_cnt = count_verb_in_captions(positive_img_ids)
        # print(negative_verb_cnt)
        # print(positive_verb_cnt)

    if args.uv_vs_pc:
        _, name_modifier = get_PCs(args)
        subj_pc = np.load(
            "%s/output/pca/%s/%s/subj%02d/pca_projections.npy"
            % (args.output_root, args.model, name_modifier, args.subj)
        )
        subj_mask = np.load(
            "%s/output/pca/%s/%s/pca_voxels/pca_voxels_subj%02d.npy"
            % (args.output_root, args.model, name_modifier, args.subj)
        )
        joint_var = load_model_performance(
            model=[
                "resnet50_bottleneck_clip_visual_resnet",
                "clip_visual_resnet_resnet50_bottleneck",
            ],
            output_root=args.output_root,
            subj=args.subj,
            measure="rsq",
        )
        resnet_var = load_model_performance(
            model="resnet50_bottleneck",
            output_root=args.output_root,
            subj=args.subj,
            measure="rsq",
        )
        u_clip = joint_var - resnet_var

        assert len(u_clip[subj_mask]) == len(subj_pc[0, :][subj_mask])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        sns.scatterplot(x=subj_pc[0, :][subj_mask], y=u_clip[subj_mask], alpha=0.7)
        plt.xlim(-1, 1)
        plt.xlabel("Projection onto 1st PC")
        plt.ylabel("Unique Var. of CLIP")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        plt.savefig(
            "figures/PCA/PC0_analysis/pc0_vs_unique_var_subj_%s.png" % args.subj
        )

    # if args.maximize_input_for_cluster:
    #     # verify they are in a patch?
    #     from analyze_clip_results import extract_text_activations, extract_emb_keywords, get_coco_anns, get_coco_image, get_coco_caps
    #     from featureprep.feature_prep import get_preloaded_features

    #     from pycocotools.coco import COCO
    #     annFile_train = "/lab_data/tarrlab/common/datasets/coco_annotations/instances_train2017.json"
    #     # annFile_val = "/lab_data/tarrlab/common/datasets/coco_annotations/instances_val2017.json"
    #     coco_train = COCO(annFile_train)
    #     # coco_val = COCO(annFile_val)

    #     model = "clip"
    #     plotting = False
    #     best_voxel_n = 20000
    #     n_clusters = 4
    #     n_pcs = 3

    #     stimulus_list = np.load(
    #         "%s/output/coco_ID_of_repeats_subj%02d.npy" % (args.output_root, 1)
    #     )

    #     activations = get_preloaded_features(
    #         1,
    #         stimulus_list,
    #         "clip",
    #         features_dir="%s/features" % args.output_root,
    #     )

    #     PCs = np.load(
    #         "%s/output/pca/%s/subj%02d/%s_pca_group_components.npy"
    #         % (args.output_root, model, args.subj, model)
    #     )[:n_pcs,:]
    #     subj_mask = np.load(
    #                 "%s/output/pca/%s/pca_voxels/pca_voxels_subj%02d_best_%d.npy"
    #                 % (args.output_root, model, args.subj, best_voxel_n)
    #             )
    #     PC_val_only = PCs[:, subj_mask]

    #     subj_w = np.load(
    #                 "%s/output/encoding_results/subj%d/weights_%s_whole_brain.npy"
    #                 % (args.output_root, args.subj, model)
    #             )
    #     subj_w = fill_in_nan_voxels(subj_w, args.subj, args.output_root)
    #     masked_weight = subj_w[:, subj_mask]

    #     from sklearn.cluster import KMeans
    #     kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(PC_val_only.T)

    #     max_text = dict()
    #     print(masked_weight.shape)
    #     for c in tqdm(range(n_clusters)):
    #         vox = kmeans.labels_==c
    #         print(np.sum(vox))

    #         #maximize image
    #         scores = np.mean(activations.squeeze() @ masked_weight[:, vox], axis=1)
    #         best_img_ids = stimulus_list[np.argsort(scores)[::-1][:20]]

    #         # plot images
    #         plt.figure()
    #         for j, id in enumerate(best_img_ids):
    #             plt.subplot(4, 5, j + 1)
    #             I = get_coco_image(id)
    #             plt.axis("off")
    #             plt.imshow(I)
    #         plt.tight_layout()
    #         fig_root = "figures/PCA/clustering/%dclusters_maximization" % n_clusters
    #         if not os.path.exists(fig_root):
    #             os.makedirs(fig_root)
    #         plt.savefig("%s/%s_cluster%d_best_images.png" % (fig_root, model, c))

    #         #maximize text
    #         from analyze_clip_results import extract_emb_keywords
    #         with open("%s/output/clip/word_interpretation/1000eng.txt" % args.output_root) as f:
    #             out = f.readlines()
    #         common_words = ["photo of " + w[:-1] for w in out]
    #         activations = np.load(
    #             "%s/output/clip/word_interpretation/1000eng_activation.npy"
    #             % args.output_root
    #         )

    #         b, w = extract_emb_keywords(masked_weight[:, vox], activations, common_words)
    #         max_text[c] = [b, w]
    #     pickle.dump(
    #         max_text,
    #         open(
    #             "%s/output/pca/%s/subj%02d/max_text.json"
    #             % (args.output_root, model, args.subj),
    #             "wb",
    #         ),
    #     )
    #     print(max_text)

    # from numpy.linalg import svd
    # w = load_weight_matrix_from_subjs_for_pca(args).T
    # print(w.shape)

    # w0 = w - w.mean(axis=0, keepdims=True)

    # U, S, Vt = svd(w0, full_matrices=False)
    # proj1 = w0.dot(Vt[:20, :].T)

    # pca = PCA(n_components=20, svd_solver="full")
    # proj2 = pca.fit_transform(w0)

    # print(((pca.components_ - Vt[:20, :])**2).mean())
    # print(((proj1 - proj2)**2).mean())

    # print(np.sum(proj1[:, 0]**2))
    # print(np.sum(proj1[:, 1]**2))
    # print(np.sum(proj2[:, 0]**2))
    # print(np.sum(proj2[:, 1]**2))

    # print(S[:20])
