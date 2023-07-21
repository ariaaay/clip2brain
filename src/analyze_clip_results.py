import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import os
import argparse

import pickle
import matplotlib as mpl

import pandas as pd
import seaborn as sns
import nibabel as nib
import numpy as np

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
    compute_sample_performance,
)
from util.coco_utils import get_coco_image
from util.model_config import *

# device = "cuda" if torch.cuda.is_available() else "cpu"

from pycocotools.coco import COCO


def plot_image_wise_performance(model1, model2, masking="sig", measure="corrs"):
    subjs = np.arange(1, 9)
    plt.figure()

    for subj in subjs:
        sample_corr1 = compute_sample_performance(
            model=model1,
            subj=subj,
            output_dir=args.output_root,
            masking=masking,
            measure=measure,
        )
        sample_corr2 = compute_sample_performance(
            model=model2,
            subj=subj,
            output_dir=args.output_root,
            masking=masking,
            measure=measure,
        )
        plt.subplot(2, 4, subj)
        plt.scatter(sample_corr1, sample_corr2, alpha=0.3)
        plt.plot([-0.1, 1], [-0.1, 1], "r")
        plt.xlabel(model1)
        plt.ylabel(model2)

    plt.savefig(
        "figures/CLIP/image_wise_performance/%s_vs_%s_samplewise_%s_%s_all_subjs.png"
        % (model1, model2, measure, masking)
    )


def find_corner_images(
    model1, model2, subj, upper_thr=0.5, lower_thr=0.03, masking="sig", measure="corrs"
):

    """
    Returns images that differ the most in performances across two models.
    """
    sp1 = compute_sample_performance(
        model=model1,
        subj=subj,
        output_dir=args.output_root,
        masking=masking,
        measure=measure,
    )
    sp2 = compute_sample_performance(
        model=model2,
        subj=subj,
        output_dir=args.output_root,
        masking=masking,
        measure=measure,
    )
    diff = sp1 - sp2
    indexes = np.argsort(
        diff
    )  # from where model 2 does the best to where model 1 does the best
    br = indexes[:20]  # model2 > 1
    tl = indexes[::-1][:20]  # model1 > 2

    best1 = np.argsort(sp1)[::-1][:20]
    best2 = np.argsort(sp2)[::-1][:20]
    worst1 = np.argsort(sp1)[:20]
    worst2 = np.argsort(sp2)[:20]

    tr = [idx for idx in best1 if idx in best2]
    bl = [idx for idx in worst1 if idx in worst2]
    corner_idxes = [br, tl, tr, bl]

    test_image_id, _ = extract_test_image_ids(subj=1)
    corner_image_ids = [test_image_id[idx] for idx in corner_idxes]
    with open(
        "%s/output/clip/%s_vs_%s_corner_image_ids_%s_sample_%s_%s.npy"
        % (args.output_root, model1, model2, masking, measure, subj),
        "wb",
    ) as f:
        pickle.dump(corner_image_ids, f)

    image_labels = [
        "%s+%s-" % (model1, model2),
        "%s+%s-" % (model2, model1),
        "%s+%s+" % (model1, model2),
        "%s-%s-" % (model1, model2),
    ]

    for i, idx in enumerate(corner_image_ids):
        plt.figure()
        for j, id in enumerate(idx[:16]):
            # print(id)
            plt.subplot(4, 4, j + 1)
            I = get_coco_image(id)
            plt.axis("off")
            plt.imshow(I)
        # plt.title(image_labels[i])
        plt.tight_layout()
        plt.savefig(
            "figures/CLIP/corner_images/sample_%s_images_%s_%s_%s.png"
            % (measure, image_labels[i], masking, subj)
        )
        plt.close()


def make_roi_df(roi_names, subjs, update=False):
    """
    Construct dataframes based on voxel performances.
    """

    if update:
        df = pd.read_csv("%s/output/clip/performance_by_roi_df.csv" % args.output_root)
    else:
        df = pd.DataFrame()

    for subj in subjs:
        try:
            subj_df = pd.read_csv(
                "%s/output/clip/performance_by_roi_df_subj%02d.csv"
                % (args.output_root, subj)
            )
        except FileNotFoundError:
            subj_df = pd.DataFrame(
                columns=[
                    "voxel_idx",
                    "var_clip",
                    "var_resnet",
                    "uv_clip",
                    "uv_resnet",
                    "uv_diff",
                    # "uv_diff_nc",
                    "joint",
                    "subj",
                    "nc",
                ]
                + roi_names
            )

            joint_var = load_model_performance(
                model=[
                    "resnet50_bottleneck_clip_visual_resnet",
                    "clip_visual_resnet_resnet50_bottleneck",
                ],
                output_root=args.output_root,
                subj=subj,
                measure="rsq",
            )
            clip_var = load_model_performance(
                model="clip_visual_resnet",
                output_root=args.output_root,
                subj=subj,
                measure="rsq",
            )
            resnet_var = load_model_performance(
                model="resnet50_bottleneck",
                output_root=args.output_root,
                subj=subj,
                measure="rsq",
            )
            nc = np.load(
                "%s/output/noise_ceiling/subj%01d/noise_ceiling_1d_subj%02d.npy"
                % (args.output_root, subj, subj)
            )

            u_clip = joint_var - resnet_var
            u_resnet = joint_var - clip_var

            for i in tqdm(range(len(joint_var))):
                vd = dict()
                vd["voxel_idx"] = i
                vd["var_clip"] = clip_var[i]
                vd["var_resnet"] = resnet_var[i]
                vd["uv_clip"] = u_clip[i]
                vd["uv_resnet"] = u_resnet[i]
                vd["uv_diff"] = u_clip[i] - u_resnet[i]
                # vd["uv_diff_nc"] = u_clip[i] / (nc[i]/100) - u_resnet[i] / (nc[i]/100)
                vd["joint"] = joint_var[i]
                vd["nc"] = nc[i]
                vd["subj"] = subj
                subj_df = subj_df.append(vd, ignore_index=True)

            cortical_mask = np.load(
                "%s/output/voxels_masks/subj%d/cortical_mask_subj%02d.npy"
                % (args.output_root, subj, subj)
            )

            for roi_name in roi_names:
                if roi_name == "language":
                    lang_ROI = np.load(
                        "%s/output/voxels_masks/language_ROIs.npy" % args.output_root,
                        allow_pickle=True,
                    ).item()
                    roi_volume = lang_ROI["subj%02d" % subj]
                    roi_volume = np.swapaxes(roi_volume, 0, 2)

                else:
                    roi = nib.load(
                        "%s/subj%02d/func1pt8mm/roi/%s.nii.gz"
                        % (PPDATA_PATH, subj, roi_name)
                    )
                    roi_volume = roi.get_fdata()
                roi_vals = roi_volume[cortical_mask]
                roi_label_dict = roi_name_dict[roi_name]
                roi_label_dict[-1] = "non-cortical"
                roi_label_dict["-1"] = "non-cortical"
                try:
                    roi_labels = [roi_label_dict[int(i)] for i in roi_vals]
                except KeyError:
                    roi_labels = [roi_label_dict[str(int(i))] for i in roi_vals]
                # print(np.array(list(df["voxel_idx"])).astype(int))
                subj_df[roi_name] = np.array(roi_labels)[
                    np.array(list(subj_df["voxel_idx"])).astype(int)
                ]

            subj_df.to_csv(
                "%s/output/clip/performance_by_roi_df_subj%02d.csv"
                % (args.output_root, subj)
            )
        df = pd.concat([df, subj_df])

    df.to_csv("%s/output/clip/performance_by_roi_df.csv" % args.output_root)
    return df


def process_bootstrap_result_for_uv(subj, model1, model2):
    joint_rsq = np.load(
        "%s/output/bootstrap/subj%01d/rsq_dist_%s_%s_whole_brain.npy"
        % (args.output_root, subj, model1, model2)
    )
    m1_rsq = np.load(
        "%s/output/bootstrap/subj%01d/rsq_dist_%s_whole_brain.npy"
        % (args.output_root, subj, model1)
    )
    m2_rsq = np.load(
        "%s/output/bootstrap/subj%01d/rsq_dist_%s_whole_brain.npy"
        % (args.output_root, subj, model2)
    )

    fdr_p = fdr_correct_p(m1_rsq)
    # print(np.sum(fdr_p[1] < 0.05))
    np.save(
        "%s/output/ci_threshold/%s_fdr_p_subj%01d.npy"
        % (args.output_root, model1, subj),
        fdr_p,
    )

    fdr_p = fdr_correct_p(m2_rsq)
    # print(np.sum(fdr_p[1] < 0.05))
    np.save(
        "%s/output/ci_threshold/%s_fdr_p_subj%01d.npy"
        % (args.output_root, model2, subj),
        fdr_p,
    )

    m1_unique_var = joint_rsq - m2_rsq
    m2_unique_var = joint_rsq - m1_rsq
    del joint_rsq
    del m1_rsq
    del m2_rsq
    fdr_p = fdr_correct_p(m1_unique_var)
    # print(np.sum(fdr_p[1] < 0.05))
    np.save(
        "%s/output/ci_threshold/%s-%s_unique_var_fdr_p_subj%01d.npy"
        % (args.output_root, model1, model2, subj),
        fdr_p,
    )

    fdr_p = fdr_correct_p(m2_unique_var)
    # print(np.sum(fdr_p[1] < 0.05))
    np.save(
        "%s/output/ci_threshold/%s-%s_unique_var_fdr_p_subj%01d.npy"
        % (args.output_root, model2, model1, subj),
        fdr_p,
    )


def compute_ci_cutoff(n, ci=0.95):
    ci_ends = np.array([0.0 + (1 - ci) / 2.0, 1 - (1 - ci) / 2.0])
    ci_ind = (ci_ends * n).astype(np.int32)
    return ci_ind


def plot_model_comparison_on_ROI(roi_regions, roi, model1, model2):
    roi_dict = roi_name_dict[roi_regions]
    roi_val = list(roi_dict.keys())[list(roi_dict.values()).index(roi)]
    # print(roi_val)

    m1_us, m1_mps, m2_us, m2_mps = [], [], [], []
    # for s in np.arange(1, 9):
    for s in [1, 2, 5, 7]:
        roi_mask = np.load(
            "%s/output/voxels_masks/subj%01d/roi_1d_mask_subj%02d_%s.npy"
            % (args.output_root, s, s, roi_regions)
        )
        roi_mask = roi_mask == roi_val

        m1_rsq = load_model_performance(
            model1, output_root=args.output_root, subj=s, measure="rsq"
        )
        m2_rsq = load_model_performance(
            model2, output_root=args.output_root, subj=s, measure="rsq"
        )
        joint_rsq = load_model_performance(
            "%s_%s" % (model1, model2),
            output_root=args.output_root,
            subj=s,
            measure="rsq",
        )

        m1_u = joint_rsq - m2_rsq
        m2_u = joint_rsq - m1_rsq
        m1_us.append(m1_u[roi_mask])
        m2_us.append(m2_u[roi_mask])
        m1_mps.append(m1_rsq[roi_mask])
        m2_mps.append(m2_rsq[roi_mask])

    m1_ru = np.hstack(m1_us)
    m2_ru = np.hstack(m2_us)
    m1_mps = np.hstack(m1_mps)
    m2_mps = np.hstack(m2_mps)

    # print(m1_ru.shape)
    # print(m1_mps.shape)

    fig, ax = plt.subplots()
    h = ax.hist2d(
        m1_ru,
        m2_ru,
        bins=100,
        norm=mpl.colors.LogNorm(),
        cmap="YlOrRd",
    )
    fig.colorbar(h[3], ax=ax)
    plt.xlabel(model_label[model1])
    plt.ylabel(model_label[model2])
    ax.set_aspect(1)
    ax.set_xlim(-0.05, 0.1)
    ax.set_ylim(-0.05, 0.1)
    ax.spines[["right", "top"]].set_visible(False)
    ax.plot([-0.02, 0.06], [-0.02, 0.06], linewidth=1, color="red")

    fig.savefig(
        "figures/CLIP/voxel_wise_performance/unique_var_hist2d_%s_v_%s_in_%s.png"
        % (model1, model2, roi)
    )

    # plt.figure(figsize=(5, 5))
    # plt.scatter(m1_ru, m2_ru, alpha=0.3)
    # plt.xlabel(model1)
    # plt.ylabel(model2)
    # plt.title("Unique Variances")
    # plt.plot([-0.08, 0.1], [-0.08, 0.1], linewidth=1, color="red")
    # plt.xlabel("YFCC SLIP")
    # plt.ylabel("YFCC simCLR")
    # plt.savefig(
    #     "figures/CLIP/voxel_wise_performance/unique_var_%s_v_%s_in_%s.png"
    #     % (model1, model2, roi)
    # )

    # plt.figure()
    # plt.scatter(m1_mps, m2_mps, alpha=0.3)
    # plt.xlabel(model1)
    # plt.ylabel(model2)
    # plt.title("Model Performance")
    # plt.plot([-0.08, 0.95], [-0.08, 0.95], linewidth=1, color="red")
    # plt.xlabel("YFCC SLIP")
    # plt.ylabel("YFCC simCLR")
    # plt.savefig(
    #     "figures/CLIP/voxel_wise_performance/%s_v_%s_in_%s.png"
    #     % (model1, model2, roi)
    # )


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
    parser.add_argument(
        "--plot_voxel_wise_performance", default=False, action="store_true"
    )
    parser.add_argument(
        "--plot_image_wise_performance", default=False, action="store_true"
    )
    parser.add_argument(
        "--performance_analysis_by_roi", default=False, action="store_true"
    )
    parser.add_argument(
        "--performance_analysis_by_roi_subset", default=False, action="store_true"
    )
    parser.add_argument("--rerun_df", default=False, action="store_true")
    parser.add_argument("--group_analysis_by_roi", default=False, action="store_true")
    parser.add_argument("--summary_statistics", default=False, action="store_true")
    parser.add_argument("--clip_rsq_across_subject", default=False, action="store_true")
    parser.add_argument(
        "--process_bootstrap_results", default=False, action="store_true"
    )
    parser.add_argument("--cross_model_comparison", default=False, action="store_true")
    parser.add_argument(
        "--roi_voxel_analysis_between_models", default=False, action="store_true"
    )
    parser.add_argument("--nc_scatter_plot", default=False, action="store_true")
    parser.add_argument("--mask", default=False, action="store_true")
    parser.add_argument("--roi_value", default=0, type=int)
    args = parser.parse_args()

    import configparser

    config = configparser.ConfigParser()
    config.read("config.cfg")
    PPDATA_PATH = config["NSD"]["PPdataPath"]

    if args.process_bootstrap_results:
        # for subj in np.arange(1,9):
        # process_bootstrap_result_for_uv(subj, "clip_visual_resnet", "resnet50_bottleneck")
        for subj in [1, 2, 5, 7]:
            # process_bootstrap_result_for_uv(subj, "YFCC_slip", "YFCC_simclr")
            # process_bootstrap_result_for_uv(subj, "laion2b_clip","laion400m_clip")
            process_bootstrap_result_for_uv(subj, "clip", "laion400m_clip")

    if args.plot_voxel_wise_performance:
        model1 = "convnet_res50"
        model2 = "clip_visual_resnet"
        corr_i = load_model_performance(model1, None, args.output_root, subj=args.subj)
        corr_j = load_model_performance(model2, None, args.output_root, subj=args.subj)

        if args.roi is not None:
            colors = np.load(
                "%s/output/voxels_masks/subj%01d/roi_1d_mask_subj%02d_%s.npy"
                % (args.output_root, args.subj, args.subj, args.roi)
            )
            if args.mask:
                mask = colors > 0
                plt.figure(figsize=(7, 7))
            else:
                mask = colors > -1
                plt.figure(figsize=(30, 15))
            # Plotting text performance vs image performances

            plt.scatter(corr_i[mask], corr_j[mask], alpha=0.07, c=colors[mask])
        else:
            plt.scatter(corr_i, corr_j, alpha=0.02)

        plt.plot([-0.1, 1], [-0.1, 1], "r")
        plt.xlabel(model1)
        plt.ylabel(model2)
        plt.savefig(
            "figures/CLIP/voxel_wise_performance/%s_vs_%s_acc_%s.png"
            % (model1, model2, args.roi)
        )

    if args.plot_image_wise_performance:
        # scatter plot by images
        import skimage.io as io
        from pycocotools.coco import COCO
        import configparser

        config = configparser.ConfigParser()
        config.read("config.cfg")
        annFile_train = config["COCO"]["AnnFileTrain"]
        annFile_val = config["COCO"]["AnnFileVal"]

        coco_train = COCO(annFile_train)
        coco_val = COCO(annFile_val)

        m1 = "clip"
        m2 = "resnet50_bottleneck"

        plot_image_wise_performance(m1, m2, measure="rsq")

        for subj in np.arange(1, 9):
            find_corner_images("clip", "convnet_res50", subj=subj, measure="rsq")

        roi_list = list(roi_name_dict.keys())
        roi_list = ["floc-faces", "floc-bodies", "prf-visualrois", "floc-places"]

        for roi in roi_list:
            plot_image_wise_performance(m1, m2, masking=roi, measure="rsq")

            for subj in np.arange(1, 9):
                find_corner_images(m1, m2, masking=roi, measure="rsq", subj=subj)

    if args.performance_analysis_by_roi:
        sns.set(style="whitegrid", font_scale=4.5)

        roi_names = list(roi_name_dict.keys())
        if not args.rerun_df:
            df = pd.read_csv(
                "%s/output/clip/performance_by_roi_df.csv" % args.output_root
            )
        else:
            df = make_roi_df(roi_names, subjs=np.arange(1, 9))

        for roi_name in roi_names:
            plt.figure(figsize=(max(len(roi_name_dict[roi_name].values()) / 4, 50), 30))
            ax = sns.boxplot(
                x=roi_name,
                y="uv_diff",
                data=df,
                dodge=True,
                order=list(roi_name_dict[roi_name].values()),
            )
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            plt.ylabel("Difference in Unique Var.")
            plt.savefig("figures/CLIP/performances_by_roi/uv_diff_%s.png" % roi_name)

        # for roi_name in roi_names:
        #     plt.figure(figsize=(max(len(roi_name_dict[roi_name].values()) / 4, 50), 30))
        #     ax = sns.boxplot(
        #         x=roi_name,
        #         y="uv_diff",
        #         data=df,
        #         dodge=True,
        #         order=list(roi_name_dict[roi_name].values()),
        #     )
        #     ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        #     plt.ylabel("Difference in Unique Var. (NC)")
        #     plt.savefig("figures/CLIP/performances_by_roi/uv_nc_diff_%s.png" % roi_name)

    if args.performance_analysis_by_roi_subset:
        roa_list = [
            ("prf-visualrois", "V1v"),
            ("prf-visualrois", "h4v"),
            ("floc-places", "RSC"),
            ("floc-places", "PPA"),
            ("floc-places", "OPA"),
            ("floc-bodies", "EBA"),
            ("floc-faces", "FFA-1"),
            ("floc-faces", "FFA-2"),
            ("HCP_MMP1", "TPOJ1"),
            ("HCP_MMP1", "TPOJ2"),
            ("HCP_MMP1", "TPOJ3"),
            ("language", "AG"),
        ]
        axis_labels = [v for _, v in roa_list]
        df = pd.read_csv("%s/output/clip/performance_by_roi_df.csv" % args.output_root)
        new_df = pd.DataFrame()
        for i, (roi_name, roi_lab) in enumerate(roa_list):
            roi_df = df[df[roi_name] == roi_lab].copy()
            roi_df["roi_labels"] = roi_lab
            roi_df["roi_names"] = roi_name
            new_df = pd.concat((new_df, roi_df))

        # plt.figure(figsize=(12, 5))
        # ax = sns.boxplot(
        #     x="roi_labels",
        #     y="uv_diff",
        #     hue="roi_names",
        #     data=new_df,
        #     dodge=False,
        #     order=axis_labels,
        # )
        # ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        # plt.ylabel("Difference in Unique Var. (NC)")
        # plt.xlabel("ROIs")
        # plt.legend([],[], frameon=False)
        # plt.savefig("figures/CLIP/performances_by_roi/uv_diff_roi_subset.png")

        # plt.figure(figsize=(12, 5))
        # ax = sns.boxplot(
        #     x="roi_labels",
        #     y="uv_diff_nc",
        #     hue="roi_names",
        #     data=new_df,
        #     dodge=False,
        #     order=axis_labels,
        # )
        # ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        # plt.xlabel("ROIs")
        # plt.ylabel("Difference in Unique Var. (NC)")
        # plt.legend([],[], frameon=False)
        # plt.savefig("figures/CLIP/performances_by_roi/uv_nc_diff_roi_subset.png")

        # plt.figure()
        # sns.relplot(x="uv_resnet", y="uv_clip", data=new_df, alpha=0.5)
        # plt.plot([-0.08, 0.3], [-0.08, 0.3], linewidth=1, color="red")
        # plt.ylabel("CLIP")
        # plt.xlabel("ResNet")
        # plt.savefig("figures/CLIP/performances_by_roi/unique_var_roi_subset.png")

        # plt.figure()
        # sns.relplot(x="var_resnet", y="var_clip", data=new_df, alpha=0.5)
        # plt.plot([-0.05, 0.8], [-0.05, 0.8], linewidth=1, color="red")
        # plt.ylabel("CLIP")
        # plt.xlabel("ResNet")
        # plt.savefig("figures/CLIP/performances_by_roi/var_roi_subset.png")

        fig, axes = plt.subplots(2, 6, sharex=True, sharey=True, figsize=(15, 6))
        for roi, ax in zip(axis_labels, axes.T.flatten()):
            # plt.subplot(3, 4, i+1)
            sns.scatterplot(
                x="uv_resnet",
                y="uv_clip",
                data=new_df[new_df["roi_labels"] == roi],
                alpha=0.5,
                ax=ax,
            )
            sns.lineplot([-0.08, 0.25], [-0.08, 0.25], linewidth=1, color="red", ax=ax)
            ax.set_title(roi)
            ax.set(xlabel=None)
            ax.set(ylabel=None)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            # plt.gca().title.set_text(roi)
        fig.supylabel("Unique Var. of " + r"$ResNet_{CLIP}$", size=18)
        fig.supxlabel("Unique Var. of " + r"$ResNet_{ImageNet}$", size=18)

        plt.tight_layout()
        plt.savefig(
            "figures/CLIP/performances_by_roi/unique_var_scatterplot_by_roi.png"
        )

        fig.supylabel("Unique Var. of " + r"$ResNet_{CLIP}$", size=20)
        fig.supxlabel("Unique Var. of " + r"$ResNet_{ImageNet}$", size=20)
        plt.savefig(
            "figures/CLIP/performances_by_roi/unique_var_scatterplot_by_roi_poster.png"
        )

    if args.group_analysis_by_roi:
        from scipy.stats import ttest_rel
        from util.util import ztransform

        roa_list = [
            ("floc-bodies", "EBA"),
            ("floc-faces", "FFA-1"),
            ("floc-places", "RSC"),
            ("floc-words", "VWFA-1"),
            ("HCP_MMP1", "MST"),
            ("HCP_MMP1", "MT"),
            ("HCP_MMP1", "PH"),
            ("HCP_MMP1", "TPOJ1"),
            ("HCP_MMP1", "TPOJ2"),
            ("HCP_MMP1", "TPOJ3"),
            ("HCP_MMP1", "PGp"),
            ("HCP_MMP1", "V4t"),
            ("HCP_MMP1", "FST"),
            ("Kastner2015", "TO1"),
            ("Kastner2015", "TO2"),
            ("language", "AG"),
            ("language", "ATL"),
            ("prf-visualrois", "V1v"),
        ]

        # roa_list = []
        # roi_names = list(roi_name_dict.keys())
        # for roi_name in roi_names:
        #     if df[roi]

        df = pd.read_csv("%s/output/clip/performance_by_roi_df.csv" % args.output_root)
        subjs = np.arange(1, 9)
        roi_by_subj_mean_clip = np.zeros((8, len(roa_list)))
        roi_by_subj_mean_resnet = np.zeros((8, len(roa_list)))
        for s, subj in enumerate(subjs):
            nc = np.load(
                "%s/output/noise_ceiling/subj%01d/noise_ceiling_1d_subj%02d.npy"
                % (args.output_root, subj, subj)
            )
            varc = df[df["subj"] == subj]["var_clip"] / (nc[nc >= 10] / 100)
            varr = df[df["subj"] == subj]["var_resnet"] / (nc[nc >= 10] / 100)
            tmp_c = ztransform(varc)
            tmp_r = ztransform(varr)

            means_c, means_r = [], []
            for i, (roi_name, roi_lab) in enumerate(roa_list):
                roiv = df[roi_name] == roi_lab
                roi_by_subj_mean_clip[s, i] = np.mean(tmp_c[roiv])
                roi_by_subj_mean_resnet[s, i] = np.mean(tmp_r[roiv])

        stats = ttest_rel(
            roi_by_subj_mean_clip,
            roi_by_subj_mean_resnet,
            axis=0,
            nan_policy="propagate",
            alternative="two-sided",
        )
        print(stats)
        results = {}
        for i, r in enumerate(roa_list):
            results[r] = (stats[0][i], stats[1][i])
        for k, v in results.items():
            print(k, v)
        # print(roa_list)

    if args.summary_statistics:
        roi_names = list(roi_name_dict.keys())
        df = pd.read_csv(
            "%s/output/clip/performance_by_roi_df_nc_corrected.csv" % args.output_root
        )
        for roi_name in roi_names:
            sns.set(style="whitegrid", font_scale=4.5)
            plt.figure(figsize=(50, 20))
            ax = sns.boxplot(
                x=roi_name,
                y="var_resnet",
                data=df,
                dodge=True,
                order=list(roi_name_dict[roi_name].values()),
            )
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            plt.savefig("figures/CLIP/performances_by_roi/var_resnet_%s.png" % roi_name)

            plt.figure(figsize=(50, 20))
            ax = sns.boxplot(
                x=roi_name,
                y="var_clip",
                data=df,
                dodge=True,
                order=list(roi_name_dict[roi_name].values()),
            )

            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            plt.savefig("figures/CLIP/performances_by_roi/var_clip_%s.png" % roi_name)

    if args.clip_rsq_across_subject:
        means = []
        for subj in range(8):
            subj_var = clip_var = load_model_performance(
                model="clip", output_root=args.output_root, subj=subj + 1, measure="rsq"
            )
            nc = np.load(
                "%s/output/noise_ceiling/subj%01d/noise_ceiling_1d_subj%02d.npy"
                % (args.output_root, subj + 1, subj + 1)
            )
            means.append(
                np.mean(subj_var / (nc / 100), where=np.isfinite(subj_var / (nc / 100)))
            )
        print(means)

    if args.nc_scatter_plot:
        # nc_array, var_array = [], []
        # plt.figure(figsize=(10, 10))
        # for subj in np.arange(8) + 1:
        #     clip_var = load_model_performance(
        #         model="clip",
        #         output_root=args.output_root,
        #         subj=subj,
        #         measure="rsq",
        #     )
        #     nc = np.load(
        #             "%s/output/noise_ceiling/subj%01d/noise_ceiling_1d_subj%02d.npy"
        #             % (args.output_root, subj, subj)
        #         ) / 100
        #     nc_array += list(nc)
        #     var_array += list(clip_var)

        #     plt.subplot(4,2, subj)
        #     sns.scatterplot(x=nc, y=clip_var, alpha=0.5, size=0.5)
        #     sns.lineplot([-0.05, 1.05], [-0.05, 1.05], linewidth=1, color="red")
        #     plt.title("subj %d" % subj)
        #     plt.ylabel("Model performance")
        #     plt.xlabel("Noise ceiling")
        # plt.tight_layout()
        # plt.savefig("figures/CLIP/var_clip_vs_nc_per_subj.png")

        # plt.figure()
        # sns.scatterplot(x=nc_array, y=var_array, alpha=0.5)
        # sns.lineplot([-0.05, 1.05], [-0.05, 1.05], linewidth=1, color="red")
        # plt.savefig("figures/CLIP/var_clip_vs_nc.png")

        df = pd.read_csv("%s/output/clip/performance_by_roi_df.csv" % args.output_root)
        df["nc"] = df["nc"] / 100
        # percentiles = [75, 90, 95, 99]
        # markers = ["*", "+", "x", "o"]

        # sns.set_theme(style="whitegrid")
        # plt.figure(figsize=(5,10))
        # labels = ["(0, 0.1]", "(0.1, 0.2]", "(0.2, 0.3]", "(0.3, 0.4]", "(0.4, 0.5]", "(0.5, 0.6]", "(0.6, 0.7]", "(0.7, 0.8]", "(0.8, 0.9]"]
        # df5['nc_bins'] = pd.cut(df5['nc'], 9, precision=1, labels=labels)
        # df5['perc_nc'] = df5["var_clip"] / df5["nc"]

        # # df5 = df5[df5["nc"]>0.1]

        # # sns.relplot(x="nc", y="perc_nc", data=df5, alpha=0.5)
        # # sns.lineplot([-0.05, 1.05], [-0.05, 1.05], linewidth=1, color="red", label="Noise Ceiling")

        # sns.boxplot(data=df5, x="nc_bins", y="perc_nc")
        # plt.ylim((-0.1, 1.1))
        # plt.xticks(rotation = 45)

        # # n = len(df5["nc"])
        # # for i, p in enumerate(percentiles):
        # #     px = np.percentile(df["nc"], p)
        # #     py = np.percentile(df["var_clip"], p)
        # #     # print(px, py)
        # #     plt.scatter(x=px, y=py, marker=markers[i], s=100, color="red", label="%d%% (n=%d)" % (p, (100-p)*n/100))
        # plt.xlabel("Noise Ceiling")
        # plt.ylabel("Model Performance as % in noise ceiling")
        # # plt.legend()
        # plt.savefig("figures/CLIP/var_clip_vs_nc_subj5.png")
        import matplotlib as mpl
        import matplotlib.pylab as plt

        # PLOTTING SUBJ5
        plt.figure()
        df5 = df[df["subj"] == 5]
        sns.lineplot(
            [-0.05, 1], [-0.05, 1], linewidth=2, color="red", label="noise ceiling"
        )
        sns.lineplot(
            [-0.05, 1],
            [-0.05, 0.85],
            linewidth=2,
            color="orange",
            linestyle="--",
            label="85% noise ceiling",
        )

        plt.hist2d(
            df5["nc"],
            df5["var_clip"],
            bins=100,
            norm=mpl.colors.LogNorm(),
            cmap="magma",
        )
        plt.colorbar()
        plt.xlabel("Noise Ceiling")
        plt.ylabel("Model Performance $(R^2)$")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True)
        plt.savefig("figures/CLIP/var_clip_vs_nc_subj5_2dhist.png")

        plt.figure()
        sns.lineplot(
            [-0.05, 1], [-0.05, 1], linewidth=2, color="red", label="noise ceiling"
        )
        sns.lineplot(
            [-0.05, 1],
            [-0.05, 0.85],
            linewidth=2,
            color="orange",
            linestyle="--",
            label="85% noise ceiling",
        )

        plt.hist2d(df5["nc"], df5["var_resnet"], bins=100, norm=mpl.colors.LogNorm())

        plt.colorbar()
        plt.xlabel("Noise Ceiling")
        plt.ylabel("Model Performance $(R^2)$")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True)
        plt.savefig("figures/CLIP/var_rn_vs_nc_subj5_2dhist.png")

        plt.figure(figsize=(5, 8))
        plt.subplot(2, 1, 1)
        sns.lineplot([-0.05, 1], [-0.05, 1], linewidth=1, color="red")

        plt.hist2d(
            df5["var_resnet"], df5["var_clip"], bins=100, norm=mpl.colors.LogNorm()
        )

        plt.colorbar()
        plt.xlabel("$ResNet_{ImageNet}$", size=20)
        plt.ylabel("$ResNet_{CLIP}$", size=20)
        plt.xlim(-0.05, 0.9)
        plt.ylim(-0.05, 0.9)
        plt.grid(True)
        plt.title("Model Performance $(R^2)$", size=24)

        plt.subplot(2, 1, 2)
        sns.lineplot([-0.1, 1], [-0.1, 1], linewidth=1, color="red")
        plt.hist2d(
            df5["uv_resnet"],
            df5["uv_clip"],
            bins=100,
            norm=mpl.colors.LogNorm(),
            cmap="magma",
        )

        plt.colorbar()
        plt.xlabel("$ResNet_{ImageNet}$", size=20)
        plt.ylabel("$ResNet_{CLIP}$", size=20)
        plt.xlim(-0.15, 0.4)
        plt.ylim(-0.15, 0.4)
        plt.grid(True)
        plt.title("Unique Variance", size=24)
        plt.tight_layout()
        plt.savefig("figures/CLIP/var_rn_vs_clip_subj5_2dhist.png", dpi=300)

        # PLOTTING ALL SUBJECTS
        fig, axes = plt.subplots(4, 2, sharex=True, sharey=True, figsize=(10, 15))
        for s, ax in zip(np.arange(8) + 1, axes.T.flatten()):
            dfs = df[df["subj"] == s]
            h = ax.hist2d(
                dfs["nc"],
                dfs["var_clip"],
                bins=100,
                norm=mpl.colors.LogNorm(),
                cmap="magma",
            )

            sns.lineplot(
                [-0.05, 1.05],
                [-0.05, 1.05],
                linewidth=2,
                color="red",
                label="Noise Ceiling",
                ax=ax,
            )
            sns.lineplot(
                [-0.05, 1],
                [-0.05, 0.85],
                linewidth=2,
                color="orange",
                linestyle="--",
                label="85% noise ceiling",
                ax=ax,
            )

            ax.grid()
            fig.colorbar(h[3], ax=ax)
            ax.set_title("subj %d" % s)
            ax.set(xlabel=None)
            ax.set(ylabel=None)
            ax.legend().set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper left")
        fig.supylabel("Model Performance $(R^2)$")
        fig.supxlabel("Noise Ceiling")
        plt.tight_layout()

        plt.savefig("figures/CLIP/var_clip_vs_nc_all_subj.png")

    if args.cross_model_comparison:
        from util.model_config import *

        # plot ROI averaged prediction across models
        df = pd.DataFrame()
        models = [
            "resnet50_bottleneck",
            "YFCC_clip",
            "YFCC_simclr",
            "YFCC_slip",
            "laion400m_clip",
            "clip",
            "laion2b_clip",
        ]
        model_sizes = ["1m", "15m", "15m", "15m", "400m", "400m", "2b"]
        # model_sizes = ["15m", "15m", "15m", "400m", "400m", "2b"]
        model_size_for_plot = {"1m": 100, "15m": 200, "400m": 400, "2b": 600}
        # subjs = [1, 2, 5, 7]
        subjs = np.arange(1, 9)
        # rois = [
        #     "prf-visualrois": 'all'],
        #     "floc-faces",
        #     "floc-places",
        #     "floc-bodies": ,
        #     "HCP": "TPOJ1"
        # ]
        # rois_name = {
        #     "prf-visualrois": "Early Visual",
        #     "floc-bodies": "Body",
        #     "floc-faces": "Face",
        #     "floc-places": "Place",
        # }

        roa_list = [
            ("prf-visualrois", "V1v"),
            ("prf-visualrois", "h4v"),
            ("floc-places", "RSC"),
            ("floc-places", "PPA"),
            ("floc-places", "OPA"),
            ("floc-bodies", "EBA"),
            ("floc-faces", "FFA-1"),
            ("floc-faces", "FFA-2"),
            # ("HCP_MMP1", "TPOJ1"),
            ("HCP_MMP1", "TPOJ2"),
            ("HCP_MMP1", "TPOJ3"),
            # ("language", "AG"),
        ]

        roi_type_name_dict = {
            "prf-visualrois": "EarlyVis",
            "floc-places": "Scene",
            "floc-bodies": "Body",
            "floc-faces": "Face",
            "HCP_MMP1": "TPOJ",
        }

        model_type = {
            "clip": "Lang (OpenAI)",
            "resnet50_bottleneck": "Categories",
            "laion400m_clip": "Lang (Laion)",
            "laion2b_clip": "Lang (Laion)",
            "YFCC_clip": "Lang (YFCC)",
            "YFCC_simclr": "SSL",
            "YFCC_slip": "SSL + Lang (YFCC)",
        }

        # marker_type={
        #     "Lang (OpenAI)": "1",
        #     # "resnet50_bottleneck": "Labels",
        #     "Lang (Laion)": "2",
        #     "Lang (YFCC)": "3",
        #     "SSL": "x",
        #     "SSL + Lang (YFCC)": (8, 2, 0),
        # }

        marker_type = {
            "Lang (OpenAI)": "^",
            "Categories": "X",
            "Lang (Laion)": "v",
            "Lang (YFCC)": "<",
            "SSL": "s",
            "SSL + Lang (YFCC)": "h",
        }

        for i, model in enumerate(tqdm(models)):
            for (roi_type, roi_lab) in roa_list:
                # rsqs = []
                for subj in subjs:
                    rsq = load_model_performance(
                        model, output_root=args.output_root, subj=subj, measure="rsq"
                    )
                    roi_mask = np.load(
                        "%s/output/voxels_masks/subj%01d/roi_1d_mask_subj%02d_%s.npy"
                        % (args.output_root, subj, subj, roi_type)
                    )
                    roi_dict = roi_name_dict[roi_type]
                    roi_val = list(roi_dict.keys())[
                        list(roi_dict.values()).index(roi_lab)
                    ]
                    # print(roi_lab, roi_val)
                    roi_selected_vox = roi_mask == int(roi_val)
                    # print(np.sum(roi_selected_vox))
                    # import pdb; pdb.set_trace()
                    # rsqs.append(np.mean(rsq[roi_selected_vox]))
                    rsq_mean = np.mean(rsq[roi_selected_vox])

                    vd = {}
                    # vd["Regions"] = roi_lab
                    vd["Regions"] = roi_type_name_dict[roi_type]
                    vd["Model"] = model
                    vd["Dataset size"] = model_sizes[i]
                    vd["Model type"] = model_type[model]
                    vd[r"Mean Performance ($R^2$)"] = rsq_mean
                    vd["Subject"] = str(subj)
                    # vd["perf_std"] = np.std(rsqs[roi_selected_vox])
                    #

                    df = df.append(vd, ignore_index=True)

        df.to_csv(
            "%s/output/clip/cross_model_comparison_roi_type.csv" % args.output_root
        )

        import seaborn.objects as so
        from seaborn import axes_style

        (
            so.Plot(
                df,
                x="Regions",
                y=r"Mean Performance ($R^2$)",
                # color="Subject",
                color="Dataset size",
                marker="Model type",
                pointsize="Dataset size",
                # ymin=0,
                # ymax=0.25,
            )
            .add(
                so.Dot(),
                so.Agg(),
                # so.Jitter(.1),
                so.Dodge(),
            )
            # .add(so.Line(color=".2", linewidth=1))
            .add(so.Range(), so.Est(errorbar="se"), so.Dodge())
            .scale(marker=marker_type, pointsize=(12, 7))
            .theme(
                {
                    **axes_style("white"),
                    **{
                        "figure.figsize": (15, 5),
                        "legend.frameon": False,
                        "axes.spines.right": False,
                        "axes.spines.top": False,
                        "axes.labelsize": 17,
                        "xtick.labelsize": 15,
                        "ytick.labelsize": 15,
                        # "axes.grid": True,
                        # "axes.grid.axis": "x",
                    },
                }
            )
            .save("figures/model_comp/roi_type_per_subj.png", bbox_inches="tight")
        )

        # compare to basedline model:
        baseline_model = "resnet50_bottleneck"
        df_baseline = pd.DataFrame()
        for i, model in enumerate(tqdm(models)):
            if model == baseline_model:
                continue
            for (roi_type, roi_lab) in roa_list:
                # rsqs = []
                for subj in subjs:
                    rsq_baseline = load_model_performance(
                        baseline_model,
                        output_root=args.output_root,
                        subj=subj,
                        measure="rsq",
                    )
                    rsq = load_model_performance(
                        model, output_root=args.output_root, subj=subj, measure="rsq"
                    )
                    roi_mask = np.load(
                        "%s/output/voxels_masks/subj%01d/roi_1d_mask_subj%02d_%s.npy"
                        % (args.output_root, subj, subj, roi_type)
                    )
                    roi_dict = roi_name_dict[roi_type]
                    roi_val = list(roi_dict.keys())[
                        list(roi_dict.values()).index(roi_lab)
                    ]
                    # print(roi_lab, roi_val)
                    roi_selected_vox = roi_mask == int(roi_val)
                    # print(np.sum(roi_selected_vox))
                    # import pdb; pdb.set_trace()
                    # rsqs.append(np.mean(rsq[roi_selected_vox]))
                    rsq_mean = np.mean(rsq[roi_selected_vox])
                    rsq_baseline_mean = np.mean(rsq_baseline[roi_selected_vox])
                    rsq_diff = rsq_mean - rsq_baseline_mean

                    vd = {}
                    # vd["Regions"] = roi_lab
                    vd["Regions"] = roi_type_name_dict[roi_type]
                    vd["Model"] = model
                    vd["Dataset size"] = model_sizes[i]
                    vd["Model type"] = model_type[model]
                    vd[r"Mean Performance ($R^2$)"] = rsq_diff
                    vd["Subject"] = str(subj)

                    df_baseline = df_baseline.append(vd, ignore_index=True)

        df_baseline.to_csv(
            "%s/output/clip/cross_model_comparison_roi_type_%s_baseline.csv"
            % (args.output_root, baseline_model)
        )

        import seaborn.objects as so
        from seaborn import axes_style

        (
            so.Plot(
                df_baseline,
                x="Regions",
                y=r"Mean Performance ($R^2$)",
                # color="Subject",
                color="Dataset size",
                marker="Model type",
                pointsize="Dataset size",
                # ymin=0,
                # ymax=0.25,
            )
            .add(
                so.Dot(),
                so.Agg(),
                # so.Jitter(.1),
                so.Dodge(),
            )
            # .add(so.Line(color=".2", linewidth=1))
            .add(so.Range(), so.Est(errorbar="se"), so.Dodge())
            .scale(marker=marker_type, pointsize=(12, 6))
            .theme(
                {
                    **axes_style("white"),
                    **{
                        "figure.figsize": (10, 5),
                        "legend.frameon": False,
                        "axes.spines.right": False,
                        "axes.spines.top": False,
                        "axes.labelsize": 17,
                        "xtick.labelsize": 15,
                        "ytick.labelsize": 15,
                        # "axes.grid": True,
                        # "axes.grid.axis": "x",
                    },
                }
            )
            .save(
                "figures/model_comp/roi_type_per_subj_%s_baseline.png" % baseline_model,
                bbox_inches="tight",
            )
        )

    if args.roi_voxel_analysis_between_models:
        from util.model_config import *

        roi = "EBA"
        roi_regions = "floc-bodies"

        import matplotlib

        matplotlib.rcParams.update({"font.size": 15})

        plot_model_comparison_on_ROI(roi_regions, roi, "YFCC_slip", "YFCC_simclr")

        plot_model_comparison_on_ROI(roi_regions, roi, "laion2b_clip", "laion400m_clip")

        plot_model_comparison_on_ROI(roi_regions, roi, "clip", "laion400m_clip")
