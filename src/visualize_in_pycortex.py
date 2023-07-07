"This scripts visualize prediction performance with pycortex."
import os
from nibabel.volumeutils import working_type
import numpy as np

import argparse
from numpy.core.fromnumeric import nonzero
from tqdm import tqdm

from util.data_util import load_model_performance

OUTPUT_ROOT = "/user_data/yuanw3/project_outputs/NSD"


def project_vals_to_3d(vals, mask):
    all_vals = np.zeros(mask.shape)
    all_vals[mask] = vals
    all_vals = np.swapaxes(all_vals, 0, 2)
    return all_vals


def project_vols_to_mni(subj, vol):
    import cortex

    xfm = "func1pt8_to_anat0pt8_autoFSbbr"
    # template = "func1pt8_to_anat0pt8_autoFSbbr"
    mni_transform = cortex.db.get_mnixfm("subj%02d" % subj, xfm)
    mni_vol = cortex.mni.transform_to_mni(vol, mni_transform)
    mni_data = mni_vol.get_fdata().T
    return mni_data


def load_fdr_mask(OUTPUT_ROOT, model, fdr_mask_name, subj):
    if type(fdr_mask_name) is list:
        sig_mask1 = np.load(
            "%s/output/ci_threshold/%s_fdr_p_subj%01d.npy"
            % (OUTPUT_ROOT, fdr_mask_name[0], subj)
        )[0].astype(bool)
        sig_mask2 = np.load(
            "%s/output/ci_threshold/%s_fdr_p_subj%01d.npy"
            % (OUTPUT_ROOT, fdr_mask_name[1], subj)
        )[0].astype(bool)
        sig_mask = (sig_mask1.astype(int) + sig_mask2.astype(int)).astype(bool)
        return sig_mask
    elif fdr_mask_name is not None:
        model = fdr_mask_name
    try:
        sig_mask = np.load(
            "%s/output/ci_threshold/%s_fdr_p_subj%01d.npy" % (OUTPUT_ROOT, model, subj)
        )[0].astype(bool)
        return sig_mask
    except FileNotFoundError:  # hasn't run the test yet
        return None


def visualize_layerwise_max_corr_results(
    model, layer_num, subj=1, threshold=95, start_with_zero=True, order="asc"
):
    val_array = list()
    for i in range(layer_num):
        if not start_with_zero:  # layer starts with 1
            continue
        val_array.append(
            load_model_performance(
                model="%s_%d" % (model, i), output_root=OUTPUT_ROOT, subj=args.subj
            )
        )

    val_array = np.array(val_array)

    threshold_performance = np.max(val_array, axis=0) * (threshold / 100)
    layeridx = np.zeros(threshold_performance.shape) - 1

    i = 0 if order == "asc" else -1
    for v in tqdm(range(len(threshold_performance))):
        if threshold_performance[v] > 0:
            layeridx[v] = (
                int(np.nonzero(val_array[:, v] >= threshold_performance[v])[0][i]) + 1
            )
            # print(layeridx[i])
    cortical_mask = np.load(
        "%s/output/voxels_masks/subj%d/cortical_mask_subj%02d.npy"
        % (OUTPUT_ROOT, args.subj, args.subj)
    )

    nc = np.load(
        "%s/output/noise_ceiling/subj%01d/noise_ceiling_1d_subj%02d.npy"
        % (OUTPUT_ROOT, subj, subj)
    )

    sig_mask = nc >= 10
    layeridx[~sig_mask] = np.nan

    # # projecting value back to 3D space
    all_vals = project_vals_to_3d(layeridx, cortical_mask)

    layerwise_volume = cortex.Volume(
        all_vals,
        "subj%02d" % subj,
        "func1pt8_to_anat0pt8_autoFSbbr",
        mask=cortex.utils.get_cortical_mask(
            "subj%02d" % subj, "func1pt8_to_anat0pt8_autoFSbbr"
        ),
        vmin=0,
        vmax=layer_num,
    )
    return layerwise_volume


def make_volume(
    subj,
    model=None,
    vals=None,
    model2=None,
    mask_with_significance=False,
    measure="corr",
    noise_corrected=False,
    cmap="hot",
    fdr_mask_name=None,
    vmin=0,
    vmax=None,
):
    if vmax is None:
        if measure == "rsq":
            vmax = 0.6
        else:
            vmax = 1
        if model2 is not None:
            vmax -= 0.3
        if noise_corrected:
            vmax = 0.85
        if measure == "pvalue":
            vmax = 0.06

    mask = cortex.utils.get_cortical_mask(
        "subj%02d" % subj, "func1pt8_to_anat0pt8_autoFSbbr"
    )

    cortical_mask = np.load(
        "%s/output/voxels_masks/subj%d/cortical_mask_subj%02d.npy"
        % (OUTPUT_ROOT, subj, subj)
    )
    nc = np.load(
        "%s/output/noise_ceiling/subj%01d/noise_ceiling_1d_subj%02d.npy"
        % (OUTPUT_ROOT, subj, subj)
    )

    # load correlation scores of cortical voxels
    if vals is None:
        if (
            type(model) == list
        ):  # for different naming convention for variance partitioning (only 1 should exist)
            model_list = model
            for model in model_list:
                try:
                    vals = load_model_performance(
                        model, output_root=OUTPUT_ROOT, subj=subj, measure=measure
                    )
                    break
                except FileNotFoundError:
                    continue
        else:
            vals = load_model_performance(
                model, output_root=OUTPUT_ROOT, subj=subj, measure=measure
            )
        print("model:" + model)

        if model2 is not None:  # for variance paritioning
            vals2 = load_model_performance(
                model2, output_root=OUTPUT_ROOT, subj=subj, measure=measure
            )
            vals = vals - vals2
            print("model2:" + model2)

    if mask_with_significance:
        if args.sig_method == "fdr":
            sig_mask = load_fdr_mask(OUTPUT_ROOT, model, fdr_mask_name, subj)
            if sig_mask is None:
                print("Masking vals with nc only")
                sig_mask = nc >= 10

        elif args.sig_method == "pvalue":
            pvalues = load_model_performance(
                model, output_root=OUTPUT_ROOT, subj=subj, measure="pvalue"
            )
            sig_mask = pvalues <= 0.05

        print(
            "Mask name: "
            + str(fdr_mask_name)
            + ". # of sig voxels: "
            + str(np.sum(sig_mask))
        )
        try:
            vals[~sig_mask] = np.nan
        except IndexError:
            non_zero_mask = np.load(
                "%s/output/voxels_masks/subj%d/nonzero_voxels_subj%02d.npy"
                % (OUTPUT_ROOT, subj, subj)
            )
            print("Masking zero voxels with mask...")
            sig_mask_tmp = np.zeros(non_zero_mask.shape)
            sig_mask_tmp[non_zero_mask] = sig_mask
            sig_mask = sig_mask_tmp.astype(bool)
            vals[~sig_mask] = np.nan

    if (measure == "rsq") and (noise_corrected):
        vals = vals / (nc / 100)
        vals[np.isnan(vals)] = np.nan
    print("max:" + str(max(vals[~np.isnan(vals)])))

    # projecting value back to 3D space
    all_vals = project_vals_to_3d(vals, cortical_mask)

    vol_data = cortex.dataset.Volume(
        all_vals,
        "subj%02d" % subj,
        "func1pt8_to_anat0pt8_autoFSbbr",
        mask=mask,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    return vol_data


def make_pc_volume(subj, vals, vmin=-0.5, vmax=0.5, cmap="BrBG"):
    import cortex

    mask = cortex.utils.get_cortical_mask(
        "subj%02d" % subj, "func1pt8_to_anat0pt8_autoFSbbr"
    )
    cortical_mask = np.load(
        "%s/output/voxels_masks/subj%d/cortical_mask_subj%02d.npy"
        % (OUTPUT_ROOT, subj, subj)
    )

    # projecting value back to 3D space
    all_vals = project_vals_to_3d(vals, cortical_mask)

    vol_data = cortex.dataset.Volume(
        all_vals,
        "subj%02d" % subj,
        "func1pt8_to_anat0pt8_autoFSbbr",
        mask=mask,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    return vol_data


def vis_roi_ind():
    roi_mask = np.load(
        "%s/output/voxels_masks/subj%01d/roi_1d_mask_subj%02d_floc-bodies.npy"
        % (OUTPUT_ROOT, args.subj, args.subj)
    )

    try:  # take out zero voxels
        non_zero_mask = np.load(
            "%s/output/voxels_masks/subj%d/nonzero_voxels_subj%02d.npy"
            % (OUTPUT_ROOT, args.subj, args.subj)
        )
        print("Masking zero voxels...")
        roi_mask = roi_mask[non_zero_mask]
    except FileNotFoundError:
        pass

    mask = roi_mask == 1

    print(str(sum(mask)) + " voxels for optimization")
    vindx = np.arange(sum(mask))
    vals = np.zeros(roi_mask.shape)
    vals[mask] = vindx
    new_vals = np.zeros(non_zero_mask.shape)
    new_vals[non_zero_mask] = vals
    return new_vals


def make_3pc_volume(subj, PCs):
    mask = cortex.utils.get_cortical_mask(
        "subj%02d" % subj, "func1pt8_to_anat0pt8_autoFSbbr"
    )

    cortical_mask = np.load(
        "%s/output/voxels_masks/subj%d/cortical_mask_subj%02d.npy"
        % (OUTPUT_ROOT, subj, subj)
    )

    pc_3d = []
    for i in range(3):
        tmp = PCs[i, :] / np.max(PCs[i, :]) * 255
        # projecting value back to 3D space
        pc_3d.append(project_vals_to_3d(tmp, cortical_mask))

    red = cortex.dataset.Volume(
        pc_3d[0].astype(np.uint8),
        "subj%02d" % subj,
        "func1pt8_to_anat0pt8_autoFSbbr",
        mask=mask,
    )
    green = cortex.dataset.Volume(
        pc_3d[1].astype(np.uint8),
        "subj%02d" % subj,
        "func1pt8_to_anat0pt8_autoFSbbr",
        mask=mask,
    )
    blue = cortex.dataset.Volume(
        pc_3d[2].astype(np.uint8),
        "subj%02d" % subj,
        "func1pt8_to_anat0pt8_autoFSbbr",
        mask=mask,
    )

    vol_data = cortex.dataset.VolumeRGB(
        red,
        green,
        blue,
        "subj%02d" % subj,
        channel1color=(194, 30, 86),
        channel2color=(50, 205, 50),
        channel3color=(30, 144, 255),
    )

    return vol_data


def make_roi_volume(roi_name):
    roi = nib.load("%s/%s.nii.gz" % (ROI_FILE_ROOT, roi_name))
    roi_data = roi.get_fdata()
    roi_data = np.swapaxes(roi_data, 0, 2)

    roi_volume = cortex.dataset.Volume(
        roi_data,
        "subj%02d" % args.subj,
        "func1pt8_to_anat0pt8_autoFSbbr",
        mask=cortex.utils.get_cortical_mask(
            "subj%02d" % args.subj, "func1pt8_to_anat0pt8_autoFSbbr"
        ),
        vmin=0,
        vmax=np.max(roi_data),
    )
    return roi_volume


def show_voxel_diff_in_repr_samples(model1, model2, quad="br"):
    def load_brain_response(model1, model2, quad):
        import glob

        fname = glob.glob(
            "./output/rdm_based_analysis/subj%d/voxel_corr_%s_vs_%s_*%s.npy"
            % (args.subj, model1, model2, quad)
        )
        vals = np.load(fname[0])

        non_zero_mask = np.load(
            "%s/output/voxels_masks/subj%d/nonzero_voxels_subj%02d.npy"
            % (OUTPUT_ROOT, args.subj, args.subj)
        )
        print("Masking zero voxels...")
        tmp = np.zeros(non_zero_mask.shape)
        tmp[non_zero_mask] = vals
        vals = tmp

        all_vals = project_vals_to_3d(vals, cortical_mask)
        return all_vals

    if quad == "br-tl":
        b1 = load_brain_response(model1, model2, "br")
        b2 = load_brain_response(model1, model2, "tl")
        all_vals = b1 - b2
    else:
        all_vals = load_brain_response(model1, model2, quad)

    rdm_volume = cortex.Volume(
        all_vals,
        "subj%02d" % args.subj,
        "func1pt8_to_anat0pt8_autoFSbbr",
        mask=cortex.utils.get_cortical_mask(
            "subj%02d" % args.subj, "func1pt8_to_anat0pt8_autoFSbbr"
        ),
        vmin=-0.1,
        vmax=0.1,
    )
    return rdm_volume


if __name__ == "__main__":
    import cortex
    import nibabel as nib

    parser = argparse.ArgumentParser(description="please specific subject to show")
    parser.add_argument(
        "--subj", type=int, default=1, help="specify which subject to build model on"
    )
    parser.add_argument("--mask_sig", default=False, action="store_true")
    parser.add_argument("--sig_method", default="negtail_fdr")
    parser.add_argument("--alpha", default=0.05)
    parser.add_argument("--show_pcs", default=False, action="store_true")
    parser.add_argument("--show_clustering", default=False, action="store_true")

    parser.add_argument("--on_cluster", action="store_true")
    # parser.add_argument("--with_noise_ceiling", default=False, action="store_true")
    parser.add_argument("--show_more", action="store_true")
    parser.add_argument("--show_repr_sim", action="store_true")
    parser.add_argument("--vis_method", type=str, default="webgl")

    args = parser.parse_args()
    print(args)
    if args.on_cluster:
        ROI_FILE_ROOT = (
            "/lab_data/tarrlab/common/datasets/NSD/nsddata/ppdata/subj%02d/func1pt8mm/roi"
            % args.subj
        )
    else:
        OUTPUT_ROOT = "."
        ROI_FILE_ROOT = "./roi_data/subj%02d" % args.subj

    # visual_roi_volume = make_roi_volume("prf-visualrois")
    # ecc_roi_volume = make_roi_volume("prf-eccrois")
    # place_roi_volume = make_roi_volume("floc-places")
    # face_roi_volume = make_roi_volume("floc-faces")
    # body_roi_volume = make_roi_volume("floc-bodies")
    # # word_roi_volume = make_roi_volume("floc-words")
    # kastner_volume = make_roi_volume("Kastner2015")
    # hcp_volume = make_roi_volume("HCP_MMP1")
    # # sulc_volume = make_roi_volume("corticalsulc")
    # nsd_general_volume = make_roi_volume("nsdgeneral")

    cortical_mask = np.load(
        "%s/output/voxels_masks/subj%d/cortical_mask_subj%02d.npy"
        % (OUTPUT_ROOT, args.subj, args.subj)
    )

    lang_ROI = np.load(
        "./output/voxels_masks/language_ROIs.npy", allow_pickle=True
    ).item()
    language_vals = lang_ROI["subj%02d" % args.subj]
    language_volume = cortex.Volume(
        language_vals,
        "subj%02d" % args.subj,
        "func1pt8_to_anat0pt8_autoFSbbr",
        mask=cortex.utils.get_cortical_mask(
            "subj%02d" % args.subj, "func1pt8_to_anat0pt8_autoFSbbr"
        ),
        vmin=np.min(language_vals),
        vmax=np.max(language_vals),
    )

    # roi_int = vis_roi_ind()
    # roi_int_volume = make_volume(subj=args.subj, vals=roi_int, measure="rsq")

    # ev_vals = np.load("%s/output/evs_subj%02d_zscored.npy" % (OUTPUT_ROOT, args.subj))
    # ev_volume = make_volume(subj=args.subj, vals=ev_vals, measure="rsq")

    # old_ev_vals = np.load("%s/output/evs_old_subj%02d_zscored.npy" % (OUTPUT_ROOT, args.subj))
    # old_ev_volume = make_volume(subj=args.subj, vals=old_ev_vals, measure="rsq")
    # nc = np.load(
    #     "%s/output/noise_ceiling/subj%01d/noise_ceiling_subj%02d.npy"
    #     % (OUTPUT_ROOT, args.subj, args.subj)
    # )
    # nc_volume = make_volume(subj=args.subj, vals=nc, measure="rsq")

    # food = np.load("%s/output/subj01_food_v_all_FDR.npy" % (OUTPUT_ROOT))
    # food_volume = make_volume(subj=args.subj, vals=food, measure="pvalue", mask_with_significance=True)

    # Food maks
    # regions_nums_to_include = [136, 138, 163, 7, 22, 154, 6] #"TE2p", "PH", "VVC", "v8", "PIT", "VMV3", "v4"
    # food_mask = np.zeros(hcp_volume.data.shape)
    # for region_num in regions_nums_to_include:
    #     food_mask[np.where(hcp_volume.data == region_num)] = 1
    # food_mask_volume = cortex.Volume(
    #     food_mask,
    #     "subj%02d" % args.subj,
    #     "func1pt8_to_anat0pt8_autoFSbbr",
    #     mask=cortex.utils.get_cortical_mask(
    #         "subj%02d" % args.subj, "func1pt8_to_anat0pt8_autoFSbbr"
    #     ),
    #     vmin=np.min(food_mask),
    #     vmax=np.max(food_mask),
    # )

    volumes = {
        # "Visual ROIs": visual_roi_volume,
        # "Eccentricity ROIs": ecc_roi_volume,
        # "Places ROIs": place_roi_volume,
        # "Faces ROIs": face_roi_volume,
        # "Bodies ROIs": body_roi_volume,
        # "Words ROIs": word_roi_volume,
        # "Kastner2015": kastner_volume,
        # "HCP": hcp_volume,
        # "sulcus": sulc_volume,
        # "Language ROIs": language_volume,
        # "Noise Ceiling": nc_volume,
        # "EV": ev_volume,
        # "EV - old": old_ev_volume,
        # "food": food_volume,
        # "food_mask": food_mask_volume,
        # "roi_int": roi_int_volume,
        # "nsd_general": nsd_general_volume,
        # "rdm_bottom_right": rdm_volume,
    }

    yfcc_simclr = load_model_performance(
        "YFCC_simclr", output_root=OUTPUT_ROOT, subj=args.subj, measure="rsq"
    )
    yfcc_slip = load_model_performance(
        "YFCC_slip", output_root=OUTPUT_ROOT, subj=args.subj, measure="rsq"
    )
    yfcc_slip_simclr_joint = load_model_performance(
        "YFCC_slip_YFCC_simclr", output_root=OUTPUT_ROOT, subj=args.subj, measure="rsq"
    )
    nc = np.load(
        "%s/output/noise_ceiling/subj%01d/noise_ceiling_1d_subj%02d.npy"
        % (OUTPUT_ROOT, args.subj, args.subj)
    )

    # volumes["clip-ViT-last r"] = make_volume(
    #     subj=args.subj,
    #     model="clip",
    #     mask_with_significance=args.mask_sig,
    # )

    # volumes["clip-RN50-last r"] = make_volume(
    #     subj=args.subj,
    #     model="clip_visual_resnet",
    #     mask_with_significance=args.mask_sig,
    # )

    # volumes["clip-text-last r"] = make_volume(
    #     subj=args.subj,
    #     model="clip_text",
    #     mask_with_significance=args.mask_sig,
    # )

    # volumes["resnet50 r"] = make_volume(
    #     subj=args.subj,
    #     # model="convnet_res50",
    #     model="resnet50_bottleneck",
    #     mask_with_significance=args.mask_sig,
    # )

    # volumes["BERT-last r"] = make_volume(
    #     subj=args.subj,
    #     model="bert_layer_13",
    #     mask_with_significance=args.mask_sig,
    # )

    # rsquare
    volumes["clip-ViT-last R^2 NC"] = make_volume(
        subj=args.subj,
        model="clip",
        mask_with_significance=args.mask_sig,
        measure="rsq",
        noise_corrected=True,
    )

    volumes["clip-ViT-last R^2"] = make_volume(
        subj=args.subj,
        model="clip",
        mask_with_significance=args.mask_sig,
        measure="rsq",
        noise_corrected=False,
    )

    # volumes["GUSE R^2"] = make_volume(
    #     subj=args.subj,
    #     model="GUSE",
    #     mask_with_significance=args.mask_sig,
    #     measure="rsq",
    #     noise_corrected=False,
    # )

    volumes["clip-text-last R^2"] = make_volume(
        subj=args.subj,
        model="clip_text",
        mask_with_significance=args.mask_sig,
        measure="rsq",
        noise_corrected=False,
    )

    volumes["clip-RN50-last R^2"] = make_volume(
        subj=args.subj,
        model="clip_visual_resnet",
        mask_with_significance=args.mask_sig,
        measure="rsq",
    )

    volumes["bert-13 R^2"] = make_volume(
        subj=args.subj,
        model="bert_layer_13",
        mask_with_significance=args.mask_sig,
        measure="rsq",
    )

    volumes["resnet50 R^2"] = make_volume(
        subj=args.subj,
        # model="convnet_res50",
        model="resnet50_bottleneck",
        mask_with_significance=args.mask_sig,
        measure="rsq",
    )

    # volumes["clip&resnet50-clip ViT R^2"] = make_volume(
    #     subj=args.subj,
    #     model=[
    #         # "convnet_res50_clip",
    #         # "clip_convnet_res50",
    #         "clip_resnet50_bottleneck",
    #         "resnet50_bottleneck_clip",
    #     ],
    #     model2="clip",
    #     mask_with_significance=args.mask_sig,
    #     measure="rsq",
    # )

    # volumes["clip&resnet50-clip RN50 R^2"] = make_volume(
    #     subj=args.subj,
    #     model=[
    #         # "convnet_res50_clip",
    #         # "clip_convnet_res50",
    #         "clip_visual_resnet_resnet50_bottleneck",
    #         "resnet50_bottleneck_clip_visual_resnet",
    #     ],
    #     model2="clip_visual_resnet",
    #     mask_with_significance=args.mask_sig,
    #     measure="rsq",
    #     fdr_mask_name="resnet_unique_var",
    # )

    volumes["clip&resnet50-clip RN50 R^2 (for2d)"] = make_volume(
        subj=args.subj,
        model=[
            # "convnet_res50_clip",
            # "clip_convnet_res50",
            "clip_visual_resnet_resnet50_bottleneck",
            "resnet50_bottleneck_clip_visual_resnet",
        ],
        model2="clip_visual_resnet",
        mask_with_significance=args.mask_sig,
        measure="rsq",
        fdr_mask_name=[
            "clip_visual_resnet-resnet50_bottleneck_unique_var",
            "resnet50_bottleneck-clip_visual_resnet_unique_var",
        ],
    )

    # volumes["clip ViT&resnet50-resnet50 R^2"] = make_volume(
    #     subj=args.subj,
    #     model=[
    #         # "convnet_res50_clip",
    #         # "clip_convnet_res50",
    #         "clip_resnet50_bottleneck",
    #         "resnet50_bottleneck_clip",
    #     ],
    #     model2="resnet50_bottleneck",
    #     mask_with_significance=args.mask_sig,
    #     measure="rsq",
    # )

    # volumes["clip RN50&resnet50-resnet50 R^2"] = make_volume(
    #     subj=args.subj,
    #     model=[
    #         # "convnet_res50_clip",
    #         # "clip_convnet_res50",
    #         "clip_visual_resnet_resnet50_bottleneck",
    #         "resnet50_bottleneck_clip_visual_resnet",
    #     ],
    #     model2="resnet50_bottleneck",
    #     mask_with_significance=args.mask_sig,
    #     measure="rsq",
    #     cmap="inferno",
    #     fdr_mask_name="clip_unique_var",
    # )

    volumes["clip RN50&resnet50-resnet50 R^2 (for2d)"] = make_volume(
        subj=args.subj,
        model=[
            # "convnet_res50_clip",
            # "clip_convnet_res50",
            "clip_visual_resnet_resnet50_bottleneck",
            "resnet50_bottleneck_clip_visual_resnet",
        ],
        model2="resnet50_bottleneck",
        mask_with_significance=args.mask_sig,
        measure="rsq",
        cmap="inferno",
        fdr_mask_name=[
            "clip_visual_resnet-resnet50_bottleneck_unique_var",
            "resnet50_bottleneck-clip_visual_resnet_unique_var",
        ],
    )

    # volumes["clip&clip_text-clip R^2"] = make_volume(
    #     subj=args.subj,
    #     model="clip_clip_text",
    #     model2="clip",
    #     mask_with_significance=args.mask_sig,
    #     measure="rsq",
    # )

    # volumes["clip&clip_text-clip_text R^2"] = make_volume(
    #     subj=args.subj,
    #     model="clip_clip_text",
    #     model2="clip_text",
    #     mask_with_significance=args.mask_sig,
    #     measure="rsq",
    # )

    # volumes["clip_v_clip_text_unique"] = cortex.dataset.Volume2D(
    #     volumes["clip&clip_text-clip_text R^2"],
    #     volumes["clip&clip_text-clip R^2"],
    #     cmap="PU_BuOr_covar_alpha",
    #     vmin=0,
    #     vmax=0.05,
    #     vmin2=0,
    #     vmax2=0.05,
    # )

    # volumes["clip&bert13-bert13 R^2"] = make_volume(
    #     subj=args.subj,
    #     model=["clip_bert_layer_13", "bert_layer_13_clip"],
    #     model2="bert_layer_13",
    #     mask_with_significance=args.mask_sig,
    #     measure="rsq",
    # )

    # volumes["clip&bert13-clip R^2"] = make_volume(
    #     subj=args.subj,
    #     model=["clip_bert_layer_13", "bert_layer_13_clip"],
    #     model2="clip",
    #     mask_with_significance=args.mask_sig,
    #     measure="rsq",
    # )

    # volumes["clip&bert13"] = make_volume(
    #     subj=args.subj,
    #     model=["clip_bert_layer_13", "bert_layer_13_clip"],
    #     mask_with_significance=args.mask_sig,
    #     measure="rsq",
    # )

    # volumes["clip&resnet50 R^2"] = make_volume(
    #     subj=args.subj,
    #     model=["clip_resnet50_bottleneck", "resnet50_bottleneck_clip"],
    #     mask_with_significance=args.mask_sig,
    #     measure="rsq",
    # )

    # volumes["clipViT_v_resnet50_unique"] = cortex.dataset.Volume2D(volumes["clip&resnet50-clip ViT R^2"], volumes["clip ViT&resnet50-resnet50 R^2"], cmap="PU_BuOr_covar_alpha", vmin=0.02, vmax=0.1, vmin2=0.02, vmax2=0.1)
    volumes["clipRN50_v_resnet50_unique"] = cortex.dataset.Volume2D(
        volumes["clip&resnet50-clip RN50 R^2 (for2d)"],
        volumes["clip RN50&resnet50-resnet50 R^2 (for2d)"],
        cmap="PU_BuOr_covar_alpha",
        vmin=0.02,
        vmax=0.1,
        vmin2=0.02,
        vmax2=0.1,
    )

    volumes["clipRN50_v_resnet50"] = cortex.dataset.Volume2D(
        volumes["resnet50 R^2"],
        volumes["clip-RN50-last R^2"],
        cmap="PU_BuOr_covar_alpha",
        vmin=0,
        vmax=0.05,
        vmin2=0,
        vmax2=0.05,
    )

    ### new volumes ###
    volumes["YFCC clip R^2"] = make_volume(
        subj=args.subj,
        model="YFCC_clip",
        mask_with_significance=args.mask_sig,
        measure="rsq",
        noise_corrected=False,
    )

    volumes["YFCC simclr R^2"] = make_volume(
        subj=args.subj,
        model="YFCC_simclr",
        mask_with_significance=args.mask_sig,
        measure="rsq",
        noise_corrected=False,
    )

    ## comparing yfcc_clip with simclr
    # volumes["clip&simclr-clip R^2"] = make_volume(
    #     subj=args.subj,
    #     model="YFCC_simclr_YFCC_clip",
    #     model2="YFCC_clip",
    #     mask_with_significance=args.mask_sig,
    #     measure="rsq",
    #     cmap="inferno",
    # )

    # volumes["clip&simclr-simclr R^2"] = make_volume(
    #     subj=args.subj,
    #     model="YFCC_simclr_YFCC_clip",
    #     model2="YFCC_simclr",
    #     mask_with_significance=args.mask_sig,
    #     measure="rsq",
    #     cmap="inferno",
    # )

    # volumes["yfcc_clip_v_simclr_unique"] = cortex.dataset.Volume2D(
    #     volumes["clip&simclr-simclr R^2"],
    #     volumes["clip&simclr-clip R^2"],
    #     cmap="PU_BuOr_covar_alpha",
    #     vmin=0,
    #     vmax=0.05,
    #     vmin2=0,
    #     vmax2=0.05,
    # )

    ## comparing clipO with yfcc_clip
    volumes["YFCC_clip&clipO-YFCC_clip R^2"] = make_volume(
        subj=args.subj,
        model="YFCC_clip_clip",
        model2="YFCC_clip",
        mask_with_significance=args.mask_sig,
        measure="rsq",
        cmap="inferno",
    )

    volumes["YFCC_clip&clipO-clipO R^2"] = make_volume(
        subj=args.subj,
        model="YFCC_clip_clip",
        model2="clip",
        mask_with_significance=args.mask_sig,
        measure="rsq",
        cmap="inferno",
    )
    volumes["yfcc_clip_v_clipO_unique"] = cortex.dataset.Volume2D(
        volumes["YFCC_clip&clipO-clipO R^2"],
        volumes["YFCC_clip&clipO-YFCC_clip R^2"],
        cmap="PU_BuOr_covar_alpha",
        vmin=0,
        vmax=0.05,
        vmin2=0,
        vmax2=0.05,
    )

    # ## comparing simclr with clipO
    # volumes["YFCC_simclr&clipO-YFCC_simclr R^2"] = make_volume(
    #     subj=args.subj,
    #     model="YFCC_simclr_clip",
    #     model2="YFCC_simclr",
    #     mask_with_significance=args.mask_sig,
    #     measure="rsq",
    #     cmap="inferno",
    # )

    # volumes["YFCC_simclr&clipO-clipO R^2"] = make_volume(
    #     subj=args.subj,
    #     model="YFCC_simclr_clip",
    #     model2="clip",
    #     mask_with_significance=args.mask_sig,
    #     measure="rsq",
    #     cmap="inferno",
    # )
    # volumes["yfcc_simclr_v_clipO_unique"] = cortex.dataset.Volume2D(
    #     volumes["YFCC_simclr&clipO-YFCC_simclr R^2"],
    #     volumes["YFCC_simclr&clipO-clipO R^2"],
    #     cmap="PU_BuOr_covar_alpha",
    #     vmin=0,
    #     vmax=0.05,
    #     vmin2=0,
    #     vmax2=0.05,
    # )

    volumes["YFCC slip R^2"] = make_volume(
        subj=args.subj,
        model="YFCC_slip",
        mask_with_significance=args.mask_sig,
        measure="rsq",
        noise_corrected=False,
    )

    ## comparing slip with simclr
    volumes["slip&simclr-slip R^2"] = make_volume(
        subj=args.subj,
        model="YFCC_slip_YFCC_simclr",
        model2="YFCC_slip",
        mask_with_significance=args.mask_sig,
        measure="rsq",
        cmap="inferno",
        fdr_mask_name=[
            "YFCC_simclr-YFCC_slip_unique_var",
            "YFCC_slip-YFCC_simclr_unique_var",
        ],
    )

    volumes["slip&simclr-simclr R^2"] = make_volume(
        subj=args.subj,
        model="YFCC_slip_YFCC_simclr",
        model2="YFCC_simclr",
        mask_with_significance=args.mask_sig,
        measure="rsq",
        cmap="inferno",
        fdr_mask_name=[
            "YFCC_simclr-YFCC_slip_unique_var",
            "YFCC_slip-YFCC_simclr_unique_var",
        ],
    )

    # volumes["slip-simclr R^2"] = make_volume(
    #     subj=args.subj,
    #     model="YFCC_slip",
    #     model2="YFCC_simclr",
    #     mask_with_significance=args.mask_sig,
    #     measure="rsq",
    #     cmap="inferno",
    # )

    volumes["yfcc_slip_v_simclr_unique"] = cortex.dataset.Volume2D(
        volumes["slip&simclr-simclr R^2"],
        volumes["slip&simclr-slip R^2"],
        cmap="PU_BuOr_covar_alpha",
        vmin=0,
        vmax=0.05,
        vmin2=0,
        vmax2=0.05,
    )

    ## Comparing clipO with clip
    volumes["clipO&slip-clipO R^2"] = make_volume(
        subj=args.subj,
        model="YFCC_slip_clip",
        model2="clip",
        mask_with_significance=args.mask_sig,
        measure="rsq",
        cmap="inferno",
    )

    volumes["clipO&slip-slip R^2"] = make_volume(
        subj=args.subj,
        model="YFCC_slip_clip",
        model2="YFCC_slip",
        mask_with_significance=args.mask_sig,
        measure="rsq",
        cmap="inferno",
    )

    volumes["yfcc_slip_v_clipO_unique"] = cortex.dataset.Volume2D(
        volumes["clipO&slip-slip R^2"],
        volumes["clipO&slip-clipO R^2"],
        cmap="PU_BuOr_covar_alpha",
        vmin=0,
        vmax=0.05,
        vmin2=0,
        vmax2=0.05,
    )

    volumes["laion2b R^2"] = make_volume(
        subj=args.subj,
        model="laion2b_clip",
        mask_with_significance=args.mask_sig,
        measure="rsq",
        noise_corrected=False,
    )

    volumes["laion400m R^2"] = make_volume(
        subj=args.subj,
        model="laion400m_clip",
        mask_with_significance=args.mask_sig,
        measure="rsq",
        noise_corrected=False,
    )

    volumes["laion2b R^2"] = make_volume(
        subj=args.subj,
        model="laion2b_clip",
        mask_with_significance=args.mask_sig,
        measure="rsq",
        noise_corrected=False,
    )

    volumes["laion400m R^2"] = make_volume(
        subj=args.subj,
        model="laion400m_clip",
        mask_with_significance=args.mask_sig,
        measure="rsq",
        noise_corrected=False,
    )

    volumes["laion400m&2b-laion2b R^2"] = make_volume(
        subj=args.subj,
        model="laion2b_clip_laion400m_clip",
        model2="laion2b_clip",
        mask_with_significance=args.mask_sig,
        measure="rsq",
        cmap="inferno",
        fdr_mask_name=[
            "laion400m_clip-laion2b_clip_unique_var",
            "laion2b_clip-laion400m_clip_unique_var",
        ],
    )

    volumes["laion400m&2b-laion400m R^2"] = make_volume(
        subj=args.subj,
        model="laion2b_clip_laion400m_clip",
        model2="laion400m_clip",
        mask_with_significance=args.mask_sig,
        measure="rsq",
        cmap="inferno",
        fdr_mask_name=[
            "laion400m_clip-laion2b_clip_unique_var",
            "laion2b_clip-laion400m_clip_unique_var",
        ],
    )

    volumes["laion400m_v_laion2b_unique"] = cortex.dataset.Volume2D(
        volumes["laion400m&2b-laion400m R^2"],
        volumes["laion400m&2b-laion2b R^2"],
        cmap="PU_BuOr_covar_alpha",
        vmin=0,
        vmax=0.05,
        vmin2=0,
        vmax2=0.05,
    )

    volumes["laion400m&clipO-laion400m R^2"] = make_volume(
        subj=args.subj,
        model="clip_laion400m_clip",
        model2="laion400m_clip",
        mask_with_significance=args.mask_sig,
        measure="rsq",
        cmap="inferno",
        fdr_mask_name=[
            "laion400m_clip-clip_unique_var",
            "clip-laion400m_clip_unique_var",
        ],
    )

    volumes["laion400m&clipO-clipO R^2"] = make_volume(
        subj=args.subj,
        model="clip_laion400m_clip",
        model2="clip",
        mask_with_significance=args.mask_sig,
        measure="rsq",
        cmap="inferno",
        fdr_mask_name=[
            "laion400m_clip-clip_unique_var",
            "clip-laion400m_clip_unique_var",
        ],
    )

    volumes["clipO_v_laion400m_unique"] = cortex.dataset.Volume2D(
        volumes["laion400m&clipO-laion400m R^2"],
        volumes["laion400m&clipO-clipO R^2"],
        cmap="PU_BuOr_covar_alpha",
        vmin=0,
        vmax=0.05,
        vmin2=0,
        vmax2=0.05,
    )

    volumes["IC title R^2"] = make_volume(
        subj=args.subj,
        model="IC_title_clip",
        mask_with_significance=args.mask_sig,
        measure="rsq",
        noise_corrected=False,
    )

    volumes["IC title+ R^2"] = make_volume(
        subj=args.subj,
        model="IC_title_tag_description_clip",
        mask_with_significance=args.mask_sig,
        measure="rsq",
        noise_corrected=False,
    )

    # volumes["IC title& I - I R^2"] = make_volume(
    #     subj=args.subj,
    #     model="IC_title_clip_resnet50_bottleneck",
    #     model2="resnet50_bottleneck",
    #     mask_with_significance=args.mask_sig,
    #     measure="rsq",
    #     cmap="inferno",
    # )

    # volumes["IC title & I - IC title R^2"] = make_volume(
    #     subj=args.subj,
    #     model="IC_title_clip_resnet50_bottleneck",
    #     model2="IC_title_clip",
    #     mask_with_significance=args.mask_sig,
    #     measure="rsq",
    #     cmap="inferno",
    # )

    # volumes["IC title+ & I - I R^2"] = make_volume(
    #     subj=args.subj,
    #     model="IC_title_tag_description_clip_resnet50_bottleneck",
    #     model2="resnet50_bottleneck",
    #     mask_with_significance=args.mask_sig,
    #     measure="rsq",
    #     cmap="inferno",
    # )
    # volumes["IC title+ & I - IC title+ R^2"] = make_volume(
    #     subj=args.subj,
    #     model="IC_title_tag_description_clip_resnet50_bottleneck",
    #     model2="IC_title_tag_description_clip",
    #     mask_with_significance=args.mask_sig,
    #     measure="rsq",
    #     cmap="inferno",
    # )

    # volumes["IC_title_v_I_unique"] = cortex.dataset.Volume2D(
    #     volumes["IC title & I - I R^2"],
    #     volumes["IC title & I - IC title R^2"],
    #     cmap="PU_BuOr_covar_alpha",
    #     vmin=0,
    #     vmax=0.05,
    #     vmin2=0,
    #     vmax2=0.05,
    # )

    # volumes["IC_title+_v_I_unique"] = cortex.dataset.Volume2D(
    #     volumes["IC title+ & I - I R^2"],
    #     volumes["IC title+ & I - IC title+ R^2"],
    #     cmap="PU_BuOr_covar_alpha",
    #     vmin=0,
    #     vmax=0.05,
    #     vmin2=0,
    #     vmax2=0.05,
    # )

    # volumes["YFCC clip n-1  R^2"] = make_volume(
    #     subj=args.subj,
    #     model="YFCC_clip_layer_n-1",
    #     mask_with_significance=args.mask_sig,
    #     measure="rsq",
    #     noise_corrected=False,
    # )

    # volumes["clip_v_bert_unique"] = cortex.dataset.Volume2D(
    #     volumes["clip&bert13-clip R^2"],
    #     volumes["clip&bert13-bert13 R^2"],
    #     cmap="PU_BuOr_covar_alpha",
    #     vmin=0.02,
    #     vmax=0.1,
    #     vmin2=0.02,
    #     vmax2=0.1,
    # )
    # volumes["clip-ViT-layerwise-asc"] = visualize_layerwise_max_corr_results("visual_layer", 12, threshold=98, order="asc")
    # volumes["clip-ViT-layerwise-des"] = visualize_layerwise_max_corr_results("visual_layer", 12, threshold=98, order="des")
    # volumes["clip_v_clip_visual_resnet"] = cortex.dataset.Volume2D(
    #     volumes["clip-ViT-last R^2"],
    #     volumes["clip-RN50-last R^2"],
    #     cmap="PU_BuOr_covar_alpha",
    #     vmin=0,
    #     vmax=0.1,
    #     vmin2=0,
    #     vmax2=0.1,
    # )

    vals = (yfcc_slip_simclr_joint - yfcc_simclr) / ((nc / 100) - yfcc_simclr)
    all_vals = project_vals_to_3d(vals, cortical_mask)
    tmp_volume = cortex.Volume(
        all_vals,
        "subj%02d" % args.subj,
        "func1pt8_to_anat0pt8_autoFSbbr",
        mask=cortex.utils.get_cortical_mask(
            "subj%02d" % args.subj, "func1pt8_to_anat0pt8_autoFSbbr"
        ),
    )
    volumes["YFCC slip uvar / (NC-simclr) R^2"] = tmp_volume
    volumes["YFCC slip & simclr"] = make_volume(
        subj=args.subj,
        model="YFCC_slip_YFCC_simclr",
        mask_with_significance=args.mask_sig,
        measure="rsq",
        cmap="inferno",
    )
    volumes[
        "YFCC slip uvar / (NC-simclr) R^2 vs joint model 2D"
    ] = cortex.dataset.Volume2D(
        volumes["YFCC slip uvar / (NC-simclr) R^2"],
        volumes["YFCC slip & simclr"],
        cmap="PU_BuOr_covar_alpha",
    )

    if args.show_repr_sim:
        model_list = [
            ["YFCC_slip", "YFCC_simclr"],
            ["YFCC_clip", "YFCC_simclr"],
            ["clip", "resnet50_bottleneck"],
        ]
        for models in model_list:
            model1, model2 = models
            volumes[
                "%s vs %s in repr sim BR" % (model1, model2)
            ] = show_voxel_diff_in_repr_samples(model1, model2, "br")
            volumes[
                "%s vs %s in repr sim TL" % (model1, model2)
            ] = show_voxel_diff_in_repr_samples(model1, model2, "tl")
            volumes[
                "%s vs %s in repr sim BR-TL" % (model1, model2)
            ] = show_voxel_diff_in_repr_samples(model1, model2, "br-tl")
            volumes[
                "%s vs %s in repr sim BR - base" % (model1, model2)
            ] = show_voxel_diff_in_repr_samples(model1, model2, "br_sub_baseline")
            volumes[
                "%s vs %s in repr sim TL - base" % (model1, model2)
            ] = show_voxel_diff_in_repr_samples(model1, model2, "tl_sub_baseline")

    if args.subj == 1 & args.show_more:
        volumes["clip_top1_object"] = make_volume(
            subj=args.subj,
            model="clip_top1_object",
            mask_with_significance=args.mask_sig,
        )

        volumes["clip_all_objects"] = make_volume(
            subj=args.subj, model="clip_object", mask_with_significance=args.mask_sig,
        )

        volumes["COCO categories -r^2"] = make_volume(
            subj=args.subj,
            model="cat",
            mask_with_significance=args.mask_sig,
            measure="rsq",
        )

        volumes["COCO super categories"] = make_volume(
            subj=args.subj, model="supcat", mask_with_significance=args.mask_sig,
        )

        volumes["CLIP&CLIPtop1 - top1"] = make_volume(
            subj=args.subj,
            model="clip_clip_top1_object",
            model2="clip_top1_object",
            mask_with_significance=args.mask_sig,
            measure="rsq",
        )

        volumes["CLIP&CLIPtop1 - CLIP"] = make_volume(
            subj=args.subj,
            model="clip_clip_top1_object",
            model2="clip",
            mask_with_significance=args.mask_sig,
            measure="rsq",
        )

        volumes["CLIP&Resnet50"] = make_volume(
            subj=args.subj,
            model="clip_resnet50_bottleneck",
            mask_with_significance=args.mask_sig,
        )

        # for model in ["resnet50_bottleneck", "clip", "cat"]:
        #     for subset in ["person", "giraffe", "toilet", "train"]:
        #         model_name = "%s_%s_subset" % (model, subset)

        #         volumes[model_name] = make_volume(
        #             subj=args.subj,
        #             model=model_name,
        #             mask_with_significance=args.mask_sig,
        #         )

        volumes["clip-person-subset"] = make_volume(
            subj=args.subj,
            model="clip_person_subset",
            mask_with_significance=args.mask_sig,
        )

        volumes["clip-no-person-subset"] = make_volume(
            subj=args.subj,
            model="clip_no_person_subset",
            mask_with_significance=args.mask_sig,
        )

        volumes["resnet-person-subset"] = make_volume(
            subj=args.subj,
            model="resnet50_bottleneck_person_subset",
            mask_with_significance=args.mask_sig,
        )

        volumes["resnet-no-person-subset"] = make_volume(
            subj=args.subj,
            model="resnet50_bottleneck_no_person_subset",
            mask_with_significance=args.mask_sig,
        )

        volumes["cat-person-subset"] = make_volume(
            subj=args.subj,
            model="cat_person_subset",
            mask_with_significance=args.mask_sig,
        )

        volumes["clip-top1-person-subset"] = make_volume(
            subj=args.subj,
            model="clip_top1_object_person_subset",
            mask_with_significance=args.mask_sig,
        )

        volumes["clip-top1-no-person-subset"] = make_volume(
            subj=args.subj,
            model="clip_top1_object_no_person_subset",
            mask_with_significance=args.mask_sig,
        )

        # for i in range(12):
        #     volumes["clip-ViT-%s" % str(i + 1)] = make_volume(
        #         subj=args.subj,
        #         model="visual_layer_%d" % i,
        #         mask_with_significance=args.mask_sig,
        #

        # for i in range(12):
        #     volumes["clip-text-%s" % str(i + 1)] = make_volume(
        #         subj=args.subj,
        #         model="text_layer_%d" % i,
        #         mask_with_significance=args.mask_sig,
        #     )

        # for i in range(7):
        #     volumes["clip-RN-%s" % str(i + 1)] = make_volume(
        #         subj=args.subj,
        #         model="visual_layer_resnet_%d" % i,
        #         mask_with_significance=args.mask_sig,
        #     )
        volumes["clip-RN-last"] = make_volume(
            subj=args.subj,
            model="clip_visual_resnet",
            mask_with_significance=args.mask_sig,
        )

        # volumes["clip&bert13 R^2 - old"] = make_volume(
        #     subj=args.subj,
        #     model="clip_bert_layer_13_old",
        #     mask_with_significance=args.mask_sig,
        #     measure="rsq",
        # )

        # volumes["clip&resnet50-clip R^2 - old"] = make_volume(
        #     subj=args.subj,
        #     model=["resnet50_bottleneck_clip", "clip_resnet50_bottleneck"],
        #     model2="clip_old",
        #     mask_with_significance=args.mask_sig,
        #     measure="rsq",
        # )

        # volumes["clip-ViT-last R^2 - old"] = make_volume(
        #     subj=args.subj,
        #     model="clip_old",
        #     mask_with_significance=args.mask_sig,
        #     measure="rsq",
        # )

        # volumes["clip-ViT-last R^2 - old(rerun)"] = make_volume(
        #     subj=args.subj,
        #     model="clip_old_rerun",
        #     mask_with_significance=args.mask_sig,
        #     measure="rsq",
        # )

        # volumes["clip&resnet50 R^2 - old"] = make_volume(
        #     subj=args.subj,
        #     model="clip_convnet_res50",
        #     mask_with_significance=args.mask_sig,
        #     measure="rsq",
        # )
        # volumes["resnet50 - old"] = make_volume(
        #     subj=args.subj,
        #     model="resnet50_bottleneck",
        #     mask_with_significance=args.mask_sig,
        # )

        # volumes["bert-13 R^2 - old"] = make_volume(
        #     subj=args.subj,
        #     model="bert_layer_13_old",
        #     mask_with_significance=args.mask_sig,
        #     measure="rsq",
        # )

        # volumes["resnet50 R^2 - old"] = make_volume(
        #     subj=args.subj,
        #     model="resnet50_bottleneck",
        #     mask_with_significance=args.mask_sig,
        #     measure="rsq",
        # )

        # volumes["clip&resnet50-resnet50 R^2 - old"] = make_volume(
        #     subj=args.subj,
        #     model=["resnet50_bottleneck_clip", "clip_resnet50_bottleneck"],
        #     model2="resnet50_bottleneck",
        #     mask_with_significance=args.mask_sig,
        #     measure="rsq",
        # )
        # volumes["clip&bert13-bert13 R^2 - old"] = make_volume(
        #     subj=args.subj,
        #     model=["clip_bert_layer_13_old", "bert_layer_13_clip_old"],
        #     model2="bert_layer_13",
        #     mask_with_significance=args.mask_sig,
        #     measure="rsq",
        # )

        # volumes["clip&bert13-clip R^2 - old"] = make_volume(
        #     subj=args.subj,
        #     model=["clip_bert_layer_13_old", "bert_layer_13_clip_old"],
        #     model2="clip_old",
        #     mask_with_significance=args.mask_sig,
        #     measure="rsq",
        # )

        # volumes["clip - clip(old) R^2"] = make_volume(
        #     subj=args.subj,
        #     model="clip",
        #     model2="clip_old",
        #     mask_with_significance=args.mask_sig,
        #     measure="rsq",
        # )

        # volumes["clip - clip(old) R^2"] = make_volume(
        #     subj=args.subj,
        #     model="clip",
        #     model2="clip_old_rerun",
        #     mask_with_significance=args.mask_sig,
        #     measure="rsq",
        # )

        volumes["oscar"] = make_volume(
            subj=args.subj,
            model="oscar",
            mask_with_significance=args.mask_sig,
            measure="corr",
        )

        volumes["clip&oscar - oscar"] = make_volume(
            subj=args.subj,
            model="oscar_clip",
            model2="oscar",
            mask_with_significance=args.mask_sig,
            measure="rsq",
        )

        volumes["clip&oscar - clip"] = make_volume(
            subj=args.subj,
            model="oscar_clip",
            model2="clip",
            mask_with_significance=args.mask_sig,
            measure="rsq",
        )

        volumes["ResNet&oscar - ResNet"] = make_volume(
            subj=args.subj,
            model="resnet50_bottleneck_oscar",
            model2="resnet50_bottleneck",
            mask_with_significance=args.mask_sig,
            measure="rsq",
        )

        volumes["ResNet&oscar - oscar"] = make_volume(
            subj=args.subj,
            model="resnet50_bottleneck_oscar",
            model2="oscar",
            mask_with_significance=args.mask_sig,
            measure="rsq",
        )

        # for i in range(13):
        #     volumes["bert-%s" % str(i + 1)] = make_volume(
        #         subj=args.subj,
        #         model="bert_layer_%d" % (i + 1),
        #         mask_with_significance=args.mask_sig,
        #     )

        volumes["clip-RN-layerwise"] = visualize_layerwise_max_corr_results(
            "visual_layer_resnet", 7, threshold=85, mask_with_significance=args.mask_sig
        )
        # volumes["clip-text-layerwise"] = visualize_layerwise_max_corr_results(
        #     "text_layer", 12, threshold=85, mask_with_significance=args.mask_sig
        # )
        # volumes["bert-layerwise"] = visualize_layerwise_max_corr_results(
        #     "bert_layer", 13, threshold=85, mask_with_significance=args.mask_sig, start_with_zero=False
        # )

    if args.show_pcs:
        model = "clip"
        # name_modifiers = ["best_20000_nc", "floc-bodies_floc-places_floc-faces_only", "floc-bodies_only", "floc-faces_only", "floc-places_only", "EBA_only", "OPA_only"]
        name_modifiers = ["best_20000_nc"]
        for name_modifier in name_modifiers:
            # visualize PC projections
            subj_proj = np.load(
                "%s/output/pca/%s/%s/subj%02d/pca_projections.npy"
                % (OUTPUT_ROOT, model, name_modifier, args.subj)
            )
            subj_mask = np.load(
                "%s/output/pca/%s/%s/pca_voxels/pca_voxels_subj%02d.npy"
                % (OUTPUT_ROOT, model, name_modifier, args.subj)
            )
            # proj_val_only = subj_proj[]

            # proj_vals = np.zeros(subj_proj.shape)
            # proj_vals[:, ~subj_mask] = np.nan
            # proj_vals[:, subj_mask] = subj_proj
            subj_proj_nan_out = subj_proj.copy()
            subj_proj_nan_out[:, ~subj_mask] = np.nan

            for i in range(subj_proj.shape[0]):
                key = "Proj " + str(i) + name_modifier
                volumes[key] = make_pc_volume(args.subj, subj_proj_nan_out[i, :],)

            import matplotlib.pyplot as plt

            plt.figure()
            plt.plot(np.sum(subj_proj ** 2, axis=1))
            plt.savefig("figures/PCA/proj_norm_%s.png" % name_modifier)

            plt.figure()
            plt.hist(subj_proj[0, :], label="0", alpha=0.3)
            plt.hist(subj_proj[1, :], label="1", alpha=0.3)
            plt.legend()
            plt.savefig("figures/PCA/proj_hist_%s.png" % name_modifier)

        # volumes["3PC"] = make_3pc_volume(
        #     args.subj,
        #     PCs_zscore,
        # )

        # # basis?
        # def kmean_sweep_on_PC(n_pc):
        #     from sklearn.cluster import KMeans

        #     inertia = []
        #     for k in range(3, 10):
        #         kmeans = KMeans(n_clusters=k, random_state=0).fit(
        #             subj_proj[:n_pc, :].T
        #         )
        #         volumes["basis %d-%d" % (n_pc, k)] = make_pc_volume(
        #             args.subj, kmeans.labels_, vmin=0, vmax=k-1, cmap="J4s"
        #         )
        #         inertia.append(kmeans.inertia_)
        #     return inertia

        # import matplotlib.pyplot as plt

        # plt.figure()
        # n_pcs = [3, 5, 10]
        # for n in n_pcs:
        #     inertia = kmean_sweep_on_PC(n)
        #     plt.plot(inertia, label="%d PCS" % n)
        # plt.savefig("figures/pca/clustering/inertia_across_pc_num.png")

        # # visualize PC projections
        # subj_proj = np.load(
        #             "%s/output/pca/%s/subj%02d/%s_feature_pca_projections.npy"
        #             % (OUTPUT_ROOT, model, args.subj, model)
        #         )
        # for i in range(PCs.shape[0]):
        #     key = "PC Proj " + str(i)
        #     volumes[key] = make_pc_volume(
        #         args.subj,
        #         subj_proj[i, :],
        #     )

        # mni_data = project_vols_to_mni(s, volume)

        # mni_vol = cortex.Volume(
        #     mni_data,
        #     "fsaverage",
        #     "atlas",
        #     cmap="inferno",
        # )

        # cortex.quickflat.make_figure(mni_vol, with_roi=False)
        # print("***********")
        # print(volumes["PC1"])

    if args.show_clustering:
        model = "clip"

        # name_modifier = "acc_0.3_minus_prf-visualrois"
        name_modifier = "best_20000_nc"
        labels_vals = np.load(
            "%s/output/clustering/spectral_subj%01d.npy" % (OUTPUT_ROOT, args.subj)
        )
        subj_mask = np.load(
            "%s/output/pca/%s/%s/pca_voxels/pca_voxels_subj%02d.npy"
            % (OUTPUT_ROOT, model, name_modifier, args.subj)
        )

        labels = np.zeros(subj_mask.shape)
        labels[~subj_mask] = np.nan
        labels[subj_mask] = labels_vals

        # volumes["spectral clustering"] = make_pc_volume(
        #             args.subj, labels, vmin=0, vmax=max(labels), cmap="J4s"
        #         )

    if args.vis_method == "webgl":
        subj_port = "4111" + str(args.subj)
        # cortex.webgl.show(data=volumes, autoclose=False, port=int(subj_port))
        cortex.webgl.show(data=volumes, port=int(subj_port), recache=False)

    elif args.vis_method == "MNI":
        group_mni_data = []
        mni_mask = cortex.utils.get_cortical_mask("fsaverage", "atlas")

        for s in range(1, 9):
            print("SUBJECT: " + str(s))

            # visualize in MNI space

            cortical_mask = np.load(
                "%s/output/voxels_masks/subj%d/cortical_mask_subj%02d.npy"
                % (OUTPUT_ROOT, s, s)
            )
            mask = cortex.utils.get_cortical_mask(
                "subj%02d" % s, "func1pt8_to_anat0pt8_autoFSbbr"
            )

            volume = make_volume(
                subj=s,
                model="YFCC_slip_YFCC_simclr",
                model2="YFCC_simclr",
                mask_with_significance=args.mask_sig,
                measure="rsq",
                cmap="inferno",
                # fdr_mask_name=["YFCC_simclr-YFCC_slip_unique_var", "YFCC_slip-YFCC_simclr_unique_var"]
            )
            mni_data = project_vols_to_mni(s, volume)
            # group_mni_data.append(mni_data.flatten())
            group_mni_data.append(mni_data[mni_mask])

            # saving MNI flatmap for individual subjects
            volume_masked = make_volume(
                subj=s,
                model="YFCC_slip_YFCC_simclr",
                model2="YFCC_simclr",
                mask_with_significance=args.mask_sig,
                measure="rsq",
                cmap="inferno",
                fdr_mask_name=[
                    "YFCC_simclr-YFCC_slip_unique_var",
                    "YFCC_slip-YFCC_simclr_unique_var",
                ],
            )
            mni_data_masked = project_vols_to_mni(s, volume_masked)
            # mni_3d = np.zeros(mni_mask.shape)
            # mni_3d[mni_mask] = mni_data_masked
            subj_vol = cortex.Volume(
                mni_data_masked,
                "fsaverage",
                "atlas",
                cmap="inferno",
                vmin=0,
                vmax=0.05,
            )
            filename = "figures/flatmap/mni/subj%d_YFCC_slip_unique_var_sig_vox.png" % s
            _ = cortex.quickflat.make_png(
                filename,
                subj_vol,
                linewidth=1,
                labelsize="17pt",
                with_curvature=True,
                recache=False,
                # roi_list=roi_list,
            )

        from scipy.stats import ttest_1samp
        from statsmodels.stats.multitest import fdrcorrection

        t, p_val = ttest_1samp(np.array(group_mni_data), 0, alternative="greater")
        print(t)
        print(p_val.shape)
        # p_val = p_val.reshape(mni_data.shape)

        # np.save("%s/output/ci_threshold/YFCC_slip_group_t_test_p.npy" % OUTPUT_ROOT, p_val)
        np.save("%s/output/ci_threshold/YFCC_slip_group_t_test_t.npy" % OUTPUT_ROOT, t)
        # import pdb

        # pdb.set_trace()

        p_val_to_show = p_val.copy()
        p_val_to_show[p_val > 0.05] = np.nan

        p_val_3d = np.zeros(mni_mask.shape)
        p_val_3d[mni_mask] = p_val_to_show
        # all_vals = np.swapaxes(all_vals, 0, 2)

        # p_val_to_show_3d = project_vals_to_3d(p_val_to_show, mni_mask)
        p_vol = cortex.Volume(
            p_val_3d, "fsaverage", "atlas", cmap="Blues_r", vmin=0, vmax=0.05,
        )

        # -log10

        # cortex.quickflat.make_figure(p_vol, with_roi=False)
        filename = "figures/flatmap/mni/group_YFCC_slip_p.png"
        _ = cortex.quickflat.make_png(
            filename,
            p_vol,
            linewidth=1,
            labelsize="17pt",
            with_curvature=True,
            recache=False,
            # roi_list=roi_list,
        )

        mask_for_nan = ~np.isnan(p_val)
        fdr_p = fdrcorrection(p_val[mask_for_nan])[1]
        # fdr_p = fdrcorrection(p_val)[1]

        fdr_p_full = np.ones(p_val.shape)
        fdr_p_full[mask_for_nan] = fdr_p

        fdr_p_val_3d = np.zeros(mni_mask.shape)
        fdr_p_val_3d[mni_mask] = fdr_p_full

        fdr_p_vol = cortex.Volume(
            fdr_p_val_3d, "fsaverage", "atlas", cmap="Blues_r", vmin=0, vmax=0.05,
        )

        # -log10

        # cortex.quickflat.make_figure(p_vol, with_roi=False)
        filename = "figures/flatmap/mni/group_YFCC_slip_fdr_p.png"
        _ = cortex.quickflat.make_png(
            filename,
            fdr_p_vol,
            linewidth=1,
            labelsize="17pt",
            with_curvature=True,
            recache=False,
            # roi_list=roi_list,
        )

        np.save(
            "%s/output/ci_threshold/YFCC_slip_group_t_test_fdr_p.npy" % OUTPUT_ROOT,
            fdr_p_full,
        )

    elif args.vis_method == "quickflat":
        roi_list = ["RSC", "PPA", "OPA", "EBA", "EarlyVis", "FFA-1", "FFA-2"]
        for k in volumes.keys():
            # vol_name = k.replace(" ", "_")
            root = "./figures/flatmap/subj%d" % args.subj
            if not os.path.exists(root):
                os.makedirs(root)

            filename = "%s/%s.png" % (root, k)
            _ = cortex.quickflat.make_png(
                filename,
                volumes[k],
                linewidth=3,
                labelsize="20pt",
                with_curvature=True,
                recache=False,
                roi_list=roi_list,
            )

    elif args.vis_method == "3d_views":
        from save_3d_views import save_3d_views

        for k, v in volumes.items():
            root = "figures/3d_views/subj%s" % args.subj
            if not os.path.exists(root):
                os.makedirs(root)
            _ = save_3d_views(
                v,
                root,
                k,
                list_views=["lateral", "bottom", "back"],
                list_surfaces=["inflated"],
                with_labels=True,
                size=(1024 * 4, 768 * 4),
                trim=True,
            )

    import pdb

    pdb.set_trace()
