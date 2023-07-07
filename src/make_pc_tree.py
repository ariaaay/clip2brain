import argparse
import numpy as np
from tqdm import tqdm

import cortex
from visualize_in_pycortex import project_vals_to_3d


def bicluster(data, idxes_for_data, subj, cortical_n, depth=0, branch=""):
    from sklearn.cluster import KMeans

    if data.shape[0] < 50:
        return
    if depth > 4:
        return
    kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
    idx = np.where(kmeans.labels_ == 0)[0]
    idx_A = idxes_for_data[idx]
    data_A = data[idx, :]
    mask_A = np.zeros(cortical_n).astype(bool)  # 100k x 1
    mask_A[idx_A] = True

    idx = np.where(kmeans.labels_ == 1)[0]
    idx_B = idxes_for_data[idx]
    data_B = data[idx, :]
    mask_B = np.zeros(cortical_n).astype(bool)  # 100k x 1
    mask_B[idx_B] = True

    current_mask = np.zeros(cortical_n).astype(bool)  # 100k x 1
    current_mask[idxes_for_data] = True
    VOLS["L%d %s" % (depth, branch)] = make_volume(
        subj, kmeans.labels_, current_mask, vmin=0, vmax=1, cmap="viridis_r"
    )

    bicluster(data_A, idx_A, subj, cortical_n, depth + 1, branch + "A")
    bicluster(data_B, idx_B, subj, cortical_n, depth + 1, branch + "B")


def split(
    subj,
    original,
    pca_voxel_idxes,
    i_PC,
    cortical_n,
    branch="",
    split_threshold=4,
    split_ratio=9,
):
    """
    Input:
        original: PC projection matrix to be split by the ith PC (20000 x 20)
        cortical_mask: cortical mask (3d mask)
        pca_voxel_idxes: indexes to select the "best 20000 voxels" from cortical voxels (dim:20000 x 1)
    Returns:
        matrix_A: subset of projection matrix with the thresholded voxels by PCs
        idx_A: coritical length mask to pick for matrix A
        vol_A: the volume correspond to matrix_A
    """
    if len(pca_voxel_idxes) < 50:
        return
    if i_PC < split_threshold:
        idx = np.where(original[:, i_PC] > 0)[
            0
        ]  # n x 1 integer (n = # of chosen voxels)
        matrix_A = original[idx]  # n x 20
        idx_A = pca_voxel_idxes[
            idx
        ]  # n x 1 integers, used to index cortical length array to pick out voxels relevant in this pc split
        idx = np.where(original[:, i_PC] < 0)[0]
        matrix_B = original[idx]
        idx_B = pca_voxel_idxes[idx]
        if split_here(idx_A, idx_B, ratio=split_ratio):
            # positive
            mask_A = np.zeros(cortical_n).astype(bool)  # 100k x 1
            mask_A[idx_A] = True
            VOLS["PC %d %s" % (i_PC, branch + "A")] = make_volume(
                subj, matrix_A[:, i_PC], mask_A
            )

            # negative
            mask_B = np.zeros(cortical_n).astype(bool)
            mask_B[idx_B] = True
            VOLS["PC %d %s" % (i_PC, branch + "B")] = make_volume(
                subj, matrix_B[:, i_PC], mask_B
            )

            split(
                subj,
                matrix_A,
                idx_A,
                i_PC + 1,
                cortical_n,
                branch + "A",
                split_ratio=split_ratio,
            )
            split(
                subj,
                matrix_B,
                idx_B,
                i_PC + 1,
                cortical_n,
                branch + "B",
                split_ratio=split_ratio,
            )
        else:  # this split is ineffective and should skip to the next PC
            split(subj, original, pca_voxel_idxes, i_PC + 1, cortical_n, branch + "X")


def group_split(
    original_list,
    pca_voxel_idxes_list,
    i_PC,
    cortical_n_list,
    branch="",
    split_threshold=4,
    split_ratio=99,
    consistency_threshold=0.1,
):
    """
    Same function as split but everything is a list of 8 subjects.
    Procedure: Do the split per subject, compute consistency, if not consistent, skip this split
    """

    subjs = np.arange(8)
    matrix_A_list, matrix_B_list, idx_A_list, idx_B_list = [], [], [], []

    idxes_length = [len(idxes) for idxes in pca_voxel_idxes_list]
    if min(idxes_length) < 50:
        return

    skip = False
    if i_PC < split_threshold:
        for s in subjs:
            original, pca_voxel_idxes, cortical_n = (
                original_list[s],
                pca_voxel_idxes_list[s],
                cortical_n_list[s],
            )
            idx = np.where(original[:, i_PC] > 0)[
                0
            ]  # n x 1 integer (n = # of chosen voxels)
            matrix_A = original[idx]  # n x 20
            idx_A = pca_voxel_idxes[
                idx
            ]  # n x 1 integers, used to index cortical length array to pick out voxels relevant in this pc split
            idx = np.where(original[:, i_PC] < 0)[0]
            matrix_B = original[idx]
            idx_B = pca_voxel_idxes[idx]
            if not split_here(
                idx_A, idx_B, ratio=split_ratio
            ):  # this split is ineffective and should skip to the next PC
                skip = True
                break

            # positive
            mask_A = np.zeros(cortical_n).astype(bool)  # 100k x 1
            mask_A[idx_A] = True
            VOLS[s]["PC %d %s" % (i_PC, branch + "A")] = make_volume(
                s + 1, matrix_A[:, i_PC], mask_A
            )

            # negative
            mask_B = np.zeros(cortical_n).astype(bool)
            mask_B[idx_B] = True
            VOLS[s]["PC %d %s" % (i_PC, branch + "B")] = make_volume(
                s + 1, matrix_B[:, i_PC], mask_B
            )

            matrix_A_list.append(matrix_A)
            matrix_B_list.append(matrix_B)
            idx_A_list.append(idx_A)
            idx_B_list.append(idx_B)

        if not skip:
            if not consistent_among_group(
                "PC %d %s" % (i_PC, branch), threshold=consistency_threshold
            ):
                print("skipping %s(A/B)" % branch)
                skip = True

        if skip:
            group_split(
                original_list,
                pca_voxel_idxes_list,
                i_PC + 1,
                cortical_n_list,
                branch + "X",
                split_ratio=split_ratio,
                consistency_threshold=consistency_threshold,
            )
        else:
            group_split(
                matrix_A_list,
                idx_A_list,
                i_PC + 1,
                cortical_n_list,
                branch + "A",
                split_ratio=split_ratio,
                consistency_threshold=consistency_threshold,
            )
            group_split(
                matrix_B_list,
                idx_B_list,
                i_PC + 1,
                cortical_n_list,
                branch + "B",
                split_ratio=split_ratio,
                consistency_threshold=consistency_threshold,
            )


def consistent_among_group(branch_name, threshold, n_subj=8):
    vol_mask = cortex.db.get_mask("fsaverage", "atlas")
    vals = np.zeros((n_subj, np.sum(vol_mask)))
    for split in ["A", "B"]:
        for s in range(n_subj):
            try:
                vol = VOLS[s][branch_name + split]
            except KeyError:
                print(branch_name + split)
                print(VOLS[s].keys())

            vol.data[np.isnan(vol.data)] = 0
            mni_vol = project_vols_to_mni(s + 1, vol)
            # mask the values and compute correlations across subj
            vals[s, :] = mni_vol[vol_mask]

        corr = np.corrcoef(vals)
        score = np.sum(np.triu(corr, k=1)) / (n_subj * (n_subj - 1) / 2)
        if score < threshold:
            print(score)
            print(branch_name)
            return False
    return True


def split_here(idx_A, idx_B, ratio):
    if ratio == 999:  # not limiting splits
        return True
    if (len(idx_A) < 5) or (len(idx_B) < 5):
        return False
    elif len(idx_A) / len(idx_B) > ratio:
        return False
    elif len(idx_B) / len(idx_A) > ratio:
        return False
    else:
        return True


def make_subj_tree(subj, split_ratio=999, visualize=False):
    subj_proj = np.load(
        "%s/output/pca/%s/%s/subj%02d/pca_projections.npy"
        % (OUTPUT_ROOT, args.model, args.name_modifier, subj)
    ).T  # 100k x 20 vectors
    subj_mask = np.load(
        "%s/output/pca/%s/%s/pca_voxels/pca_voxels_subj%02d.npy"
        % (OUTPUT_ROOT, args.model, args.name_modifier, subj)
    )  # 1 x 100k vectors

    cortical_mask = np.load(
        "%s/output/voxels_masks/subj%d/cortical_mask_subj%02d.npy"
        % (OUTPUT_ROOT, subj, subj)
    )  # 1 x 100k vectors

    proj_val_only = subj_proj[subj_mask, :]
    proj_val_only /= proj_val_only.std(axis=0)
    pca_voxel_idxes = np.where(subj_mask == True)[0]

    split(
        subj,
        proj_val_only,
        pca_voxel_idxes,
        i_PC=0,
        cortical_n=np.sum(cortical_mask),
        split_threshold=5,
        split_ratio=split_ratio,
    )
    if visualize:
        subj_port = "2111" + str(subj)
        cortex.webgl.show(data=VOLS, port=int(subj_port), recache=False)
        import pdb

        pdb.set_trace()


def make_volume(subj, vals, pca_mask, vmin=-2, vmax=2, cmap="BrBG_r"):
    import cortex

    mask = cortex.utils.get_cortical_mask(
        "subj%02d" % subj, "func1pt8_to_anat0pt8_autoFSbbr"
    )
    cortical_mask = np.load(
        "%s/output/voxels_masks/subj%d/cortical_mask_subj%02d.npy"
        % (OUTPUT_ROOT, subj, subj)
    )
    cortical_vals = np.zeros(np.sum(cortical_mask)) * np.nan
    cortical_vals[pca_mask] = vals

    # projecting value back to 3D space
    three_d_vals = project_vals_to_3d(cortical_vals, cortical_mask)

    vol_data = cortex.Volume(
        three_d_vals,
        "subj%02d" % subj,
        "func1pt8_to_anat0pt8_autoFSbbr",
        mask=mask,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    return vol_data


def single_split(subj, original, pca_voxel_idxes, i_PC, cortical_n, vol_name=""):
    """
    Input:
        original: PC projection matrix to be split by the ith PC (20000 x 20)
        cortical_mask: cortical mask (3d mask)
        pca_voxel_idxes: indexes to select the "best 20000 voxels" from cortical voxels (dim:20000 x 1)
    Returns:
        matrix_A: subset of projection matrix with the thresholded voxels by PCs
        idx_A: coritical length mask to pick for matrix A
        vol_A: the volume correspond to matrix_A
        branch
    """
    vol_name += "PC" + str(i_PC)
    idx = np.where(original[:, i_PC] > 0)[0]  # n x 1 integer (n = # of chosen voxels)
    matrix_A = original[idx]  # n x 20
    idx_A = pca_voxel_idxes[
        idx
    ]  # n x 1 integers, used to index cortical length array to pick out voxels relevant in this pc split
    idx = np.where(original[:, i_PC] < 0)[0]
    matrix_B = original[idx]
    idx_B = pca_voxel_idxes[idx]
    # positive
    mask_A = np.zeros(cortical_n).astype(bool)  # 100k x 1
    mask_A[idx_A] = True
    vol_name_A = vol_name + "A"
    VOLS[vol_name_A] = make_volume(subj, matrix_A[:, i_PC], mask_A)

    # negative
    mask_B = np.zeros(cortical_n).astype(bool)
    vol_name_B = vol_name + "B"
    mask_B[vol_name_B] = make_volume(subj, matrix_B[:, i_PC], mask_B)

    return matrix_A, matrix_B, idx_A, idx_B, vol_name_A, vol_name_B


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--subj", default=1, type=int)
    parser.add_argument("--model", default="clip")
    parser.add_argument("--name_modifier", default="best_20000_nc")
    parser.add_argument("--compute_consistency", default=False, action="store_true")
    parser.add_argument("--split_single_subject", default=False, action="store_true")
    parser.add_argument("--group_split", default=False, action="store_true")
    parser.add_argument("--show_group_split", default=False, action="store_true")
    parser.add_argument("--recursive_biclustering", default=False, action="store_true")
    parser.add_argument("--manual_split", default=False, action="store_true")

    parser.add_argument

    args = parser.parse_args()

    OUTPUT_ROOT = "."

    if args.split_single_subject:
        VOLS = {}
        make_subj_tree(args.subj, visualize=True)

    if args.compute_consistency:
        from visualize_in_pycortex import project_vols_to_mni

        save_name = "PC_split_" + args.name_modifier
        n_subj = 8
        all_volumes = []
        for s in np.arange(n_subj):
            make_subj_tree(s + 1, split_ratio=999)
            vols = VOLS.copy()
            all_volumes.append(vols)
            VOLS = {}

        # make them pycortex volume if they are not and project them to mni
        corrs, corrs_mean, labels = [], [], []
        vol_mask = cortex.db.get_mask("fsaverage", "atlas")

        n_nodes = len(
            all_volumes[0].keys()
        )  # some subject might not have this many nodes
        for i in tqdm(range(n_nodes)):
            vals = np.zeros((n_subj, np.sum(vol_mask)))
            split_name = list(all_volumes[0].keys())[
                i
            ]  # get the split name from subj 1
            for s in range(n_subj):
                try:
                    vol = all_volumes[s][split_name]
                    vol.data[np.isnan(vol.data)] = 0
                    mni_vol = project_vols_to_mni(s + 1, vol)
                    # mask the values and compute correlations across subj
                    vals[s, :] = mni_vol[vol_mask]
                    skip_this_split = False
                except KeyError:
                    skip_this_split = True
                    break

            if not skip_this_split:
                labels.append(split_name)
                corr = np.corrcoef(vals)
                corrs.append(corr)
                corrs_mean.append(
                    np.sum(np.triu(corr, k=1)) / (n_subj * (n_subj - 1) / 2)
                )

        np.save(
            "%s/output/pca/%s/%s_corr_across_subjs.npy"
            % (OUTPUT_ROOT, args.model, save_name),
            corrs,
        )

        import matplotlib.pyplot as plt

        plt.figure(figsize=(20, 15))
        plt.plot(np.arange(len(labels)), corrs_mean)
        plt.xlabel("Splits")
        plt.xticks(np.arange(len(labels)), labels=labels, rotation=45)
        plt.ylabel("correlation")
        plt.grid()
        plt.savefig("figures/PCA/%s_%s_corr_across_subjs.png" % (args.model, save_name))

    if args.group_split:
        from visualize_in_pycortex import project_vols_to_mni

        subjs = np.arange(1, 9)
        cortical_n_list, proj_val_list, pca_voxel_idxes_list = [], [], []
        VOLS = [{}, {}, {}, {}, {}, {}, {}, {}]
        for s in subjs:
            cortical_mask = np.load(
                "%s/output/voxels_masks/subj%d/cortical_mask_subj%02d.npy"
                % (OUTPUT_ROOT, s, s)
            )  # 1 x 100k vectors
            cortical_n_list.append(np.sum(cortical_mask))

            subj_proj = np.load(
                "%s/output/pca/%s/%s/subj%02d/pca_projections.npy"
                % (OUTPUT_ROOT, args.model, args.name_modifier, s)
            ).T  # 100k x 20 vectors
            subj_mask = np.load(
                "%s/output/pca/%s/%s/pca_voxels/pca_voxels_subj%02d.npy"
                % (OUTPUT_ROOT, args.model, args.name_modifier, s)
            )  # 1 x 100k vectors

            proj_val_only = subj_proj[subj_mask, :]
            proj_val_only /= proj_val_only.std(axis=0)
            proj_val_list.append(proj_val_only)
            pca_voxel_idxes_list.append(np.where(subj_mask == True)[0])

        group_split(
            proj_val_list,
            pca_voxel_idxes_list,
            0,
            cortical_n_list,
            branch="",
            split_threshold=4,
            split_ratio=99,
            consistency_threshold=0.05,
        )

        import pickle

        pickle.dump(
            VOLS,
            open(
                "%s/output/pca/%s/%s/tree/tree.pkl"
                % (OUTPUT_ROOT, args.model, args.name_modifier),
                "wb",
            ),
        )

    if args.show_group_split:
        import pickle

        cortical_mask = np.load(
            "%s/output/voxels_masks/subj%d/cortical_mask_subj%02d.npy"
            % (OUTPUT_ROOT, args.subj, args.subj)
        )
        subj_mask = np.load(
            "%s/output/pca/%s/%s/pca_voxels/pca_voxels_subj%02d.npy"
            % (OUTPUT_ROOT, args.model, args.name_modifier, args.subj)
        )
        subj_mask_3d = project_vals_to_3d(subj_mask, cortical_mask).astype(bool)

        VOLS = pickle.load(
            open(
                "%s/output/pca/%s/%s/tree/tree.pkl"
                % (OUTPUT_ROOT, args.model, args.name_modifier),
                "rb",
            )
        )
        subj_vol = VOLS[args.subj - 1]
        for v in subj_vol.values():
            v.data[~subj_mask_3d] = np.nan

        cortex.webgl.show(data=subj_vol, recache=False)
        import pdb

        pdb.set_trace()

    if args.recursive_biclustering:
        VOLS = {}
        # subj = np.arange(1,9)
        subj_proj = np.load(
            "%s/output/pca/%s/%s/subj%02d/pca_projections.npy"
            % (OUTPUT_ROOT, args.model, args.name_modifier, args.subj)
        ).T  # 100k x 20 vectors
        subj_mask = np.load(
            "%s/output/pca/%s/%s/pca_voxels/pca_voxels_subj%02d.npy"
            % (OUTPUT_ROOT, args.model, args.name_modifier, args.subj)
        )  # 1 x 100k vectors
        cortical_mask = np.load(
            "%s/output/voxels_masks/subj%d/cortical_mask_subj%02d.npy"
            % (OUTPUT_ROOT, args.subj, args.subj)
        )  # 1 x 100k vectors
        proj_val_only = subj_proj[subj_mask, :]
        proj_val_only /= proj_val_only.std(axis=0)
        pca_voxel_idxes = np.where(subj_mask == True)[0]
        bicluster(proj_val_only, pca_voxel_idxes, args.subj, np.sum(cortical_mask))
        cortex.webgl.show(data=VOLS, recache=False)

        import pdb

        pdb.set_trace()

    if args.manual_split:
        # 0+4
        VOLS = {}
        # subj = np.arange(1,9)
        subj_proj = np.load(
            "%s/output/pca/%s/%s/subj%02d/pca_projections.npy"
            % (OUTPUT_ROOT, args.model, args.name_modifier, args.subj)
        ).T  # 100k x 20 vectors
        subj_mask = np.load(
            "%s/output/pca/%s/%s/pca_voxels/pca_voxels_subj%02d.npy"
            % (OUTPUT_ROOT, args.model, args.name_modifier, args.subj)
        )  # 1 x 100k vectors
        cortical_mask = np.load(
            "%s/output/voxels_masks/subj%d/cortical_mask_subj%02d.npy"
            % (OUTPUT_ROOT, args.subj, args.subj)
        )  # 1 x 100k vectors
        proj_val_only = subj_proj[subj_mask, :]
        proj_val_only /= proj_val_only.std(axis=0)
        pca_voxel_idxes = np.where(subj_mask == True)[0]
        matrix_A, matrix_B, idx_A, idx_B, vol_name_A, vol_name_B = single_split(
            args.subj,
            proj_val_only,
            pca_voxel_idxes,
            0,
            np.sum(cortical_mask),
            vol_name="",
        )

        _ = single_split(
            args.subj, matrix_A, idx_A, 4, np.sum(cortical_mask), vol_name_A
        )
        _ = single_split(
            args.subj, matrix_B, idx_B, 4, np.sum(cortical_mask), vol_name_B
        )

        # 0+6
        matrix_A, matrix_B, idx_A, idx_B, vol_name_A, vol_name_B = single_split(
            args.subj,
            proj_val_only,
            pca_voxel_idxes,
            0,
            np.sum(cortical_mask),
            vol_name="",
        )

        _ = single_split(
            args.subj, matrix_A, idx_A, 6, np.sum(cortical_mask), vol_name_A
        )
        _ = single_split(
            args.subj, matrix_B, idx_B, 6, np.sum(cortical_mask), vol_name_B
        )

        cortex.webgl.show(data=VOLS, recache=False)

        import pdb

        pdb.set_trace()
