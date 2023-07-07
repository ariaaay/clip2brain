import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cortex

from visualize_in_pycortex import project_vols_to_mni, make_pc_volume

OUTPUT_ROOT = "."


def analyze_data_correlation_in_mni(
    data, model, save_name, subjs, dim=0, volumes=False, xtick_label=None
):
    n_subj = len(data)

    # dim is number of volumes per subjects
    if dim == 0:
        dim = len(data[0])

    # make them pycortex volume if they are not and project them to mni
    corrs, corrs_mean = [], []
    vol_mask = cortex.db.get_mask("fsaverage", "atlas")
    # print(vol_mask.shape)
    # print(np.sum(vol_mask))

    for i in tqdm(range(dim)):
        vals = np.zeros((n_subj, np.sum(vol_mask)))
        for s in range(n_subj):
            if not volumes:
                vol = make_pc_volume(subjs[s], data[s][i])
            else:
                vol = data[s][i]
            mni_vol = project_vols_to_mni(subjs[s], vol)

            # mask the values and compute correlations across subj
            vals[s, :] = mni_vol[vol_mask]

        corr = np.corrcoef(vals)
        corrs.append(corr)
        corrs_mean.append(np.sum(np.triu(corr, k=1)) / (n_subj * (n_subj - 1) / 2))

    np.save(
        "%s/output/pca/%s/%s_corr_across_subjs.npy" % (OUTPUT_ROOT, model, save_name),
        corrs,
    )

    plt.plot(np.arange(dim), corrs_mean)
    plt.xlabel(save_name)
    if xtick_label is not None:
        plt.xticks(np.arange(dim), labels=xtick_label, rotation=90)
    plt.ylabel("correlation")
    plt.legend()
    plt.savefig("figures/PCA/%s_%s_corr_across_subjs.png" % (model, save_name))


if __name__ == "__main__":
    subjs = [1, 2, 5, 7]
    model = "clip"
    # load all PCS from all four subjs
    all_PCs = []
    for subj in subjs:
        all_PCs.append(
            np.load(
                "%s/output/pca/%s/subj%02d/%s_pca_group_components.npy"
                % (OUTPUT_ROOT, model, subj, model)
            )
        )

    analyze_data_correlation_in_mni(all_PCs, model, save_name="PC", subjs=subjs, dim=20)
