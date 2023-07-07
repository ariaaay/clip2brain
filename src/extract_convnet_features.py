import argparse
import copy

import pandas as pd
import numpy as np
from torch._C import Value
from tqdm import tqdm
from PIL import Image

from sklearn.decomposition import PCA

import torch
import torch.nn as nn

# import torchvision
from torchvision import transforms, utils, models

import torchextractor as tx

device = "cuda" if torch.cuda.is_available() else "cpu"
stimuli_dir = "/lab_data/tarrlab/common/datasets/NSD_images/images"
preprocess = transforms.Compose(
    [
        # transforms.Resize(375),
        transforms.ToTensor()
    ]
)


def extract_resnet_prePCA_feature():
    layers = ["layer1", "layer2", "layer3", "layer4", "layer4.2.relu"]
    # model, preprocess = clip.load("RN50", device=device)
    model = models.resnet50(pretrained=True)
    model = tx.Extractor(model, layers)
    compressed_features = [copy.copy(e) for _ in range(len(layers)) for e in [[]]]
    subsampling_size = 5000

    print("Extracting ResNet features")
    for cid in tqdm(all_coco_ids):
        with torch.no_grad():
            image_path = "%s/%s.jpg" % (stimuli_dir, cid)
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

            _, features = model(image)

            for i, f in enumerate(features.values()):
                # print(f.size())
                if len(f.size()) > 3:
                    c = f.data.shape[1]  # number of channels
                    k = int(np.floor(np.sqrt(subsampling_size / c)))
                    tmp = nn.functional.adaptive_avg_pool2d(f.data, (k, k))
                    # print(tmp.size())
                    compressed_features[i].append(tmp.squeeze().cpu().numpy().flatten())
                else:
                    compressed_features[i].append(
                        f.squeeze().data.cpu().numpy().flatten()
                    )

    for l, f in enumerate(compressed_features):
        np.save("%s/convnet_resnet_prePCA_%01d.npy" % (feature_output_dir, l), f)


def extract_visual_resnet_feature():
    for l in range(7):
        try:
            f = np.load("%s/convnet_resnet_prePCA_%01d.npy" % (feature_output_dir, l))
        except FileNotFoundError:
            extract_resnet_prePCA_feature()
            f = np.load("%s/convnet_resnet_prePCA_%01d.npy" % (feature_output_dir, l))

        print("Running PCA")
        print("feature shape: ")
        print(f.shape)
        pca = PCA(n_components=min(f.shape[0], 64), svd_solver="auto")

        fp = pca.fit_transform(f)
        print("Feature %01d has shape of:" % l)
        print(fp.shape)

        np.save("%s/resnet_%01d.npy" % (feature_output_dir, l), fp)


def extract_resnet_last_layer_feature(cid=None, saving=True):
    model = models.resnet50(pretrained=True).to(device)
    model = tx.Extractor(model, "avgpool")

    # print("Extracting ResNet features")
    if cid is None:
        output = list()
        for cid in tqdm(all_coco_ids):
            with torch.no_grad():
                image_path = "%s/%s.jpg" % (stimuli_dir, cid)
                image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

                _, features = model(image)
                output.append(
                    features["avgpool"].squeeze().data.cpu().numpy().flatten()
                )
    else:
        with torch.no_grad():
            image_path = "%s/%s.jpg" % (stimuli_dir, cid)
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
            _, features = model(image)
            output = features["avgpool"].data.squeeze().cpu()
    if saving:
        np.save("%s/convnet_resnet_avgpool.npy" % feature_output_dir, output)

    return output


if __name__ == "__main__":
    # stim = pd.read_pickle(
    #     "/lab_data/tarrlab/common/datasets/NSD/nsddata/experiments/nsd/nsd_stim_info_merged.pkl"
    # )
    # all_coco_ids = stim.cocoId
    # all_images_paths = list()
    # all_images_paths += ["%s/%s.jpg" % (stimuli_dir, id) for id in all_coco_ids]
    # print("Number of Images: {}".format(len(all_images_paths)))
    parser = argparse.ArgumentParser()
    parser.add_argument("--subj", default=1, type=int)
    parser.add_argument(
        "--feature_dir",
        type=str,
        default="/user_data/yuanw3/project_outputs/NSD/features",
    )
    parser.add_argument(
        "--project_output_dir",
        type=str,
        default="/user_data/yuanw3/project_outputs/NSD/output",
    )
    args = parser.parse_args()
    feature_output_dir = "%s/subj%01d" % (args.feature_dir, args.subj)
    # feature_output_dir = "/lab_data/tarrlab/common/datasets/features/NSD/"
    all_coco_ids = np.load(
        "%s/coco_ID_of_repeats_subj%02d.npy" % (args.project_output_dir, args.subj)
    )

    extract_resnet_last_layer_feature()
