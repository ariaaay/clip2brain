"""
This scripts takes an features space and runs encoding models (ridge regression) to
predict NSD brain data.
"""
import argparse
import numpy as np
from encodingmodel.encoding_model import fit_encoding_model, bootstrap_test
from featureprep.feature_prep import (
    get_preloaded_features,
    extract_feature_with_image_order,
)
from util.data_util import load_subset_trials


def run(
    fm, br, model_name, test, fix_testing, cv, saving_dir,
):
    if test:
        print("Running Bootstrap Test")
        bootstrap_test(
            fm, br, model_name=model_name, subj=args.subj, saving_dir=saving_dir,
        )

    else:
        print("Fitting Encoding Models")
        fit_encoding_model(
            fm,
            br,
            model_name=model_name,
            subj=args.subj,
            fix_testing=fix_testing,
            cv=cv,
            saving=True,
            saving_dir=saving_dir,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Please specify features to model from and parameters of the encoding model."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="convnet",
        nargs="+",
        help="input the names of the features.",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default=None,
        help="input name of the layer. e.g. input_layer1",
    )
    parser.add_argument("--test", action="store_true", help="Run bootstrap testing.")
    parser.add_argument(
        "--subj",
        type=int,
        default=1,
        help="Specify which subject to build model on. Currently it supports subject 1, 2, 5, 7",
    )
    parser.add_argument(
        "--fix_testing",
        action="store_true",
        help="Use fixed sampling for training and testing (for model performance comparison purpose)",
    )
    parser.add_argument(
        "--cv", action="store_true", default=False, help="run cross-validation."
    )
    parser.add_argument(
        "--get_features_only",
        action="store_true",
        default=False,
        help="only generate and save the feature matrix but not running the encoding models (for preloaded features)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Specify the path to the intermediate output directory that contains voxels masks etc",
    )

    parser.add_argument(
        "--saving_dir",
        type=str,
        default="output",
        help="saving dir for the encoding model results. Default is the same as the output but could be somewhere else.",
    )

    parser.add_argument(
        "--features_dir",
        type=str,
        default="features",
        help="Specify the path to the features directory",
    )
    parser.add_argument(
        "--feature_matrix",
        type=str,
        default=None,
        help="Specify the path to the feature matrix (should be a numpy array)",
    )
    parser.add_argument(
        "--feature_order",
        type=str,
        default=None,
        help="Specify the path to the ordering of the feature matrix (should be a numpy array)",
    )
    parser.add_argument(
        "--model_name_to_save",
        type=str,
        default=None,
        help="Specify a name to save the performance with",
    )
    parser.add_argument(
        "--subset_data",
        type=str,
        default=None,
        help="specify a category to subset training and testing data",
    )

    args = parser.parse_args()
    print(args)

    brain_path = (
        "%s/cortical_voxels/averaged_cortical_responses_zscored_by_run_subj%02d.npy"
        % (args.output_dir, args.subj)
    )

    # Load brain data
    br_data = np.load(brain_path)
    # deal with voxels that are zeros in runs and therefore cause nan values in zscoring
    # only happens in some subjects (e.g. subj5)
    try:
        non_zero_mask = np.load(
            "%s/voxels_masks/subj%d/nonzero_voxels_subj%02d.npy"
            % (args.output_dir, args.subj, args.subj)
        )
        print("Masking zero voxels...")
        br_data = br_data[:, non_zero_mask]
    except FileNotFoundError:
        pass

    # dead with trials that are nan because subjects has never seen the images
    trial_mask = np.sum(np.isnan(br_data), axis=1) <= 0
    br_data = br_data[trial_mask, :]

    print("NaNs? Finite?:")
    print(np.any(np.isnan(br_data)))
    print(np.all(np.isfinite(br_data)))
    print("Brain response size is: " + str(br_data.shape))

    stimulus_list = np.load(
        "%s/coco_ID_of_repeats_subj%02d.npy" % (args.output_dir, args.subj)
    )

    # Load feature spaces
    if args.feature_matrix is not None:  # for general design matrix input
        feature_mat_unordered = np.load(args.feature_matrix)
        image_order = np.laod(args.image_order)
        model_name_to_save = args.model_name_to_save
        feature_mat = extract_feature_with_image_order(
            stimulus_list, feature_mat_unordered, image_order
        )
    else:
        if args.layer is not None:
            model_name_to_save = args.model[0] + "_" + args.layer
        else:
            model_name_to_save = args.model[0]

        feature_mat = get_preloaded_features(
            args.subj,
            stimulus_list,
            args.model[0],
            layer=args.layer,
            features_dir=args.features_dir,
        )

        if len(args.model) > 1:
            for model in args.model[1:]:
                more_feature = get_preloaded_features(
                    args.subj, stimulus_list, model, features_dir=args.features_dir
                )
                feature_mat = np.hstack((feature_mat, more_feature))

                model_name_to_save += "_" + model

    feature_mat = feature_mat[trial_mask, :]

    if args.subset_data is not None:
        subset_cat = args.subset_data
        print("Subsetting training data with criteria: " + subset_cat)
        if (
            "no" in subset_cat
        ):  # selecting the trials that didnt contain certain categories
            subset_cat = subset_cat.split("_")[-1]
            subset_trial_id = load_subset_trials(stimulus_list, subset_cat, negcat=True)
        else:
            subset_trial_id = load_subset_trials(stimulus_list, subset_cat)
        br_data = br_data[subset_trial_id, :]
        feature_mat = feature_mat[subset_trial_id, :]
        model_name_to_save += "_" + args.subset_data + "_subset"

    print("=======================")
    print("Running ridge encoding model on :")
    print(model_name_to_save)

    print("Feature size is: " + str(feature_mat.shape))
    print("=======================")

    if not args.get_features_only:
        run(
            feature_mat,
            br_data,
            model_name=model_name_to_save,
            test=args.test,
            fix_testing=args.fix_testing,
            cv=args.cv,
            saving_dir=args.saving_dir,
        )
