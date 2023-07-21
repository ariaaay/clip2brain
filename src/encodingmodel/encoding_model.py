import os
from random import weibullvariate
import torch
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import (
    KFold,
    PredefinedSplit,
    train_test_split,
    ShuffleSplit,
)

# from sklearn.metrics import r2_score
from scipy.stats import pearsonr

from util.util import r2_score
from encodingmodel.ridge import RidgeCVEstimator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def scoring(y, yhat):
    return -torch.nn.functional.mse_loss(yhat, y)


def ridge_cv(
    X,
    y,
    tol=8,
    nfold=7,
    cv=False,
    fix_testing=False,
):
    # fix_tsesting can be True (42), False, and a seed
    if fix_testing is True:
        fix_testing_state = 42
    else:
        fix_testing_state = None

    # scoring = lambda y, yhat: -torch.nn.functional.mse_loss(yhat, y)

    alphas = torch.from_numpy(
        np.logspace(-tol, 1 / 2 * np.log10(X.shape[1]) + tol, 100)
    )

    # split train and test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=fix_testing_state
    )

    X_train = torch.from_numpy(X_train).to(dtype=torch.float64).to(device)
    y_train = torch.from_numpy(y_train).to(dtype=torch.float64).to(device)
    X_test = torch.from_numpy(X_test).to(dtype=torch.float64).to(device)

    # model selection

    if cv:
        kfold = KFold(n_splits=nfold)
    else:
        tr_index, _ = next(
            ShuffleSplit(test_size=0.15).split(
                X_train, y_train
            )  # split training and testing
        )
        # set predefined train and validation split
        test_fold = np.zeros(X_train.shape[0])
        test_fold[tr_index] = -1
        kfold = PredefinedSplit(test_fold)
        assert kfold.get_n_splits() == 1

    clf = RidgeCVEstimator(alphas, kfold, scoring, scale_X=False)

    print("Fitting ridge models...")

    clf.fit(X_train, y_train)

    weights, bias = clf.get_model_weights_and_bias()

    print("Making predictions using ridge models...")
    yhat = clf.predict(X_test).cpu().numpy()
    try:
        rsqs = r2_score(y_test, yhat)
    except ValueError:  # debugging for NaNs in subj 5
        print("Ytest: NaNs? Finite?")
        print(np.any(np.isnan(y_test)))
        print(np.all(np.isfinite(y_test)))
        print("Yhat: NaNs? Finite?")
        print(np.any(np.isnan(yhat)))
        print(np.all(np.isfinite(yhat)))

    corrs = [pearsonr(y_test[:, i], yhat[:, i]) for i in range(y_test.shape[1])]

    return (
        corrs,
        rsqs,
        clf.mean_cv_scores.cpu().numpy(),
        clf.best_l_scores.cpu().numpy(),
        clf.best_l_idxs.cpu().numpy(),
        [yhat, y_test],
        weights.cpu().numpy(),
        bias.cpu().numpy(),
        clf,
    )


def fit_encoding_model(
    X,
    y,
    model_name=None,
    subj=1,
    fix_testing=False,
    cv=False,
    saving=True,
    saving_dir=None,
):

    model_name += "_whole_brain"

    if cv:
        print("Running cross validation")

    outpath = "%s/encoding_results/subj%d" % (saving_dir, subj)
    if not os.path.isdir(outpath):
        os.makedirs(outpath)

    assert (
        y.shape[0] == X.shape[0]
    )  # test that shape of features spaces and the brain are the same

    cv_outputs = ridge_cv(
        X,
        y,
        cv=False,
        fix_testing=fix_testing,
    )

    if saving:
        pickle.dump(cv_outputs[0], open(outpath + "/corr_%s.p" % model_name, "wb"))

        if len(cv_outputs) > 0:
            pickle.dump(cv_outputs[1], open(outpath + "/rsq_%s.p" % model_name, "wb"))
            pickle.dump(
                cv_outputs[2],
                open(outpath + "/cv_score_%s.p" % model_name, "wb"),
            )
            pickle.dump(
                cv_outputs[3],
                open(outpath + "/l_score_%s.p" % model_name, "wb"),
            )
            pickle.dump(
                cv_outputs[4],
                open(outpath + "/best_l_%s.p" % model_name, "wb"),
            )

            if fix_testing:
                pickle.dump(
                    cv_outputs[5],
                    open(outpath + "/pred_%s.p" % model_name, "wb"),
                )

            np.save("%s/weights_%s.npy" % (outpath, model_name), cv_outputs[6])
            np.save("%s/bias_%s.npy" % (outpath, model_name), cv_outputs[7])
            pickle.dump(
                cv_outputs[8], open("%s/clf_%s.pkl" % (outpath, model_name), "wb")
            )

    return cv_outputs


def bootstrap_sampling(weights, bias, X_mean, X_test, y_test, repeat, seed):
    np.random.seed(seed)
    rsq_dist = list()
    label_idx = np.arange(X_test.shape[0])
    for _ in tqdm(range(repeat)):
        sampled_idx = np.random.choice(label_idx, replace=True, size=len(label_idx))
        X_test_sampled = X_test[sampled_idx, :]
        y_test_sampled = y_test[sampled_idx, :]
        yhat = (X_test_sampled - X_mean) @ weights + bias

        rsqs = r2_score(y_test_sampled, yhat.cpu().numpy())
        rsq_dist.append(rsqs)

    return rsq_dist


def bootstrap_test(
    X,
    y,
    model_name,
    repeat=2000,
    subj=1,
    saving_dir=None,
):
    print("Running bootstrap test of {} for {} times".format(model_name, repeat))

    # save rsq
    outpath = "%s/bootstrap/subj%d/" % (saving_dir, subj)
    if not os.path.isdir(outpath):
        os.makedirs(outpath)

    try:
        weights = np.load(
            "%s/encoding_results/subj%d/weights_%s_whole_brain.npy"
            % (saving_dir, subj, model_name)
        )
        bias = np.load(
            "%s/encoding_results/subj%d/bias_%s_whole_brain.npy"
            % (saving_dir, subj, model_name)
        )

    except FileNotFoundError:
        print("Running encoding models for bootstrap test")
        cv_outputs = fit_encoding_model(
            X,
            y,
            model_name=model_name,
            subj=subj,
            cv=False,
            saving=True,
            fix_testing=42,
            saving_dir=saving_dir,
        )
        weights, bias = cv_outputs[6], cv_outputs[7]
        print(weights.shape)

    X_train, X_test, _, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train = torch.from_numpy(X_train).to(dtype=torch.float64).to(device)
    X_test = torch.from_numpy(X_test).to(dtype=torch.float64).to(device)
    weights = torch.from_numpy(weights).to(dtype=torch.float64).to(device)
    bias = torch.from_numpy(bias).to(dtype=torch.float64).to(device)

    X_mean = X_train.mean(dim=0, keepdim=True)

    rsq_dists = bootstrap_sampling(
        weights, bias, X_mean, X_test, y_test, repeat=repeat, seed=41
    )
    np.save("%s/rsq_dist_%s_whole_brain.npy" % (outpath, model_name), rsq_dists)
