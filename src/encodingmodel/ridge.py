"""ridge.py: pytorch implementation of grid-searching ridge regression.

Ridge solutions for multiple regularization values are computed efficiently by
using the Woodbury identity. With X (n x d) representing the feature matrix, and
y (n x 1) the outcomes, the ridge solution is given by

    β = (X'X + l*I)^{-1}X'y

where l is the regularization coefficient. This can be reduced to

    (1/l)*(X'y - X'V(e + l*I)^{-1}(X'V)'X'y)

where Ue^{1/2}V' is the singular-value decomposition of X'. Since (e + lI) is a
diagonal matrix, its inverse can be computed efficiently simply by taking the
reciprocal of the diagonal elements. Then, (X'V)'X'y is a vector; so it can be
multiplied by (e + lI)^{-1} just by scalar multiplication.
"""

import torch


def _validate_ls(ls):
    """Ensure that ls is a 1-dimensional torch float/double tensor."""
    try:
        assert isinstance(ls, torch.Tensor)
        assert ls.dtype is torch.float or ls.dtype is torch.double
        assert len(ls.shape) == 1
    except AssertionError:
        raise AttributeError(
            "invalid ls: should be 1-dimensional torch float/double tensor"
        )


def _validate_XY(X, Y):
    """Ensure that X and Y are 2-dimensional torch float/double tensors, with
    proper sizes."""
    try:
        for inp in [X, Y]:
            assert isinstance(inp, torch.Tensor)
            assert inp.dtype is torch.float or inp.dtype is torch.double
            assert len(inp.shape) == 2
        assert X.dtype is Y.dtype
        assert X.shape[0] == Y.shape[0]
    except AssertionError:
        raise AttributeError(
            "invalid inputs: X and Y should be float/double tensors of shape "
            "(n, d) and (n, m) respectively, where n is the number of samples, "
            "d is the number of features, and m is the number of outputs"
        )


class MultiRidge:

    """Ridge model for multiple outputs and regularization strengths. A separate
    model is fit for each (output, regularization) pair."""

    def __init__(self, ls, scale_X=True, scale_thresh=1e-8):
        """
        Arguments:
                       ls: 1-dimensional torch tensor of regularization values.
                  scale_X: whether or not to scale X.
             scale_thresh: when standardizing, standard deviations below this
                           value are not used (i.e. they are set to 1).
        """
        self.ls = ls
        self.scale_X = scale_X
        self.scale_thresh = scale_thresh
        self.X_t = None
        self.Xm = None
        self.Xs = None
        self.e = None
        self.Q = None
        self.Y = None
        self.Ym = None

    def fit(self, X, Y):
        """
        Arguments:
            X: 2-dimensional torch tensor of shape (n, d) where n is the number
               of samples, and d is the number of features.
            Y: 2-dimensional tensor of shape (n, m) where m is the number of
               targets.
        """
        self.Xm = X.mean(dim=0, keepdim=True)
        X = X - self.Xm
        if self.scale_X:
            self.Xs = X.std(dim=0, keepdim=True)
            self.Xs[self.Xs < self.scale_thresh] = 1
            X = X / self.Xs

        self.X_t = X.t()
        _, S, V = self.X_t.svd()
        self.e = S.pow_(2)
        self.Q = self.X_t @ V

        self.Y = Y
        self.Ym = Y.mean(dim=0)

        return self

    def _compute_pred_interms(self, y_idx, X_te_p):
        Y_j, Ym_j = self.Y[:, y_idx], self.Ym[y_idx]
        p_j = self.X_t @ (Y_j - Ym_j)
        r_j = self.Q.t() @ p_j
        N_te_j = X_te_p @ p_j
        return Ym_j, r_j, N_te_j

    def _predict_single(self, l, M_te, Ym_j, r_j, N_te_j):
        Yhat_te_j = (1 / l) * (N_te_j - M_te @ (r_j / (self.e + l))) + Ym_j
        return Yhat_te_j

    def _compute_single_beta(self, l, y_idx):
        Y_j, Ym_j = self.Y[:, y_idx], self.Ym[y_idx]
        beta = (1 / l) * (
            self.X_t @ (Y_j - Ym_j)
            - self.Q / (self.e + l) @ self.Q.t() @ self.X_t @ (Y_j - Ym_j)
        )
        return beta

    def get_model_weights_and_bias(self, l_idxs):
        betas = torch.zeros((self.X_t.shape[0], len(l_idxs)))
        for j, l_idx in enumerate(l_idxs):
            l = self.ls[l_idx]
            betas[:, j] = self._compute_single_beta(l, j)
        return betas, self.Ym

    def get_prediction_scores(self, X_te, Y_te, scoring):
        """Compute predictions for each (regulariztion, output) pair and return
        the scores as produced by the given scoring function.

        Arguments:
               X_te: 2-dimensional torch tensor of shape (n, d) where n is the
                     number of samples, and d is the number of features.
               Y_te: 2-dimensional tensor of shape (n, m) where m is the
                     number of targets.
            scoring: scoring function with signature scoring(y, yhat).

        Returns a (m, M) torch tensor of scores, where M is the number of
        regularization values.
        """
        X_te = X_te - self.Xm
        if self.scale_X:
            X_te = X_te / self.Xs
        M_te = X_te @ self.Q

        scores = torch.zeros(Y_te.shape[1], len(self.ls), dtype=X_te.dtype)
        for j, Y_te_j in enumerate(Y_te.t()):
            Ym_j, r_j, N_te_j = self._compute_pred_interms(j, X_te)
            for k, l in enumerate(self.ls):
                Yhat_te_j = self._predict_single(l, M_te, Ym_j, r_j, N_te_j)
                scores[j, k] = scoring(Y_te_j, Yhat_te_j).item()
        return scores

    def predict_single(self, X_te, l_idxs):
        """Compute a single prediction corresponding to each output.

        Arguments:
              X_te: 2-dimensional torch tensor of shape (n, d) where n is the
                    number of samples, and d is the number of features.
            l_idxs: iterable of length m (number of targets), with indexes
                    specifying the l value to use for each of the targets.

        Returns a (n, m) tensor of predictions.
        """
        X_te = X_te - self.Xm
        if self.scale_X:
            X_te = X_te / self.Xs

        M_te = X_te @ self.Q

        Yhat_te = []
        for j, l_idx in enumerate(l_idxs):
            Ym_j, r_j, N_te_j = self._compute_pred_interms(j, X_te)
            l = self.ls[l_idx]
            Yhat_te_j = self._predict_single(l, M_te, Ym_j, r_j, N_te_j)
            Yhat_te.append(Yhat_te_j)

        Yhat_te = torch.stack(Yhat_te, dim=1)
        return Yhat_te


class RidgeCVEstimator:
    def __init__(self, ls, cv, scoring, scale_X=True, scale_thresh=1e-8):
        """Cross-validated ridge estimator.

        Arguments:
                       ls: 1-dimensional torch tensor of regularization values.
                       cv: cross-validation object implementing split.
                  scoring: scoring function with signature scoring(y, yhat).
                  scale_X: whether or not to scale X.
             scale_thresh: when standardizing, standard deviations below this
                           value are not used (i.e. they are set to 1).
        """
        _validate_ls(ls)
        self.ls = ls
        self.cv = cv
        self.scoring = scoring
        self.scale_X = scale_X
        self.scale_thresh = scale_thresh
        self.base_ridge = None
        self.mean_cv_scores = None
        self.best_l_scores = None
        self.best_l_idxs = None

    def fit(self, X, Y, groups=None):
        """Fit ridge model to given data.

        Arguments:
                 X: 2-dimensional torch tensor of shape (n, d) where n is the
                    number of samples, and d is the number of features.
                 Y: 2-dimensional tensor of shape (n, m) where m is the number
                    of targets.
            groups: groups used for cross-validation; passed directly to
                    cv.split.

        A separate model is learned for each target i.e. Y[:, j].
        """
        _validate_XY(X, Y)
        cv_scores = []

        for idx_tr, idx_val in self.cv.split(X, Y, groups):
            X_tr, X_val = X[idx_tr], X[idx_val]
            Y_tr, Y_val = Y[idx_tr], Y[idx_val]

            self.base_ridge = MultiRidge(self.ls, self.scale_X, self.scale_thresh)
            self.base_ridge.fit(X_tr, Y_tr)
            split_scores = self.base_ridge.get_prediction_scores(
                X_val, Y_val, self.scoring
            )
            cv_scores.append(split_scores)
            del self.base_ridge

        cv_scores = torch.stack(cv_scores)
        self.mean_cv_scores = cv_scores.mean(dim=0)
        self.best_l_scores, self.best_l_idxs = self.mean_cv_scores.max(dim=1)
        self.base_ridge = MultiRidge(self.ls, self.scale_X, self.scale_thresh)
        self.base_ridge.fit(X, Y)
        return self

    def predict(self, X):
        """Predict using cross-validated model.

        Arguments:
            X_te: 2-dimensional torch tensor of shape (n, d) where n is the
                  number of samples, and d is the number of features.

        Returns a (n, m) matrix of predictions.
        """
        if self.best_l_idxs is None:
            raise RuntimeError("cannot predict without fitting")
        return self.base_ridge.predict_single(X, self.best_l_idxs)

    def get_model_weights_and_bias(self):
        if self.best_l_idxs is None:
            raise RuntimeError("cannot return weight without fitting")
        return self.base_ridge.get_model_weights_and_bias(self.best_l_idxs)
