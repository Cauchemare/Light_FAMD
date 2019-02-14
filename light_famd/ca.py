"""Correspondence Analysis (CA)"""
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn import base
from sklearn import utils


from . import svd


class CA(base.BaseEstimator, base.TransformerMixin):

    def __init__(self, n_components=2, n_iter=10, copy=True, check_input=True, random_state=None,
                 engine='auto'):
        self.n_components = n_components
        self.n_iter = n_iter
        self.copy = copy
        self.check_input = check_input
        self.random_state = random_state
        self.engine = engine

    def fit(self, X, y=None):

        # Check input
        if self.check_input:
            utils.check_array(X)

        # Check all values are positive
        if np.any(X < 0):
            raise ValueError("All values in X should be positive")


        if isinstance(X, pd.DataFrame):
            X = X.values

        if self.copy:
            X = np.copy(X)

        # Compute the correspondence matrix which contains the relative frequencies
        X = X / np.sum(X)

        # Compute standardised residuals
        self.r=  X.sum(axis=1)
        self.c=  X.sum(axis=0)
        S = sparse.diags(self.r ** -0.5) @ (X - np.outer(self.r, self.c)) @ sparse.diags(self.c ** -0.5)

        # Compute SVD on the standardised residuals
        self.U_, self.singular_values_, self.components_ = svd.compute_svd(
            X=S,
            n_components=self.n_components,
            n_iter=self.n_iter,
            random_state=self.random_state,
            engine=self.engine
        )


        # Compute total inertia
        if not hasattr(self,'total_inertia_'):
            self.total_inertia_ = np.einsum('ij,ji->',S,S.T)

        return self

    def transform(self, X):
        """Computes the row principal coordinates of a dataset.

        Same as calling `row_coordinates`. In most cases you should be using the same
        dataset as you did when calling the `fit` method. You might however also want to included
        supplementary data.
        """
        utils.validation.check_is_fitted(self, 'singular_values_')
        if self.check_input:
            utils.check_array(X)
        return self._transform(X)

    @property
    def explained_variance_(self):
        """The eigenvalues associated with each principal component."""
        utils.validation.check_is_fitted(self, 'singular_values_')
        return self.singular_values_ **2 

    @property
    def explained_variance_ratio_(self):
        """The percentage of explained inertia per principal component."""
        utils.validation.check_is_fitted(self, 'total_inertia_')
        return [eig / self.total_inertia_ for eig in self.explained_variance_]

    def _transform(self, X):
        """The row principal coordinates."""
        if isinstance(X, pd.SparseDataFrame):
            X = X.to_coo()
        elif isinstance(X, pd.DataFrame):
            X = X.values

        if self.copy:
            X = X.copy()

        # Normalise the rows so that they sum up to 1
        if isinstance(X, np.ndarray):
            X = X / X.sum(axis=1)[:, None]
        else:
            X = X / X.sum(axis=1)

        return X @ sparse.diags(self.c ** -0.5) @ self.components_.T






