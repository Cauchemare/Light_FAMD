"""Multiple Factor Analysis (MFA)"""

import numpy as np
import pandas as pd
from sklearn import utils
from .import util
from . import mca
from . import pca
from sklearn.utils.validation import check_is_fitted


class MFA(pca.PCA):

    def __init__(self, groups=None, normalize=True, n_components=2, n_iter=2,
                 copy=True, check_input=True, random_state=None, engine='auto'):
        super().__init__(
            rescale_with_mean=False,
            rescale_with_std=False,
            n_components=n_components,
            n_iter=n_iter,
            copy=copy,
            check_input=check_input,
            random_state=random_state,
            engine=engine
        )
        self.groups = groups
        self.normalize = normalize

    def fit(self, X, y=None):

        # Checks groups are provided
        if self.groups is None:
            raise ValueError('Groups have to be specified')

        # Check input
        if self.check_input:
            utils.check_array(X, dtype=[str, np.number])

        # Prepare input
        X = self._prepare_input(X)

        # Check group types are consistent
        self.partial_factor_analysis_ = {}
        for name, cols in self.groups.items():
            all_num = all(pd.api.types.is_numeric_dtype(X[c]) for c in cols)
            all_cat = all(pd.api.types.is_string_dtype(X[c]) for c in cols)
            if not (all_num or all_cat):
                raise ValueError('Not all columns in "{}" group are of the same type'.format(name))
        # Run a factor analysis in each group
            if all_num:
                fa = pca.PCA(
                    rescale_with_mean=self.rescale_with_mean,
                    rescale_with_std=self.rescale_with_std,
                    n_components=self.n_components,
                    n_iter=self.n_iter,
                    copy=self.copy,
                    random_state=self.random_state,
                    engine=self.engine
                )
            else:
                fa = mca.MCA(
                    n_components=self.n_components,
                    n_iter=self.n_iter,
                    copy=self.copy,
                    random_state=self.random_state,
                    engine=self.engine
                )
            self.partial_factor_analysis_[name] = fa.fit(X.loc[:, cols])

        # Fit the global PCA
        _X_global=  self._build_X_global(X)
        self._usecols= _X_global.columns
        super().fit(_X_global)  
        return self

    def _prepare_input(self, X):

        # Make sure X is a DataFrame for convenience
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Copy data
        if self.copy:
            X = X.copy()

        if self.normalize:
            # Scale continuous variables to unit variance
            num = X.select_dtypes(np.number).columns
            # If a column's cardinality is 1 then it's variance is 0 which can
            # can cause a division by 0
            X.loc[:, num] = (X.loc[:, num] - X.loc[:, num].mean()).apply(
                        lambda x: x / (np.sqrt((x ** 2).sum()) or 1), 
                        axis='rows')

        return X

    def _build_X_global(self, X):
        X_partials = []

        for name, cols in self.groups.items():
            X_partial = X.loc[:, cols]

            if self.partial_factor_analysis_[name].__class__.__name__ == 'MCA':
                check_is_fitted(self.partial_factor_analysis_[name],'_usecols')
                X_partial = self.partial_factor_analysis_[name].one_hot_.transform(X_partial).loc[:, self.partial_factor_analysis_[name]._usecols]
            
            X_partials.append(X_partial / self.partial_factor_analysis_[name].singular_values_[0])

        X_global = pd.concat(X_partials, axis='columns')
        X_global.index = X.index
        return X_global

    def transform(self, X):
        """Returns the row principal coordinates of a dataset."""
        utils.validation.check_is_fitted(self, 'singular_values_')
        if self.check_input:
        	utils.check_array(X, dtype=[str, np.number])
        X = self._prepare_input(X)
        X_global = self._build_X_global(X)
        
        return self._transform(X_global)


    def _transform(self, X_global):
        """Returns the row principal coordinates."""
        return  len(X_global) ** 0.5 * super()._transform(X_global)
    
    def fit_transform(self,X):
                # Checks groups are provided
        if self.groups is None:
            raise ValueError('Groups have to be specified')

        # Check input
        if self.check_input:
            utils.check_array(X, dtype=[str, np.number])

        # Prepare input
        X = self._prepare_input(X)

        # Check group types are consistent
        self.partial_factor_analysis_ = {}
        for name, cols in self.groups.items():
            all_num = all(pd.api.types.is_numeric_dtype(X[c]) for c in cols)
            all_cat = all(pd.api.types.is_string_dtype(X[c]) for c in cols)
            if not (all_num or all_cat):
                raise ValueError('Not all columns in "{}" group are of the same type'.format(name))
        # Run a factor analysis in each group
            if all_num:
                fa = pca.PCA(
                    rescale_with_mean=False,
                    rescale_with_std=False,
                    n_components=self.n_components,
                    n_iter=self.n_iter,
                    copy=True,
                    random_state=self.random_state,
                    engine=self.engine
                )
            else:
                fa = mca.MCA(
                    n_components=self.n_components,
                    n_iter=self.n_iter,
                    copy=self.copy,
                    random_state=self.random_state,
                    engine=self.engine
                )
            self.partial_factor_analysis_[name] = fa.fit(X.loc[:, cols])

        # Fit the global PCA
        X_global= self._build_X_global(X)
        self._usecols= X_global.columns
        super().fit(X_global)  

        return self._transform(X_global)
    
        
    def column_correlation(self,X,same_input=True):
        
        if   same_input: #X is fitted and the the data fitting and the data transforming is the same
            #not need to check_array since deriving from the same data
            X = self._prepare_input(X)
            X_global = self._build_X_global(X)
            X_t=self._transform(X_global)
            
        else:
            X_t=self.fit_transform(X)
            X = self._prepare_input(X)
            X_global = self._build_X_global(X)
        
        return util.df_correlation(X_t,X_global)
    


   

 
