"""Principal Component Analysis (PCA)"""

import numpy as np
import pandas as pd
from sklearn import base
from sklearn import preprocessing
from sklearn import utils

from . import util
from . import svd

from sklearn.exceptions import DataConversionWarning
import warnings
warnings.filterwarnings('ignore',category=DataConversionWarning)

class PCA(base.BaseEstimator, base.TransformerMixin):

    """
    Args:
        rescale_with_mean (bool): Whether to substract each column's mean or not.
        rescale_with_std (bool): Whether to divide each column by it's standard deviation or not.
        n_components (int): The number of principal components to compute.
        n_iter (int): The number of iterations used for computing the SVD.
        copy (bool): Whether to perform the computations inplace or not.
        check_input (bool): Whether to check the consistency of the inputs or not.
    """

    def __init__(self, rescale_with_mean=True, rescale_with_std=True, n_components=2, n_iter=2,
                 copy=True, check_input=True, random_state=None, engine='auto'):
        self.n_components = n_components
        self.n_iter = n_iter
        self.rescale_with_mean = rescale_with_mean
        self.rescale_with_std = rescale_with_std
        self.copy = copy
        self.check_input = check_input
        self.random_state = random_state
        self.engine = engine

    def fit(self, X, y=None):
        # Check input
        if self.check_input:
            utils.check_array(X)


        # Convert pandas DataFrame to numpy array
        
        
        if isinstance(X, pd.DataFrame):
            self.columns=X.columns
            X = X.values

        # Copy data
        
        if self.copy:
            X = np.copy(X)
        # Scale data
        if self.rescale_with_mean or self.rescale_with_std:
            self.scaler_ = preprocessing.StandardScaler(
                copy=False,
                with_mean=self.rescale_with_mean,
                with_std=self.rescale_with_std
            ).fit(X)
            X = self.scaler_.transform(X)

        # Compute SVD
        self.U_, self.singular_values_, self.components_ = svd.compute_svd(
            X=X,
            n_components=self.n_components,
            n_iter=self.n_iter,
            random_state=self.random_state,
            engine=self.engine
        )
        
        # Compute total intertia
        
        self.total_inertia_ = np.einsum('ij,ji->',X,X.T)

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

    def _transform(self, X):
        """Returns the row principal coordinates.

        The row principal coordinates are obtained by projecting `X` on the right eigenvectors.
        """


        # Scale data
        if hasattr(self, 'scaler_'):
            X = self.scaler_.transform(X)

        return  np.dot(X,self.components_.T)

    def invert_transform(self,X):
        '''
        if whiten: X_raw= X* np.sqrt(explained_variance_) @ components
        else:  X_raw =X @ components
        '''

        return  np.dot(X,self.components_)		



    def fit_transform(self, X,y=None):
        self.fit(X)
        U=self.U_*self.singular_values_
        return U 
               
        # Convert numpy array to pandas DataFrame
    def column_correlation(self,X,same_input=True):
        """Returns the column correlations with each principal component."""
        #same_input: input for fit process is the same with that of X
        # self.components_  and outer array  X
        if   isinstance(X,pd.DataFrame):
            col_names=X.columns
            X=X.values        
        else:
            col_names=np.arange(X.shape[1])
        
        if   same_input: #X is fitted and the the data fitting and the data transforming is the same
            X_t=self.transform(X)
        else:
            X_t=self.fit_transform(X)

        return  pd.DataFrame({index_comp:{ 
                                        col_name: util._pearsonr(X_t[:,index_comp],X[:,index_col])
                                          for index_col,col_name in enumerate(col_names)  
                                                }
                                                for index_comp  in range(X_t.shape[1])})
                

    @property
    def explained_variance_(self):
        """Returns the eigenvalues associated with each principal component."""
        utils.validation.check_is_fitted(self, 'singular_values_')       
        return self.singular_values_ **2

    @property
    def explained_variance_ratio_ (self):
        """Returns the percentage of explained inertia per principal component."""
        utils.validation.check_is_fitted(self, 'singular_values_')
        return [eig / self.total_inertia_ for eig in self.explained_variance_]



    
    




  
