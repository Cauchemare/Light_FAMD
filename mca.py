"""Multiple Correspondence Analysis (MCA)"""

import numpy as np
from sklearn import utils

from . import ca
from . import one_hot



class MCA(ca.CA):

    def fit(self, X, y=None):

        if self.check_input:
            utils.check_array(X, dtype=[str, np.number])
            
        n_initial_columns = X.shape[1]

        # One-hot encode the data
        self.one_hot_ = one_hot.OneHotEncoder().fit(X)
        
        n_new_columns = len(self.one_hot_.column_names_)
        self.total_inertia_ = (n_new_columns - n_initial_columns) / n_initial_columns
        # Apply CA to the indicator matrix
        super().fit(self.one_hot_.transform(X))

        # Compute the total inertia


        return self

    def _transform(self, X):
        return super()._transform(self.one_hot_.transform(X))



    def transform(self, X):
        """Computes the row principal coordinates of a dataset."""
        utils.validation.check_is_fitted(self, 's_')
        if self.check_input:
            utils.check_array(X, dtype=[str, np.number])
        return self._transform(X)

    
