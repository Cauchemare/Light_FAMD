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
        
        _X_t=  self.one_hot_.transform(X) 
        
        _0_freq_serie= (_X_t == 0).sum(axis=0)/ len(_X_t)
        
        self._usecols=_0_freq_serie[_0_freq_serie < 0.99].index
        print('MCA PROCESS ELIMINATED {0}  COLUMNS SINCE THEIR MISS_RATES >= 99%'.format( _X_t.shape[1] - len(self._usecols) ))
        
        n_new_columns = len(self._usecols)
        self.total_inertia_ = (n_new_columns - n_initial_columns) / n_initial_columns
        # Apply CA to the indicator matrix
        super().fit(_X_t.loc[:,self._usecols])

        return self

    def _transform(self, X):
        return super()._transform(self.one_hot_.transform(X).loc[:,self._usecols])



    def transform(self, X):
        """Computes the row principal coordinates of a dataset."""
        utils.validation.check_is_fitted(self, 'singular_values_')
        if self.check_input:
            utils.check_array(X, dtype=[str, np.number])
        return self._transform(X)

