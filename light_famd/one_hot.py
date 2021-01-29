"""This module contains a custom one-hot encoder. It inherits from sklearn's
OneHotEncoder and returns a pandas.SparseDataFrame with appropriate column
names and index values.
"""


import numpy as np
import pandas as pd
from sklearn import preprocessing


class OneHotEncoder(preprocessing.OneHotEncoder):

    def __init__(self):
        super().__init__(sparse=False, dtype=np.uint8,handle_unknown='ignore')

    def fit(self, X, y=None):

        self = super().fit(X)
        self.column_names_ = self.get_feature_names(X.columns if hasattr( X,'columns') else None)

        return self

    def transform(self, X):
        print(self.column_names_)
        return pd.DataFrame(
            columns=self.column_names_,
            data=super().transform(X)
        )
