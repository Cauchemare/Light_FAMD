import pandas as pd
from scipy.stats import pearsonr
import numpy as np



def _pearsonr(A,B,threshold=0.05):
    co_,p_value=pearsonr(A,B)
    return co_ if p_value <=threshold else  np.nan
    
    
    
def  df_correlation(DF_C,DF_INDEX):
    if not isinstance(DF_C,(pd.DataFrame,pd.SparseDataFrame)):
        DF_C=pd.DataFrame(DF_C)
    if not isinstance(DF_INDEX,(pd.DataFrame,pd.SparseDataFrame)):
        DF_INDEX=pd.DataFrame(DF_INDEX)		
    _data={}
    for col in DF_C:
        if isinstance(DF_C[col],pd.SparseSeries) :
            _c= DF_C[col].values.to_dense()
        else:
            _c= DF_C[col].values
        _embed_dict={}
        for index in  DF_INDEX:
            if isinstance(DF_INDEX[index],pd.SparseSeries) :
                _i= DF_INDEX[index].values.to_dense()
            else:
                _i= DF_INDEX[index].values
            _embed_dict[index]=_pearsonr(_c,_i)
        _data[col]=_embed_dict
            
    return pd.DataFrame(_data )    
