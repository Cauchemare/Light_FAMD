
# Light_FAMD
REFERENCE TO PROJECT: [prince](https://github.com/MaxHalford/prince)

THE MAIN DIFFERENCES  AS SHOWN BELOW:
1. LIGHTER: REMOVE REPETITIVE CALCAULATION,ACCELERATING MAIN CALCULATION.
2. MORE STANDARD:  EACH ALGORITHM IMPLEMENTS  fit,transform,fit_transform METHODS FOLLOWING THE MAIN STRUCTURE of scikit-learn API

`Light_FAMD` is a library for prcessing [factor analysis of mixed data](https://www.wikiwand.com/en/Factor_analysis). This includes a variety of methods including [principal component analysis (PCA)](https://www.wikiwand.com/en/Principal_component_analysis) and [multiply correspondence analysis (MCA)](https://www.researchgate.net/publication/239542271_Multiple_Correspondence_Analysis). The goal is to provide an efficient and light implementation for each algorithm along with a scikit-learn API.

## Table of contents

- [Usage](##Usage)
  - [Guidelines](###Guidelines)
  - [Principal component analysis (PCA)](#principal-component-analysis-pca)
  - [Correspondence analysis (CA)](#correspondence-analysis-ca)
  - [Multiple correspondence analysis (MCA)](#multiple-correspondence-analysis-mca)
  - [Multiple factor analysis (MFA)](#multiple-factor-analysis-mfa)
  - [Factor analysis of mixed data (FAMD)](#factor-analysis-of-mixed-data-famd)

`Light_FAMD` doesn't have any extra dependencies apart from the usual suspects (`sklearn`, `pandas`, `numpy`) which are included with Anaconda.

## Usage

```python
import numpy as np; np.random.set_state(42)  # This is for doctests reproducibility
```

### Guidelines

Each base estimator(CA,PCA) provided by `Light_FAMD` extends scikit-learn's `(TransformerMixin,BaseEstimator)`.which means we could use directly `fit_transform`,and `(set_params,get_params)` methods.
 
Under the hood `Light_FAMD` uses a [randomised version of SVD](https://scikit-learn.org/dev/modules/generated/sklearn.utils.extmath.randomized_svd.html). This algorithm finds a (usually very good) approximate truncated singular value decomposition using randomization to speed up the computations. It is particularly fast on large matrices on which you wish to extract only a small number of components. In order to obtain further speed up, n_iter can be set <=2 (at the cost of loss of precision). However if you want reproducible results then you should set the `random_state` parameter.

In this package,inheritance relationship as shown  below(A->B:A is superclass of B):

- PCA -> MFA -> FAMD
- CA ->MCA

You are supposed to use each method depending on your situation:

- All your variables are numeric: use principal component analysis (`PCA`)
- You have a contingency table: use correspondence analysis (`CA`)
- You have more than 2 variables and they are all categorical: use multiple correspondence analysis (`MCA`)
- You have groups of categorical **or** numerical variables: use multiple factor analysis (`MFA`)
- You have both categorical and numerical variables: use factor analysis of mixed data (`FAMD`)

Notice that `Light_FAMD` does't support the sparse input,see [Truncated_FAMD](https://github.com/Cauchemare/Truncated_FAMD) for an alternative of sparse and big data.


###	Principal-Component-Analysis: PCA

**PCA**(rescale_with_mean=True, rescale_with_std=True, n_components=2, n_iter=3,
                 copy=True, check_input=True, random_state=None, engine='auto'):
	
**Args:**
- `rescale_with_mean` (bool): Whether to substract each column's mean or not.
- `rescale_with_std` (bool): Whether to divide each column by it's standard deviation or not.
- `n_components` (int): The number of principal components to compute.
- `n_iter` (int): The number of iterations used for computing the SVD.
- `copy` (bool): Whether to perform the computations inplace or not.
- `check_input` (bool): Whether to check the consistency of the inputs or not.
- `engine`(string):"auto":randomized_svd,"fbpca":Facebook's randomized SVD implementation
- `random_state`(int, RandomState instance or None, optional (default=None):The seed of the -pseudo random number generator to use when shuffling the data. If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random.
Return ndarray (M,k),M:Number of samples,K:Number of components.

**Fitted Estimator**
**Attributes:**
- `components_`(array), shape (n_components, n_features)
Principal axes in feature space, representing the directions of maximum variance in the data. The components are sorted by explained_variance_.
- `explained_variance_`(array), shape (n_components,):The amount of variance explained by each of the selected components.
- `explained_variance_ratio_`(array), shape (n_components,):Percentage of variance explained by each of the selected components.
- `singular_values_`(array),shape (n_components,):The singular values corresponding to each of the selected components. The singular values are equal to the 2-norms of the n_components variables in the lower-dimensional space.


**Examples:**
```
>>>import numpy as np
>>>from Light_Famd import PCA
>>>X = pd.DataFrame(np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]]),columns=list('ABC'))
>>>pca = PCA(n_components=2)
>>>pca.fit(X)
PCA(check_input=True, copy=True, engine='auto', n_components=2, n_iter=3,
  random_state=None, rescale_with_mean=True, rescale_with_std=True)

>>>print(pca.explained_variance_)
[11.89188304  0.10811696]

>>>print(pca.explained_variance_ratio_)
[0.9909902530309821, 0.00900974696901714]
>>>print(pca.column_correlation(X))  #You could call this method once estimator is >fitted.correlation_ratio is pearson correlation between 2 columns values,
where p-value >=0.05 this similarity is `Nan`.
          0   1
A -0.995485 NaN
B -0.995485 NaN

>>>print(pca.transform(X))
[[ 0.82732684 -0.17267316]
 [ 1.15465367  0.15465367]
 [ 1.98198051 -0.01801949]
 [-0.82732684  0.17267316]
 [-1.15465367 -0.15465367]
 [-1.98198051  0.01801949]]
>>>print(pca.fit_transform(X))
>[[ 0.82732684 -0.17267316]
 [ 1.15465367  0.15465367]
 [ 1.98198051 -0.01801949]
 [-0.82732684  0.17267316]
 [-1.15465367 -0.15465367]
 [-1.98198051  0.01801949]]

```
###	Correspondence-Analysis: CA

**CA**(n_components=2, n_iter=10, copy=True, check_input=True, random_state=None,
                 engine='auto'):
	
**Args:**
- `n_components` (int): The number of principal components to compute.
- `copy` (bool): Whether to perform the computations inplace or not.
- `check_input` (bool): Whether to check the consistency of the inputs or not.
- `engine`(string):"auto":randomized_svd,"fbpca":Facebook's randomized SVD implementation
- `random_state`(int, RandomState instance or None, optional (default=None):The seed of the -pseudo random number generator to use when shuffling the data. If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random.

Return ndarray (M,k),M:Number of samples,K:Number of components.

**Examples:**
```
>>>import numpy as np
>>>from Light_Famd import CA
>>>X  = pd.DataFrame(data=np.random.randint(0,100,size=(10,4)),columns=list('ABCD'))
>>>ca=CA(n_components=2,n_iter=2)
>>>ca.fit(X)
CA(check_input=True, copy=True, engine='auto', n_components=2, n_iter=2,
  random_state=None)

>>> print(ca.explained_variance_)
[0.09359686 0.04793262]

>>>print(ca.explained_variance_ratio_)
[0.5859714238674507, 0.3000864001658787]

>>>print(ca.transform(X))
[[-0.18713811  0.09830335]
 [ 0.34735892  0.34924107]
 [ 0.33511949 -0.29842395]
 [-0.26200927 -0.14201485]
 [-0.21803569  0.0977655 ]
 [-0.25482535 -0.16019826]
 [ 0.09899818 -0.15015664]
 [-0.24835074  0.54054788]
 [-0.21056433 -0.29941039]
 [ 0.33904416  0.04835469]]


```

###	Multiple-Correspondence-Analysis: MCA
MCA class inherits from  CA  class.

```
>>>import pandas as pd
>>>X=pd.DataFrame(np.random.choice(list('abcde'),size=(10,4),replace=True),columns =list('ABCD'))
>>>print(X)
   A  B  C  D
0  e  a  a  b
1  b  e  c  a
2  e  b  a  c
3  e  e  b  c
4  b  c  d  d
5  c  d  a  c
6  a  c  e  a
7  d  b  d  b
8  e  a  e  e
9  c  a  e  b
>>>mca=MCA(n_components=2)
>>>mca.fit(X)
MCA(check_input=True, copy=True, engine='auto', n_components=2, n_iter=10,
  random_state=None)

>>>print(mca.explained_variance_)
[0.8286237  0.67218257]

>>>print(mca.explained_variance_ratio_)
[0.2071559239010482, 0.16804564240579373]

>>>print(mca.transform(X)) 
[[-0.75608657  0.17650888]
 [ 1.39846026 -1.17201511]
 [-0.77421024 -0.04847214]
 [-0.32829309 -1.19959921]
 [ 1.49371661  0.90485916]
 [-1.00518879 -0.41815679]
 [ 1.11265365 -0.14764943]
 [-0.07786514  1.66121318]
 [-0.51081888 -0.06676941]
 [-0.55236782  0.31008086]]

```
###	Multiple-Factor-Analysis: MFA
MFA class inherits from  PCA  class.
Since FAMD class inherits from  MFA and the only thing to do for FAMD is to determine `groups` parameter compare to its  superclass `MFA`.therefore we skip this chapiter and go directly to `FAMD`.


###	Factor-Analysis-of-Mixed-Data: FAMD
The `FAMD` inherits from the `MFA` class, which entails that you have access to all it's methods and properties of `MFA` class.
```
>>>import pandas as pd
>>>X_n = pd.DataFrame(data=np.random.randint(0,100,size=(10,2)),columns=list('AB'))
>>>X_c =pd.DataFrame(np.random.choice(list('abcde'),size=(10,4),replace=True),columns =list('CDEF'))
>>>X=pd.concat([X_n,X_c],axis=1)
>>>print(X)
    A   B  C  D  E  F
0  11  67  a  a  d  e
1  43  67  d  d  d  a
2  40   3  d  b  c  b
3  81  66  e  b  c  c
4  36  50  e  a  c  e
5  95  69  b  d  e  a
6  57  71  d  c  d  c
7  29  58  e  e  d  d
8  67  27  b  e  d  e
9  78  20  e  d  a  a

>>>famd = Light_FAMD.FAMD(n_components=2)
>>>famd.fit(X)
FAMD(check_input=True, copy=True, engine='auto', n_components=2, n_iter=3,
   random_state=None)

>>>print(famd.explained_variance_)
[15.41428212  9.53118994]

>>>print(famd.explained_variance_ratio_)
[0.27600556629884937, 0.17066389830189396]

>>> print(famd.column_correlation(X))
            0         1
A         NaN       NaN
B         NaN       NaN
C_a       NaN       NaN
C_b       NaN       NaN
C_d       NaN       NaN
C_e       NaN       NaN
D_a       NaN       NaN
D_b       NaN       NaN
D_c       NaN       NaN
D_d       NaN  0.947742
D_e       NaN       NaN
E_a       NaN       NaN
E_c       NaN       NaN
E_d  0.759576       NaN
E_e       NaN       NaN
F_a       NaN  0.947742
F_b       NaN       NaN
F_c       NaN       NaN
F_d       NaN       NaN
F_e       NaN       NaN



>>>print(famd.transform(X)) 
[[ 4.15746579 -2.87023941]
 [ 4.95755717  3.74813131]
 [ 2.6358626  -1.87761681]
 [ 3.4203849  -2.2485009 ]
 [ 4.10436826 -3.57317268]
 [ 2.88436338  5.65046057]
 [ 3.92172253 -0.41161253]
 [ 4.48442501 -1.30359035]
 [ 4.42018651 -0.77402381]
 [ 3.66615694  4.15701604]]

print(famd.fit_transform(X))
[[ 4.15746579 -2.87023941]
 [ 4.95755717  3.74813131]
 [ 2.6358626  -1.87761681]
 [ 3.4203849  -2.2485009 ]
 [ 4.10436826 -3.57317268]
 [ 2.88436338  5.65046057]
 [ 3.92172253 -0.41161253]
 [ 4.48442501 -1.30359035]
 [ 4.42018651 -0.77402381]
 [ 3.66615694  4.15701604]]

```


```python
>>> import Light_FAMD
>>> pca = Light_FAMD.PCA(engine='fbpca')

```

