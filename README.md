# Function for estimating VIFs for contrasts

This tool calculates VIFs for contrasts of parameter estimates in fMRI time series design matrices. The VIF here extends the standard VIF, comparing the variance of the contrast estimate in the specified design matrix to an ideal design with uncorrelated regressors—without manual orthogonalization, which would lose critical information for accurate VIF estimates.

$$VIF(c\hat\beta) = \frac{c(X'X\circ I_N)^{-1}c'}{c(X'X)^{-1}c'},$$

where $\circ$ represents elementwise matrix multiplication and $I_N$ is a $N\times N$ identity matrix.  This element-wise multiplication sets the correlation of the regressors to 0.

### Important requirement
The estimates will only be accurate if each regressor represents a single stimulus and no orthogonalization has been done. Parametric regressors are fine if they represent a continuous measure (e.g., trial difficulty). However, a parametrically modulated regressor coded as Stimulus A - Stimulus B will miscalculate the VIF and should be recoded with separate Stimulus A and Stimulus B regressors.




# Example use
To add a copy icon to the code block, you can use HTML within your markdown. Here's how you can do it:

```python
import pandas as pd
import numpy as np

from vif_contrasts import est_contrast_vifs # note, Nilearn is required

# Create design matrix with correlated regressors
mean = [0, 0, 0]
cov = [[1, 0.9, 0.7], [0.9, 1, 0.7], [0.7, 0.7, 1]]  
data = np.random.multivariate_normal(mean, cov, 100)
desmat = pd.DataFrame(data, columns=['x1', 'x2', 'x3'])
desmat['constant'] = 1

# Define contrasts
contrasts = {
    'x1 vs baseline': 'x1',
    'x2 vs baseline': 'x2',
    'x1 vs x2': 'x1 - x2',
}

vifs = est_contrast_vifs(desmat, contrasts)

for contrast_name, vif in vifs.items():
    print(f'{contrast_name} has VIF={vif}')
```