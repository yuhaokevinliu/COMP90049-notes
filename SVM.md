# Support Vector Machine
- A maximum Margin classifier



### Concept
- assume $f(x;\theta) = sign(\theta,x)= sign(\sum(\theta_i,x_i)) \\=1 \space if \theta\cdot x > 0, =-1 \space \space otherwise$

- calculate with $\frac{y_i (\boldsymbol{\theta}^* \cdot \mathbf{x}_i)}{\lVert \boldsymbol{\theta}^* \rVert}$ This is the minimum distance from point to the separating boundary
- $\gamma_g = min_i \frac{y_i (\boldsymbol{\theta}^* \cdot \mathbf{x}_i)}{\lVert \boldsymbol{\theta}^* \rVert}$

### Optimisation with constraints
- e.g. Minimize $(z-1)^2$
- maintain $z>=3$

### maximum Margin classifier
- simulate test performance by evaluating Leave-One-Out Cross-Validation error
