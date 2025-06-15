# Dimension Reduction

### Motivation
- when given dataset of massive sample and instance
- It is hard to visualise
- Hard to find correlations
![](assets/dimensions.png)

### Goal
- Identify and describe the “dimensions” that underlie the data
    - May be more fundamental than those directly measured but hidden to the
user
- Reduce dimensionality of modeling problem
- Benefit is simplification, it reduces the number of variables you have to
deal with in modeling
- Can identify set of variables with similar behavior

## Methods
- Principal component analysis (PCA)
    - Linear transformation, find orthogonal directions of maximum variance
- Factor analysis
    - Linear combination of small number of latent variables
- Multidimensional scaling (MDS)
    - Project into low-dimensional subspace while preserving distance between
    points (can be non-linear)

## Linear Algebra Revisit

### Eigen Decomposition

- **Symmetric matrix**:
  $$
  \Sigma = U \Lambda U^T
  $$

- **"Orthonormal" matrix**:
  $$
  U^T U = I
  $$

- **Diagonal matrix** $\Lambda$ of eigenvalues

- **Eigenvectors**: columns of \( U \)

### Singular value decomposition (SVD)
- **Matrix**: $X = VSU^T$
- **"Orthonormal" matrices**: $V^T V = I$, $U^T U = I$
- **Diagonal matrix** $S$ contains the **singular values**

### Covariance
- Measures how variables A and B vary together  
    - given a set of $a_n \space and \space b_n$ values

  $$
  \text{Mean}(a) = \frac{1}{n} \sum_{i=1}^{n} a_i
  $$

    - **Covariance** between vectors $a$ and $b$:
  
  $$
  \text{Cov}(a, b) = \frac{1}{n} \sum_{i=1}^{n} (a_i - \text{Mean}(a))(b_i - \text{Mean}(b))
  $$

- Measures linear relationship
![](assets/covariance.png)

### Covariance Matrix
