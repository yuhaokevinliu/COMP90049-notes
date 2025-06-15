# Generative Probablistic Modelling

## Generative Probabilistic Modeling

### Train/Test Data Generation
- Assume dataset is generated with some unknown distribution of $P(x,y) = P(x|y)P(y)$
- Our goal is to mimic the
data generation process
and find parameters that
best recreate the data

- two broad approach to classification problem
    - Discriminative 
    - Generative

### Discriminative 
- Model = a set of classifiers  
- Choose $f$ that **classifies** training examples well  
- Label new inputs $x$ based on:  
  $$
  y = f(x)
  $$
### Generative
- Model = a set of distributions $P(x, y; \theta)$ for any $\theta$  
- Choose $P(x, y; \hat{\theta})$ such that training examples are likely samples from this distribution  
- Label new inputs $x$ as $y$ where $y$ **maximizes** the joint likelihood:  
  $$
  y = \arg\max_y P(x, y; \hat{\theta})
  $$

- We can use simple **spherical Gaussian models** for the class-conditional distributions:

$$
P(x \mid y; \theta) = \mathcal{N}(x; \mu_y, \sigma_y^2 I)
$$

This expands to:

$$
P(x \mid y; \theta) =
\frac{1}{(2\pi \sigma_y^2)^{d/2}} \exp\left\{
-\frac{1}{2\sigma_y^2} \| x - \mu_y \|^2
\right\}
$$

> Note: both $x$ and $\mu_y$ are $d$-dimensional vectors.
![](assets/spherical_gaussian.png)

- We can also use **full Gaussian models** for the class-conditional distributions:

$$
P(x \mid y; \theta) = \mathcal{N}(x; \mu_y, \Sigma_y)
$$

Expanded form:

$$
P(x \mid y; \theta) = \frac{1}{(2\pi)^{d/2} |\Sigma_y|^{1/2}} 
\exp\left\{
-\frac{1}{2}(x - \mu_y)^T \Sigma_y^{-1} (x - \mu_y)
\right\}
$$

- The covariance matrix $\Sigma_y$ can be decomposed as:

$$
\Sigma_y = R
\begin{bmatrix}
\sigma_{y1}^2 & 0 \\
0 & \sigma_{y2}^2
\end{bmatrix}
R^T
$$

Where:

- $R$ is a **rotation matrix**  
- The diagonal elements $\sigma_{y1}^2, \sigma_{y2}^2$ represent **variances along the two principal axes**

- The contour plot on the right shows ellipses of constant probability for class $y = -1$
![](assets/full_gaussian.png)

### Maximum Likelihood Estimation MLE
- Recall [lec4 MLE](Probs.md#maximum-likelihood-estimate-mle)
- Our parameterized Gaussian model is:

$$
P(x, y; \theta) = P(x \mid y; \theta) \, P(y; \theta)
= \mathcal{N}(x; \mu_y, \sigma_y^2 I) \cdot q^{\delta(y, 1)} (1 - q)^{\delta(y, -1)}
$$

- We find parameters:

$$
\theta = (\mu_{+1}, \mu_{-1}, \sigma_{+1}^2, \sigma_{-1}^2, q)
$$

that **maximize the log-likelihood** of the training data:

$$
\ell(\mathcal{D}; \theta) = \sum_{i=1}^n \log P(x_i, y_i; \theta)
= \sum_{i=1}^n \left[
    \log P(x_i \mid y_i; \theta) + \log P(y_i; \theta)
\right]
$$
$$
\sum_{i=1}^{n} \left[
    -\frac{d}{2} \log(2\pi \sigma_{y_i}^2)
    - \frac{1}{2\sigma_{y_i}^2} \| x_i - \mu_{y_i} \|^2
\right]
+
\sum_{i=1}^{n} \left[
    \delta(y_i, 1) \log q + \delta(y_i, -1) \log(1 - q)
\right]
$$

### Classification and Decision Boundary
- Decision Boundary
    - Given $x$, predict the label ($+1$ or $-1$) with highest probability

    - Predict label $y = +1$ if:

    $$
    P(y = 1 \mid x; \hat{\theta}) > P(y = -1 \mid x; \hat{\theta})
    $$

    - By conditional probability (Bayes' rule):

    $$
    \frac{P(x, y = 1; \hat{\theta})}{P(x; \hat{\theta})}
    >
    \frac{P(x, y = -1; \hat{\theta})}{P(x; \hat{\theta})}
    $$

    - Which is equivalent to:

    $$
    P(x, y = 1; \hat{\theta}) > P(x, y = -1; \hat{\theta})
    $$

- The decision boundary is the set of x for which we do not know what label (+1 or -1) to predict
$$
P(x, y = 1; \hat{\theta}) = P(x, y = -1; \hat{\theta})
$$

![](assets/Decision_boundary.png)

### Probability Predictions
- Model also allow us to evaluate probabilities over the possible class labels 
$$
P(y = 1 \mid x; \hat{\theta}) = \frac{P(x, y = 1; \hat{\theta})}
{\sum_{y' \in \{-1, 1\}} P(x, y'; \hat{\theta})}
$$

The denominator is:

$$
P(x; \hat{\theta}) = \sum_{y' \in \{-1, 1\}} P(x, y'; \hat{\theta})
$$
![](assets/probalistic_predict.png)