# Naive Bayes
- A supervised learning algorithm
- Recall probability and product rule
- $P(x, y) = P(y \mid x) P(x) = P(x \mid y) P(y)$
- $P(y \mid x) = \frac{P(x \mid y) P(y)}{P(x)}$

- To make it more feasible, we come up with 
$P(x_1, x_2, \dots, x_M \mid y) P(y) 
\approx P(x_1 \mid y) P(x_2 \mid y) \cdots P(x_M \mid y) P(y) \\
= P(y) \prod_{m=1}^{M} P(x_m \mid y)$

### Conditional Independence Assumption
- Where each features are assumed to be independent
- Intuitively: if know the class of email is spam, then none of the words depend on their surronding words

### Complete Naive Bayes Classifier
- $\hat{y} = \arg\max_{y \in \mathcal{Y}} P(y) P(x_1, x_2, x_3, x_4, \dots, x_n \mid y) \\
= \arg\max_{y \in \mathcal{Y}} P(y) \prod_{m=1}^{M} P(x_m \mid y)$

- features are conditionally independent given the class
- Instances are independent on each other
- The distributino in test data is same as in training data


### Smooth Categorical features
- The problem with unseen features
- If any $P(Xm|y) = 0$, then $P(y|x) = 0$
- But it is possible in real life, not all combinations stored in train dataset
- A single zero renders many additional meaningful observations irrelevant
- Solution
    - No event is impossible
    - add a small and positive value to every unseen class-feature combination
    - readjust the remaining model parameters to maintain valid probabilty distribution

### Epsilon smoothing
- if a 0 is observed, replace with a small $\epsilon$
- $\epsilon < 1/N $
- Effectively it reduces most comparisons to the cordinality of $\epsilon$ 

### Laplace Smoothing
- add a pseudocount $\alpha$ to each feature count observed during traning
- $\alpha$ is a parameter, usually = 1
- $P(x_m = j \mid y = k) = 
\frac{\alpha + \text{count}(y = k, x_m = j)}
{M\alpha + \text{count}(y = k)}$ where M is the number of values m can take
- Probability change drastically when dataset is small, minor otherwise
- Reduce Variance because reduce sensitivity to individual observations in dataset
- adds bias to the NB classifier, no longer have a true maximum likelihood estimator

### Maximum likelihood calculation
- Categorical Naive Bayes
    - Parameter $\phi$ of the Categorical distribution over class labels are the relative frequencies of classes observed in training data
    $\space$ $\phi_y = \frac{\text{count}(y)}{N}$
    - Parameter $\psi$ of the Categorical distribution over features given a class label are the observed relative frequencies of (class,label) among all instances with that class
    $\psi_{y,m} = \frac{\text{count}(y, m)}{\text{count}(y)}$

    - These parameters maximize the probability of the observed dataset. They are the maximum likelihood estimator of $\phi and \psi$


- Gaussian Naive Bayes


    - For each class y and each feature $x_m$ , we learn an individual Gaussian
    distribution parameterized by a mean $μy,m$ and a standard deviation $σy,m$

    - Mean:  
    The average of all observed feature values for \( x_m \) under class \( y \):

    $\mu_{y,m} = \frac{1}{\text{count}(y)} \sum_{i : y_i = y} x_m^i $

    - Standard deviation:  
    Sum of squared differences of observed values from the mean, normalized, and square rooted:

    $\sigma_{y,m} = \sqrt{ \frac{ \sum_{i : y_i = y} (x_m^i - \mu_{y,m})^2 }{ \text{count}(y) } }$

### Calculate Priors
- One prior $P(Y=k)$
- Normalize the count for $y_i$ by the total number of training instance N by:
    - divide each entry by the sum of the entries in the list
    - keep a separate counter for the total number of instance N, which is often useful

### Calculate likelihood parameters
- One likelihood $P(x=j|y=k)$ per attribute per class, for each X:
    - Each likelihood is a Gaussian distribution parameterized by a mean and standard deviation

### Making predictions using a NB Classifier


$\hat{y} = \arg\max_{k \in \mathcal{Y}} P(y = k) \prod_m P(x_m = j \mid y = k; \mu_{k,m}, \sigma_{k,m})$

- $( P(y = k) )$ can be read off the data structures from the training phase.
- $( P(x_m = j \mid y = k; \mu_{k,m}, \sigma_{k,m}) )$ can be computed using the likelihood function of the Gaussian distribution:


- $\frac{1}{\sqrt{2\pi \sigma_{m,k}^2}} \exp\left( -\frac{1}{2} \frac{(x_m - \mu_{m,k})^2}{\sigma_{m,k}^2} \right)$

- We only care about the class corresponding to the maximal value, so as we progress through the classes, we can keep track of the greatest value so far.

- We’re multiplying a bunch of numbers $(0, 1]$ together — because of our
floating-point number representation, we tend to get underflow.
One common solution is a log-transformation:  
$\hat{y} = \arg\max_{k \in \mathcal{Y}} P(y = k) \prod_m P(x_m = j \mid y = k)$ \\ $= \arg\max_{k \in \mathcal{Y}} \left[ \log P(y = k) + \sum_m \log P(x_m = j \mid y = k) \right]$

### Sommthing continuous features
- what if a class of 0 variance, all observed values are the same

1. ignore the feature
    - might loss information if 0 variance only for some classes
    - safe to do if the features has the same value across all classes
2. add small smoothing value to the PDF  
    - $p(x = j|μ, σ) → p(x = j|μ, σ+𝜀)$
    - set $\epsilon$ as a small fraction of the largest observed variance to all variance

### Final thoughts 
- we don't need the true distribution, just need to identify the most likely outcome
- Advantage
    - easy to build
    - easy to scale to many dimensions
    - reasonably easy to explain why a specific class was predicted
    - good starting point for a classification project

### Summary Questions
- what is NB algorithm
- what is Bayes's rule and how it relate to Naive Bates
- what are the simplifying assumptions
- how and why do we use smoothing in NB
- how can we implement a NB classifier
