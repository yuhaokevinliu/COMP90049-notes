# Probability revisit

### Very Basic
P(A=a) -> $0 <= P  <= 1 $

### Some formulas
- $P(A,B) = P(A \cap B)$

- $P(A|B) = P(A\cap B)/P(B)$

- $P(A,B) = P(A)*P(B)$ if A,B are independent

### Notation
- $P(A=x)$ indicates the likelyhood of A = x, also write as $P(x)$
- $P(A)$ shows the probability distribution, a histogram
![P(A)](assets\P_A.png)

### Rules
- independence
    - if $P(A,B) = P(A)*P(B)$
- disjoint events
    - $P(A \cap B) = \emptyset$ and $P(A or B) = P(A) + P(B)$
- product rule
    - $P(A \cap B) = P(A|B)P(B) = P(B|A)P(A)$
- chain rule
    - $P(A1\cap ... \cap An) = P(A1)P(A2|A1)P(A3|A2 \cap A1)...P(An|\cap ..Ai)$

- Bayes
    - $P(A|B) =P(A)P(B|A)/P(B)$ derived from $P(A|B) = P(A \cap B)/P(B)$ and $P(A \cap B) = P(A|B)P(B) = P(B|A)P(A)$

    - Bayes rule allow us to compute P(A|B) with known of P(B|A)
![Bayes](assets\Bayes.png)

### Marginalization
- $P(A) = \sum_{b \in \mathbb{B}} P(A,B=b)$
- $P(A) = \sum_{b \in \mathbb{B}} P(A|B=b)P(B=b)$ 
- $P(A|C) = \sum_{b \in \mathbb{B}} P(A|C,B=b)P(B=b|C)$

### Probability Distributions
- Random Variable
    - Discrete Random Variable -> a countable number of distinct values
    - Continuous Random Variable -> infinite

### Gaussian Nromal Distribution
- two parameter
    - mean $\mu$
    - standard deviation $\sigma$
    - $P(A = x \mid \mu, \sigma) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{1}{2} \left( \frac{x - \mu}{\sigma} \right)^2}$

### Bernoulli Distribution
- a single trial with binary outcome

### Binomial Distribution
- a series of independent trials with only two outcomes
- $P(m, n, p) = \binom{n}{m} p^m (1 - p)^{n - m}$

### Multinomial Distribution
- The outcome of a series of trials where each trial can result in one of more than two possible outcomes
- $P(X_1 = x_1, X_2 = x_2, \dots, X_n = x_n; \mathbf{p}) =
\frac{(\sum_i x_i)!}{x_1! \cdots x_n!} \, p_1^{x_1} p_2^{x_2} \cdots p_n^{x_n}
= \frac{(\sum_i x_i)!}{x_1! \cdots x_n!} \prod_i p_i^{x_i}$

### Categorical Distribution
- events result from a single trial with more than two possible outcomes
- e.g rolling a fair-dice face **once** and the probability observing a five

### Relation with Machine Learning
- Uncertainty
- Model features as following certain **probability distributions**
- Soft predictions

### Probabilistic Models
- allow us to reason about random events in a principled way.
- allow to formalise hypotheses as different types of probability distribution

### Maximum Likelihood Estimate MLE
- $\hat{\theta} = \arg\max_{\theta} \, p(X = x; \theta, N)$
- Use known X to pick $\theta$
- With picked $\theta$, we can predict unseen X

### Maximum Posteriori Estimate
- $\hat{\theta} = \arg\max_{\theta} \, P(\theta) P(x \mid \theta)$
