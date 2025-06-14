### Decision Tree
- With features calculate from One-R, we can have different branches.
- Create Optimal Desicion Tree

### ID3 with Inforamation Gain (Entropy)
- A low-prob happens -> High entropy
- A high-prob happens -> low entropy

- Self Information $\textit{Selfinfo}(x) = \frac{1}{p(x)} \approx -\log_2 p(x)$

- $H(x) = -\sum_{i=1}^{n} P(i) \log_2 P(i)$ where $0 \times \log_2 0 \overset{\text{def}}{=} 0$

- Conditioned Entropy of X given Y $H(X \mid Y) = \sum_{y \in \mathcal{Y}} p(y) \cdot H(X \mid Y = y)$

### Information Gain
- Information gain measures the reduction in entropy about the target variable achieved by partitioning the data based on a given feature.
- How important a feature is to the label
- $IG(X, Y) = H(X) - H(X \mid Y)$ $= H(X) - \sum_{y \in \mathcal{Y}} p(y) \cdot H(X \mid Y = y)$
- Calculate for all features and pick the biggest one

### Shortcoming of Information Gain
- A subset of information is more likely to be homogenous if there are only few instances
- Attributes with many values will have fewer instances at each child node 
- May lead to **overfitting**/fragmentation

### Solution
- Gain Ratio (GR)
    - reduces the bias for information gain toward highly branching attributes by normalising relative to the split information 

- Split info (SI)
    - is the entropy of a given split (evenness of the distribution of instances to attribute values)
- Discourages the selection of attributes with many uniformly distributed values, which have high entropy

### Stopping Criteria 
- IG and GR may help choose the best attribute at a given node
- IG = 0 means no improvement, a very small improvment is often unjustifiable
- Stop when $purity(Root) > \gamma$
- Choose the best attribute when IG/GR > threshold $\theta$
- pruning, a post-process the tree to remove branches with few instance or low IG/GR

### Decision Tree with numeric values
- A tree with numeric features is typically a binary tree
- The algorithm finds a threshold $t$ that best split the data to maximize some purity metric (IG/GR)
- left child: contains data points where feature value is less than $t$
- right child: contains data points where feature value is greater than $t$

### Regression Tree
- For predicting continuous values
- Uses metrics like the sum of squared residuals (SSR) to select the most suitable attribute to use as the node within each subset
- Outputs a numeric value, which is the mean of the target variable in the leaf node
- a regression tree is typically a binary tree

### Why use DT
- Pros
    - highly regarded among basic learners
    - fast to train, even faster to classify
    - very transparent 
- Cons
    - prone to overfitting
    - loss of information for coutinuous variables
    - complex calculation if there are many classes
    - no guarantee to return the globally optimal decision
    - Information gain: Bias for attribute with greater no. of values

### Other
- [Random Forest](Random_Forest.md)
