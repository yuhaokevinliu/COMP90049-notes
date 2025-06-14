# K- Nearest Neighbors

- K nearest neighbor = k closest stored data points
- K = 1 -> only take the very close neighbor

### Training
- store all training examples

### Testing
- Compute distance of test instance to all training data points
- Find the K neighbors
- Compute **target concept** of the test instance based on labels of training instances

- KNN Classification
    - Return the most common class label

- KNN Regression
    - Return the average value 

### Key questions
- How to represent each data point -> Feature vectors
- How to measure the distance  
- What if neighbors disagree
- How to select K

### How to represent each data point
- Feature vectors

### How to measure the distance
- convert nominal into numeric if need (one-hot) 
    -> Hamming Distance = difference in objects
    -> Simple Matching Distance = $d = 1-k/m$ (d:distance, k: matching, m: total)
    -> Jaccard Distance = Intersection over union d = $1 - (A \cap B/A \cup B)$
    -> Manhattan Distance = d = $\sum |Ai - Bi|$
    -> Euclidean Distance = d = $\sqrt(\sum(ai-bi)^2)$
    -> Cosine Distance = d = $1 - cos(a,b) = a \cdot b/|a||b| $

- Comparing Ordinal Feature Vectors
    - sort values and return a rank r
    - map ranks to evenly spaced value between 0 and 1
    - compute distance function for numeric features

### What if neighbors disagree
- Majority Voting
    Head count

- Inverse Distance
    $Wj = 1/Dj + e$ where $e = 1e-10$

- Inverse Linear Distance
    $Wj = (Dk-Dj)/(Dk-D1)$ where Dk is max and D1 is min

### How to select K

- Small K
    - Jagged decision boundary
    - we capture noise
    - lower classifier performance

- Large K
    - smooth decision boundary
    - danger of grouping together unrelated classes
    - also: lower classifier performance
    - what if K == N

![Draw Validation error:](assets\draw_validation_error.png)
Take the maximum point in the graph

### Breaking Tie
- Tied Distance
    - Arbitrarily choose one (e.g., lexicographically).

    - Choose the majority class among tied neighbors (if k > 1).

    - Add a random tie-breaking step (non-deterministic).

- Tied vote
    - Use odd values of k (e.g., 3, 5) to avoid ties in most cases.

    - Use weighted voting: closer neighbors get more vote weight.

    - Apply a deterministic tie-breaker, like prioritizing a specific class.



### Why KNN
- Pros
    - Intuitive and simple
    - No assumptions
    - Supports classification and regressino
    - No training: new data join > evolve and adapt immediately

- Cons
    - How to decide best distance
    - How to combine multiple neighbors
    - How to select K
    - Expensive with large dataset

--- 
## Future 
### Lazy Learning
- AKA Instance based learning
- Store training data
- fixed distance function
- fixed prediction rule
- compare test instances with stored instances
- no learning
![Lazy Learn](assets\lazy_learn.png)

### Eager Learning
- train a model with labelled data training instances
- generalize from seen data to unseen data
- predict labels for test instances

![Eager Learn](assets\eager_learn.png)