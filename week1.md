# Machine Learning Concepts

### concept discriptions
- each example of data is called **Instance**
- each Instance have **Feature** or **Attributes**
- also have **Concepts**
- **Label** or **classes**

### Concept we aim to learn
- **Classification** -> (Y/N)
- **Regression** -> a specific value
- **Clustering** -> group similar data
- **Association Learning** -> some features may appear together

### Unsupervised learning
- Do not have access to an inventory of classes.
- Use un-labelled dataset to learn
- aim to discover hidden patterns, group or relationship in the data without target label

### advantage of unsupervised learning
#### Grouping of instances
- customer segmentation
- anomaly detection

#### Other methods
- Dimensionality reduction -> reduce number of features
- Market Basket Analysis -> products purchased together

### Clustering
- finding groups of items that are similar
- unsupervised
- Class is unknown or not passing to algorithm
- Success is often measured subjectively; evaluation is problematic

### Supervised
- Have prior knowledge of classes and set out to discover and categorise new instance
- Use labelled dataset to learn
- To predict a **label** or **numeric value**

### Classification
- Assign an instance a discrete class label
- Supervised
- providewith actual outcome or class
- provided with a set of classified training data
- Measure success on held-out data for which class labels are known

### Regression 
- The class is continuous
- numeric prediction
- can have infinitely labels for an instance
- correct when the numeric value is close to the true value

### Feature vector
- Assume a feature of fruit are Colour, Shape, Sweetness
- Apple = [red, round, medium], Orange = [orange, round, low]

### Feature Data Types
- Discrete
- Continuous

### Nominal (Categorical) Quantities
- Values are distinct
- == categorical or discrete
- boolean is a special case
- Can only perform **equality** test

### Ordinal Quantities
- explicit order
- no distance between the values, indicate some order but does not quantify the precise distance
- No addition or subtraction
- the distinction between Ordinal and Nominal is not always clear

### Numeric Quantities
- **real-value** attribute
- Scalar -> attribute distance
- All mathematical operations are allowed

### Algorithm <---> Attribute type
- Naive Bayes -> nominal or numeric
- Logistic/Linear Regression -> numeric
- Perceptron/Neural Networks -> numeric

- When meeting wrong attribute type, we can :
    - Select only attribute with correct type
    - Change the model assumptions to match the data
    - **Change attributes to match the model**

### Functions to convert
#### Nominal to Numeric
- Encoding
    - Pros
    - simplicity
    - space efficiency
    
    - Cons
    - Arbitrary Ordinal Relationships
    - Meaningless distance
- One-hot encoding
    - Problem
    - Increase dimensionality of feature space
    - Increase data sparsity -> tons of zeros
#### Numeric to Nominal
- Discretisation
    - group numeric values into a pre-defined set of distinct categories
    - -> map housing price to {high, medium, low}
    - Decide number of categories -> decide the boundaries

- Equal width Binning -> Make categories with same range
    - Pros
    - Simplicity
    - Interpretablity
    - Cons
    - Unequal Data Distribution

- Equal frequency Binning -> Group similar values
    - Pros
    - Robust to outliers
    - Preserve Data Distribution
    - Mitigate Sparse Intervals
    
    - Cons
    - Loss of Interpretablity
    - Can lead to inconsisitency
    - Information Loss

- Clustering -> use unsupervised machine learning to group the value
    - example -> K-means clustering

#### Numeric Feature Normalisation
- Features of vastly different scales can be problematic -> kilo/dollar or gram/dollar
- Feature standardisation -> normal scaling (x-U/sigma) -> mean = 0 and standard deviation = 1
- Feature Normalisation -> x' = (x-min/max-min) ->  0 < x < 1

### Prepare input
- Problem: different data sources -> different style, convention, time periods, data aggregation, keys, errors
- Data must be assembled, integrated, cleaned up

### Missing values
- missing values
- inter-dependent attributes

### out-range values
- unknown, unrecorded, irrelevant
- Reasons
    - Malfunctioning equipment
    - changes in experimental design
    - Collation of different datasets
- Most algorithm assume **missing on random**
- Some values may be significant like test in medical exam -> **Missing not at random**

#### Missing values 
- Miss on random -> Can not predict
- Miss not at Random -> Can have a rule to predict
- Disguised missing data -> Look for unusual or suspicious values in the dataset

### Simple imputation for numerical data
- Imputation -> methods to guess missing data
- Statistical Measurement Imputation -> simple strategies to fill out numerical data instance
    - Mean
    - Median
    - Mode
    - Pro: easy to compute, no loss for record
    - Con: biased other stats measures:
    -       Variance
    -       Standard Deviation

### Inaccurate Values
- Typographical errors
- In nominal attributes -> Values need to be checked for consistency
- In numerical -> Outliers need to be identified
- Errors may be deliberate -> not in the correct format

### Getting to know the data
- Simple visualisation tool
    - nominal -> histogram, numeric -> scatter plot
- 2-D, 3-D plot
- need to consult domain experts
- Take a sample for too much data