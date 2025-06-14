#!/bin/bash

# Output file
output="main.md"

# Overwrite main.md with title
echo "# Machine Learning Notes" > $output
echo "" >> $output

# Ordered merge of files with section headers
for file in \
    week1.md \
    Lifecycle.md \
    K-NN.md \
    Probs.md \
    Naive_Bayes.md \
    DecisionTree.md \
    Regression.md \
    SVM.md \
    Evaluation.md \


do
    echo "---" >> $output
    echo "## $(basename "$file" .md)" >> $output
    echo "" >> $output
    cat "$file" >> $output
    echo "" >> $output
done

echo "✅ Merged into $output"
