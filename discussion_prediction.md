
Decision Trees:

I perform decision tree analysis to test nonlinear relationships among a set of explanatory variables (amount of residual sugar and alcohol in wine) and a binary, categorical response variable (`0` - if the quality of a wine sample is 3, 4, or 5, `1` - if 6, 7, 8, or 9). All possible cut points (for explanatory variables) are tested. For the present analysis, the entropy “goodness of split” criterion was used to grow the tree and a cost complexity algorithm was used for pruning the full tree into a final subtree. A classification tree was build for each of the wine sets (red and white) separately. In each set, 80% of the samples were used for the training, and 20% - for testing. 

Parameter Selection:
These are the parameters used for training Decision Trees.

criterion : ”gini”
The function to measure the quality of a split. 

splitter : ”best”
The strategy used to choose the split at each node. Supported strategies are “best” to choose the best split.

max_depth : None.
Nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.

min_samples_split : 2
The minimum number of samples required to split an internal node.

min_samples_leaf : 1
The minimum number of samples required to be at a leaf node:

min_weight_fraction_leaf : 0
The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node.

max_features : auto
The number of features to consider when looking for the best split.
If “auto”, then max_features=sqrt(n_features).

random_state : None

max_leaf_nodes : None
Grow a tree with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.

min_impurity_decrease : 0
A node will be split if this split induces a decrease of the impurity greater than or equal to this value.

The weighted impurity decrease equation is the following:

N_t / N * (impurity - N_t_R / N_t * right_impurity
                    - N_t_L / N_t * left_impurity)
where N is the total number of samples, N_t is the number of samples at the current node, N_t_L is the number of samples in the left child, and N_t_R is the number of samples in the right child.

class_weight : None

Weights associated with classes in the form {class_label: weight}. If not given, all classes are supposed to have weight one. For multi-output problems, a list of dicts can be provided in the same order as the columns of y.

Note that for multioutput (including multilabel) weights should be defined for each class of every column in its own dict. For example, for four-class multilabel classification weights should be `[{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}]` instead of `[{1:1}, {2:5}, {3:1}, {4:1}]`.

presort : False

Result Analysis:

-----Red Wine Decision Tree
confusion matrix: `[[106,  31]`
                  `[ 50, 133]]`
accuracy: 0.74687499999999996
f1_score: 0.7665706051873199)

-----White Wine Decision Tree
confusion matrix: `[[ 98,  42]`
                  `[ 47, 133]]`
accuracy: 0.72187500000000004)
f1_score: 0.74929577464788732)


The resulted trees are too big to be examined and visualized. It might indicate that the selected variables are not suitable for proper tree formation, or that the tree analysis is not suitable for these data. The work on this problem is continued in the next paragraph.
 
 
Random Forests: 

Here I perform random forest analysis to evaluate the importance of all the explanatory variables in predicting the quality of wine (binary target variable: `0` - if the quality of a wine sample is 3, 4, or 5, `1` - if 6, 7, 8, or 9). Analysis was performed for each wine set (red and white) separately. In each set, 80% of the sample were used for the training, and 20% - for testing. 

The analysis consists of two steps. Firstly, I create the random forest model with `32` trees and examine its results. Secondly, I train random forests with different numbers of trees (1-100) to see the effect of the number on the accuracy of the prediction.

The results of the random forest model with `32` trees for the **red** wine show that the accuracy of the prediction is `0.778` and the most important predictor is `alcohol`, followed by `volatile acidity`, `sulphates`, and `total sulfur dioxide`. It is interesting to note that the results of the *multivariate regression* for the **red** wine mark different set of variables (`chlorides`, `volatile acidity`, `sulphates`, and `pH`) as the most influential variables in describing the quality of wine.

The results of the random forest model with `32` trees for the **white** wine show that the accuracy of the prediction is `0.816` and the most important predictor is `alcohol`, followed by `volatile acidity`, and `density`. It is interesting to notice that the results of the *multivariate regression* for the **white** wine mark the same set of variables (`alcohol`, `volatile acidity`, and `density`) as the most influential variables in describing the quality of wine.

Training random forests with different numbers of trees (1-100) shows that, after approximately 24 trees, the subsequent growing of number of trees adds little to the overall accuracy of the forest. It is true for both sets of wine: red and white.

Parameter Selection:
n_estimators : 32
The number of trees in the forest.

criterion : "gini"
The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain. 

max_features : "auto"
The number of features to consider when looking for the best split:
If “auto”, then max_features=sqrt(n_features).

max_depth : None
The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.

min_samples_split : 2
The minimum number of samples required to split an internal node:

min_samples_leaf : 1
The minimum number of samples required to be at a leaf node.

min_weight_fraction_leaf : 0
The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided.

max_leaf_nodes : None
Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.

bootstrap : True
Whether bootstrap samples are used when building trees.

oob_score : False
Whether to use out-of-bag samples to estimate the generalization accuracy.

n_jobs : 1
The number of jobs to run in parallel for both fit and predict. If -1, then the number of jobs is set to the number of cores.

random_state : 123

verbose : 0
Controls the verbosity of the tree building process.

warm_start : False
When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just fit a whole new forest.

class_weight : None

Result Analysis:

-----Red Wine Random Forest
confusion matrix: `[[237,  75]`
                  `[ 65, 263]]`
accuracy: 0.78125
f1_score: 0.78978978978978975


-----White Wine Random Forest
confusion matrix: `[[238,  62]`
                  `[ 79, 261]]`
accuracy: 0.77968749999999998
f1_score: 0.78733031674208154


