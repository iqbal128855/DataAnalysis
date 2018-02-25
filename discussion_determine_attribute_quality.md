### Determine Attribute Quality: 

This is a very important step for feature selection. My idea was to select most 
important features to use for classification. I am intending to use Decision 
Trees or Random Forest for classifictaion. Utilizing the best features in the 
upper nodes is a very good choice for good classification performance. Random 
Forest also employs a variable importance but that can be little biased to use 
because  it tends to give importance to features that it has already used. To 
find an indepent estimate I perform a Lasso Regression to find the most important 
features. 

Before performing regression I normalize the data set.

Data were randomly split into a training set (80% of the observations) and a test 
set (20% of the observations). Using this kind of split, though, a little old trend. 
Modern trend is to use a 98% Training data and 1% Test Data and 1% Validation data. 
Then I used a 10 fold cross validation to estimate the lasso regression model. 
To identify the best subset I used Mean Squared Error. In Deep Learning community, 
the trend is to use negative log likelihood error. But, in machibe learning due to 
relatively lower dimension of data Mean Squared Error ensures optimization in convexity.

Not all predictor variables were retained in the selected models. For the red wine, 
citric acid, fixed acidity, and free sulfur dioxide variables received zero coefficients, 
and therefore do not participate in the prediction. The results of the training indicate 
the alcohol, volatile acidity, and sulphates variables as the most strongly associated with 
the quality of wine and, therefore, the most influential for the prediction. Interestingly, 
this results differ from the ones of random forests analysis and multivariate regression. 
Mean squared error and R-squared values prove the model being robust for testing on new 
examples. The predictors account for 33% of the variance in the target variable.

For the white wine, citric acid and fixed acidity variables received zero coefficients, 
and therefore do not participate in the prediction. The results of the training indicate 
the alcohol, residual sugar, density, and volatile acidity variables as the most strongly 
associated with the quality of wine and, therefore, the most influential for the prediction. 
Interestingly, this results similar to the ones of random forests analysis and multivariate 
regression. Mean squared error and R-squared values prove the model being robust for testing 
on new examples. The predictors account for 28% of the variance in the target variable.

We can see that for the white wine the predictive algorithms are more unanimous in the 
selection of most influential predictors than they are for red.
