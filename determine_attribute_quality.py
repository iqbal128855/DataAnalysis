from sklearn.cross_validation import train_test_split
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import operator
from sklearn import preprocessing
from sklearn.linear_model import LassoLarsCV
import numpy as np
from sklearn.metrics import mean_squared_error
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi

class determine_attribute_quality(object):
    
    def __init__(self,red,white):
        self.red=red
        self.white=white
    
    def remove_column_spaces(self,wine_data):
        wine_data.columns = [x.strip().replace(' ', '_') for x in wine_data.columns]
        return wine_data
    
    def regression(self,wine_data):

        self.pred = wine_data[['density', 
                               'alcohol', 
                               'sulphates', 
                               'pH', 
                               'volatile_acidity', 
                               'chlorides', 
                               'fixed_acidity',
                               'citric_acid', 
                               'residual_sugar', 
                               'free_sulfur_dioxide', 
                               'total_sulfur_dioxide']]
        self.predictors = self.pred.copy()
        self.targets = wine_data.quality

        # Normalization
        self.predictors = pd.DataFrame(preprocessing.scale(self.predictors))
        self.predictors.columns = self.pred.columns
    
        # Split into Training and Testing sets
        (self.pred_train, 
         self.pred_test, 
         self.target_train, 
         self.target_test) = train_test_split(self.predictors, 
                                             self.targets, 
                                             test_size=.2, 
                                             random_state=123)

        # Lasso Regression Model
        self.model = LassoLarsCV(cv=10, precompute=False).fit(self.pred_train, self.target_train)

        print('Predictors and their Regression coefficients:')
        d = dict(zip(self.predictors.columns, self.model.coef_))
        for k in d:
            print(k, ':', d[k])

        # Plot Coefficient Progression
        m_log_alphas = -np.log10(self.model.alphas_)
    
        plt.plot(m_log_alphas, self.model.coef_path_.T)
        print('\nAlpha:', self.model.alpha_)
        plt.axvline(-np.log10(self.model.alpha_), linestyle="dashed", color='k', label='alpha CV')
        plt.ylabel("Regression coefficients")
        plt.xlabel("-log(alpha)")
        plt.title('Regression coefficients progression for Lasso paths')
        plt.show()

        # Plot MSE for each fold
        m_log_alphascv = -np.log10(self.model.cv_alphas_)
        plt.plot(m_log_alphascv, self.model.cv_mse_path_, ':')
        plt.plot(m_log_alphascv, self.model.cv_mse_path_.mean(axis=-1), 'k', label='Average across the folds', linewidth=2)
        plt.legend()
        plt.xlabel('-log(alpha)')
        plt.ylabel('Mean Squared Error')		
        plt.title('Mean Squared Error on Each Fold')
        plt.show()

        # Mean Squared Error from Training and Test data
        self.train_error = mean_squared_error(self.target_train, self.model.predict(self.pred_train))
        self.test_error = mean_squared_error(self.target_test, self.model.predict(self.pred_test))
        print('\nMean squared error for training data:', self.train_error)
        print('Mean squared error for test data:', self.test_error)

        self.rsquared_train = self.model.score(self.pred_train, self.target_train)
        self.rsquared_test = self.model.score(self.pred_test, self.target_test)
        print('\nR-square for training data:', self.rsquared_train)
        print('R-square for test data:', self.rsquared_test)

 
