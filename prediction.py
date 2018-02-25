from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn import preprocessing
from sklearn.linear_model import LassoLarsCV
from sklearn.metrics import f1_score
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import operator
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi


class prediction(object):
    def __init__(self,red,white):
        self.red=red
        self.white=white

    def remove_column_spaces(self,wine_data):
        wine_data.columns = [x.strip().replace(' ', '_') for x in wine_data.columns]
        return wine_data

    def decision_tree(self, wine_data):
        
        self.w=wine_data
        
        # Transform quality into 2 class: 0:{3,4,5}, 1:{6,7,8,9}
        label = {3: 0, 4: 0, 5: 0, 6: 1, 7: 1, 8: 1, 9: 1}
        wine_data['quality_c'] = wine_data['quality'].map(label)
      
        # Split into Training and Testing sets
        self.predictors = wine_data[['density', 
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
        self.targets = wine_data.quality_c
        self.pred_train, self.pred_test, self.target_train, self.target_test = train_test_split(self.predictors, self.targets, test_size=.2)

        # Build model on training data
        self.classifier = DecisionTreeClassifier()
        self.classifier = self.classifier.fit(self.pred_train, self.target_train)
        self.predictions = self.classifier.predict(self.pred_test)
        
        # Print the confusion matrix, accuracy and f1_score of the model
        print('confusion matrix:', sklearn.metrics.confusion_matrix(self.target_test, self.predictions))
        print('accuracy:', sklearn.metrics.accuracy_score(self.target_test, self.predictions))
        print('f1_score:',sklearn.metrics.f1_score(self.target_test,self.predictions))
        
        # Export the tree for viewing
        if self.w.equals(self.red):
            export_graphviz(self.classifier, out_file="red_decision_tree.dot")
        else:
            export_graphviz(self.classifier, out_file="white_decision_tree.dot")
    

    def random_forest(self, wine_data):
        # Transform quality into 2 class: 0:{3,4,5}, 1:{6,7,8,9}
        label = {3: 0, 4: 0, 5: 0, 6: 1, 7: 1, 8: 1, 9: 1}
        wine_data['quality_c'] = wine_data['quality'].map(label)
        
        # Split into training and testing sets
        self.predictors = wine_data[['density', 
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

        self.targets = wine_data.quality_c

        self.pred_train, self.pred_test, self.target_train, self.target_test = train_test_split(self.predictors, self.targets, test_size=.4)

        # Build model on training data
        self.classifier = RandomForestClassifier(n_estimators=32)
        self.classifier = self.classifier.fit(self.pred_train, self.target_train)
        self.predictions = self.classifier.predict(self.pred_test)
        
        # Print the confusion matrix, accuracy and f1_score of the model
        print('confusion matrix:', sklearn.metrics.confusion_matrix(self.target_test, self.predictions))
        print('accuracy:', sklearn.metrics.accuracy_score(self.target_test, self.predictions))
        print('f1_score:',sklearn.metrics.f1_score(self.target_test,self.predictions))
        
        # Display the relative importance of each predictive variable
        model = ExtraTreesClassifier()
        model.fit(self.pred_train, self.target_train)

        print('Importance of Predictors:')
        dct = dict()
        for i in range(len(self.predictors.columns)):
            dct[self.predictors.columns[i]] = model.feature_importances_[i]
        print(sorted(dct.items(), key=operator.itemgetter(1), reverse=True))

        # Finding the best number of trees
        n = 100
        accuracy = [0]*n

        for i in range(n):
            self.classifier = RandomForestClassifier(n_estimators=i+1)
            self.classifier = self.classifier.fit(self.pred_train, self.target_train)
            self.predictions = self.classifier.predict(self.pred_test)
            accuracy[i] = sklearn.metrics.accuracy_score(self.target_test, self.predictions)

        plt.plot(range(1, n+1), accuracy)
        plt.xlabel("Number of Trees")
        plt.ylabel("Prediction Accuracy")
        plt.title("Effect of the Number of Trees on the Prediction Accuracy")
        plt.show()

        print(accuracy)



