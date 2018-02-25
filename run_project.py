import pandas as pd
from data_inspection import data_inspection
from determine_attribute_quality import determine_attribute_quality
from prediction import prediction


def run_data_inspection(data_red,data_white):
    di_obj=data_inspection(data_red,data_white)
    # Data Set Description
    print "Red Wine Data Description" 
    di_obj.data_description(data_red)
    di_obj.frequency_distribution(data_red)
    di_obj.countplot(data_red)
    di_obj.factorplot(data_red)
    
    print "White Wine Data Description" 
    di_obj.data_description(data_white)
    di_obj.frequency_distribution(data_white)
    di_obj.countplot(data_white)
    di_obj.factorplot(data_white)

def run_determine_attribute_quality(data_red,data_white):
    daq_obj=determine_attribute_quality(data_red,data_white)
    processed_data_red=daq_obj.remove_column_spaces(data_red)
    processed_data_white=daq_obj.remove_column_spaces(data_red)
    
    #Regression
    print "Red Wine Regression" 
    daq_obj.regression(processed_data_red)
    print "White Wine Regression"
    daq_obj.regression(processed_data_white)
    

def run_prediction(data_red,data_white):
    pred_obj=prediction(data_red,data_white)
    # Decision Tree Classifier
    processed_data_red=pred_obj.remove_column_spaces(data_red)
    processed_data_white=pred_obj.remove_column_spaces(data_red)
    
    print "Red Wine Decision Tree"
    pred_obj.decision_tree(processed_data_red)
    print "White Wine Decision Tree"
    pred_obj.decision_tree(processed_data_white)
    
    # Random Forest Classifier
    print "Red Wine Random Forest"
    pred_obj.random_forest(processed_data_red)
    print "White Wine Random Forest"
    pred_obj.random_forest(processed_data_white)
    

if __name__=="__main__":
    # Read Data
    red=pd.read_csv('winequality-red.csv', low_memory=False, sep=';')
    white=pd.read_csv('winequality-white.csv', low_memory=False, sep=';')  
    
    run_data_inspection(red,white)
    run_determine_attribute_quality(red,white)
    run_prediction(red,white) 
