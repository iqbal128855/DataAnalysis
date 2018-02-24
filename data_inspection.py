import pandas as pd
import seaborn
import matplotlib.pyplot as plt


class data_inspection(object):
    def __init__(self, red, white): 
        print "Data Inspection" 
        self.red = red 
        self.white=white

    def data_description(self, wine_data):
        print("Length of the dataset is:" + str(len(wine_data)))
        print("Number of Columns :" + str(len(wine_data.columns)))
        print("Values: " + str(list(wine_data.columns.values)))
        print(wine_data.ix[:10,:4])

    # Frequency Distribution of Wine Quality
    def frequency_distribution(self,wine_data):
        print "Frequency Distribution of Wine Quality."
        print(wine_data.groupby("quality").size()*100 / len(wine_data))
        print()

    # Visualization
    # 1. Count Plot
    def countplot(self,wine_data):
        wine_data["quality"] = pd.Categorical(wine_data["quality"])
        seaborn.countplot(x="quality", data=wine_data)
        plt.xlabel("Quality level of wine (0-10 scale)")
        plt.show()
    
    # 2. Factor Plot
    def factorplot(self,wine_data):
        seaborn.factorplot(x="quality", y="alcohol", data=wine_data, kind="strip")
        plt.xlabel("Quality level of wine, 0-10 scale")
        plt.ylabel("Alcohol level in wine, % ABV")
        if wine_data.equals(self.red):
            plt.title("%Alcohol in each level of red wine's quality")
        else:
            plt.title("%Alcohol in each level of white wine's quality")
        plt.show()


