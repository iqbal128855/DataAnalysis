Data Inspection:

This is a summary of the steps performed to develop a data preprocessing steps using UCI Wine Data Repository. 

Step 1. 
Once data is downloaded, I checked the data to determine necessary preprocessing steps.
At first, I check for any missing data values, which the dataset does not contain. The idea was to remove the rows for which a column would be missing and if there are lot of missing values the idea was to impute those values with the mean of that column. 

Next, I check for outliers. This dataset does not contains too many outliers. So, I did not take any steps for outlier removal.

Next, I did not perform any further normalization as the data is almost uniformly distributed. Minmax standardization can be performed. 

After that, I explored the frequency distribution of the wines' quality. This is a very important step to get idea about the number fo classes present in a dataset. For both wines, the majority of the samples have quality ranks 5, 6, and 7 (on the 0-10 scale). There are no samples ranked 0, 1, 2, or 10. However, the quality ranks of the white samples on average are higher the those of the reds.

Finally, I looked into a factorplot of alcholol quality and alcohol percentage to identify correlations among them. It is obvious that they have a strong positive correlation.   
