1. FILE LIST

classify.py: the LPPT classifier
map.shp: the map of a simulated data
sample_sp.csv: the training data extracted from map.shp
test.csv: the unseen objects in map.shp

2. HOW TO USE THIS PROGRAM

You must have python install as well as the scikit-learn package. This has been tested under UBUNTU 18.04 and PYTHON 2.7.15. Please unpack the code and change working directory to where classify.py is. Then input the following command using a terminal:

$ python classify.py sample_sp.csv test.csv

There will be five files generated under the same directory. "NB.csv" is the Naive Bayesian classification results. "prob_NB.csv" is the corresponding soft classification result from Naive Bayesian. "NB.csv" is the LPPT classification results. "prob_NB_sp.csv" is the corresponding soft classification result using LPPT. "result_real.csv" is the real categories of all unseen objects.

3. INPUT FILE FORMAT

sample_sp.csv and test.csv follows the following format.

The first column is the number of categories. The second column is the polygon id of each object in the training data. The succeeding three columns (the 3rd to 5th columns) are the conditional features. If map.shp have four conditional features, the sample_sp.csv will have the four conditional features from the third column. Then the succeeding three columns (the 6th to 9th columns) are the number of objects of different categories in the neighborhood of the current object in the sample data. The final column is the real category of the object.
