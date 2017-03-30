#Python Implementation of K-means clustering(PAM) and K-modes clustering(Huang)
A robust implementation of K-means and K-mode clustering algorithm with max-min normalization in Python to cluster continuous variables. K-means is for datasets with continous attributes and K-modes is for datasets with categorical attributes.

## Download and Usage
To use this program you can download this package from Github and run the following command after you are under the directory of `K-means`:

		python kmeans.py glass.csv 2 glass.out
		
		python kmeans.py wine_data.csv 4 wine_data.out

The first argument can be any input file. The second argument is the k, the number of clusters we want the program to gather. The third argument is the name of any output file where each cluster and its centroids are written.


To run k-modes: (the dafault dataset is the mushroom.training dataset) 

		python kmodes.py

The dependencies for the programs include (Python2.7):

- pandas 
- numpy
- sklearn


## Other
Alpha version, so it might not be the best implementation.
