Approximate Nearest Neighbors, using Hierarchical K-Means clustering

This module is built on top of the scikit-learn KMeans clustering model. A new class called HierarchicalKMeans is created, which builds a hierarchical clustering where a dataset is clustered as desired using KMeans, and at each additional stratum, the centroids of the lower stratum
are clustered using another KMeans model. 

As the Hierarchical clustering is built, it is represented in an N-Ary tree called ClusterTree, which includes a method that allows a query point to be classified at each stratum from the top down. Functioning like a form of Locality-Sensitive Hashing
