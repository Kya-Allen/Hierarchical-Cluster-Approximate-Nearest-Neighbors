import HCANN as hc
import numpy as np

level0_centroids = np.array([1.5, 2, 2.5, 5, 5.5, 6, 10, 11, 10, 15, 15.5, 16, 20, 21, 22, 28, 29, 30])
level0_labels = np.array([0])
level1_centroids = np.array([2, 5.5, 11, 15.5, 21, 29])
level1_labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5])
level2_centroids = np.array([6, 22])
level2_labels = np.array([0, 0, 0, 1, 1, 1])

my_tree = hc.ClusterTree()
my_tree.insert_clusters(level0_centroids, level0_labels)
my_tree.insert_clusters(level1_centroids, level1_labels)
my_tree.insert_clusters(level2_centroids, level2_labels)

for cluster in my_tree.root.children:
    print(cluster.children[0].children[0].centroid)

