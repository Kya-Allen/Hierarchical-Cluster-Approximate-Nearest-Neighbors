import HCANN as hc
import numpy as np
import unittest as unit

class ClusterTreeTestCase(unit.TestCase):
    def setUp(self):
        self.level0_centroids = np.array([1.5, 2, 2.5, 5, 5.5, 6, 10, 11, 10, 15, 15.5, 16, 20, 21, 22, 28, 29, 30])
        self.level0_labels = np.array([0])
        self.level1_centroids = np.array([2, 5.5, 11, 15.5, 21, 29])
        self.level1_labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5])
        self.level2_centroids = np.array([6, 22])
        self.level2_labels = np.array([0, 0, 0, 1, 1, 1])
        self.my_tree = hc.ClusterTree()

    def test_one_layer(self):
        self.my_tree.insert_clusters(self.level0_centroids, self.level0_labels)
        correct_level0_centroids = self.level0_centroids
        actual_level0_centroids = np.array([node.centroid for node in self.my_tree.leaves])
        self.assertEqual(correct_level0_centroids.all(), actual_level0_centroids.all())

    def test_two_layer(self):
        self.my_tree.insert_clusters(self.level0_centroids, self.level0_labels)
        self.my_tree.insert_clusters(self.level1_centroids, self.level1_labels)
        correct_level0_centroids = self.level0_centroids
        correct_level1_cetroids = self.level1_centroids
        actual_level0_centroids = np.array([node.centroid for node in self.my_tree.leaves])
        actual_level1_centroids = np.array([node.centroid for node in self.my_tree.root.children])
        self.assertEqual(correct_level0_centroids.all(), actual_level0_centroids.all())
        self.assertEqual(correct_level1_cetroids.all(), actual_level1_centroids.all())

    def test_nn_traversal(self):
        self.my_tree.insert_clusters(self.level0_centroids, self.level0_labels)
        self.my_tree.insert_clusters(self.level1_centroids, self.level1_labels)
        query_point = 1.9
        correct_result = [0, 1]
        actual_result = self.my_tree.nn_traversal([query_point]) 
        self.assertEqual(correct_result, actual_result)


        

#my_tree.insert_clusters(level0_centroids, level0_labels)
#my_tree.insert_clusters(level1_centroids, level1_labels)
#my_tree.insert_clusters(level2_centroids, level2_labels)
        
if __name__ == '__main__':
    unit.main()

