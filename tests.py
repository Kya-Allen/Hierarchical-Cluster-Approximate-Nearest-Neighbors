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

class HierarchicalKMeansTestCase(unit.TestCase):
    def setUp(self):
        self.upper_cluster_1 = np.random.normal(200, 50, 20)
        self.upper_cluster_2 = np.random.normal(50, 50, 20)
        np.set_printoptions(suppress=True)
        self.upper_data = np.concatenate((self.upper_cluster_1, self.upper_cluster_2))
        self.full_data = np.array([])
        for datum in self.upper_data:
            x = np.random.normal(datum, 5, 20)
            self.full_data = np.concatenate((self.full_data, x))
        self.model = hc.HierarchicalKMeans()
        self.model.fit(self.full_data.reshape(-1, 1), 3)

    def test_strata(self):
        self.assertEqual(5, len(self.model.strata))
        correct_higest_n_clusters = 3
        self.assertEqual(correct_higest_n_clusters, self.model.strata['t4'].n_clusters)

    def test_prediction(self):
        print(self.model.predict([[125]]))
        query_classification = self.model.predict([[125]])
        test_list = [
            self.model.tree.root.children[query_classification[0][0]].centroid[0],
            self.model.tree.root.children[query_classification[0][0]].children[query_classification[0][1]].centroid[0],
            self.model.tree.root.children[query_classification[0][0]].children[query_classification[0][1]].children[query_classification[0][2]].centroid[0],
            self.model.tree.root.children[query_classification[0][0]].children[query_classification[0][1]].children[query_classification[0][2]].children[query_classification[0][3]].centroid[0],
            self.model.tree.root.children[query_classification[0][0]].children[query_classification[0][1]].children[query_classification[0][2]].children[query_classification[0][3]].children[query_classification[0][4]].centroid[0],
        ]

        # Test that each subsequent cluster centroid gets closer and closer to the query point. This should suffice to show that the Hierarchical clustering is correct
        proximity_path = []
        for index, centroid in enumerate(test_list):
            if (index + 1) >= len(test_list): break
            first_diff = abs(125 - centroid)
            next_diff = abs(125 - test_list[index+1]) 
            proximity_path.append(first_diff - next_diff)
        for improvement in proximity_path:
            self.assertGreater(improvement, 0)
        
if __name__ == '__main__':
    unit.main()

