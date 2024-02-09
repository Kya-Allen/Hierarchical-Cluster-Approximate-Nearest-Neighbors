import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

class ClusterNode():
  '''Individual Node for ClusterTree

  Parameters
  ----------
  centroid : array-like, (or None for root node)
    vector describing the position of the cluster

  parent : ClusterNode, (or None for root node)
    the cluster in the upper-adjacent stratum, which this cluster's centroid belongs to.

  Attributes
  ----------
  children : array-like
    list of clusters in the lower-adjacent stratum belonging to this cluster.
  '''
  def __init__(self, centroid, parent):
    self.centroid = centroid
    self.parent = parent
    self.children = []
    return

  children = []

class ClusterTree():
  '''N-ary Tree of HierarchicalKMeans clusters

  Attributes
  ----------
  previous_set : array-like
    stores a list of the last set of nodes insterted by method insert_clusters()
  '''
  def __init__(self):
    self.root = ClusterNode(None, None)
    self.leaves: list[ClusterNode] = []
    self.previous_set: list[ClusterNode] = []
    return

  def insert_clusters(self, centroids: NDArray, labels: NDArray):
    '''insert single stratum of HierarchicalKMeans from the bottom up, into the tree as ClusterNodes

    Parameters
    ----------
    centroids : NDArray
      centroids of the clusters in the given stratum

    labels : NDArray
      cluster indecies of the items clustered
    '''
    set_leaves: bool = False
    set_children: bool = True
    if self.leaves == []:
      set_leaves = True
      set_children = False
    current_set: list[ClusterNode] = []
    self.root.children.clear()
    for centroid_index, centroid in enumerate(centroids):
      node = ClusterNode(centroid, self.root)
      self.root.children.append(node)
      if set_leaves: self.leaves.append(node)
      elif set_children: self.__setting_children(labels, node, centroid_index)
      current_set.append(node)
    self.previous_set = current_set
    return
  
  def __setting_children(self, labels, node, centroid_index):
    if len(self.previous_set) != len(labels): raise Exception("labels should be the same size as previous set")
    for i, j, in zip(self.previous_set, labels):
      self.__set_child(node, i, j, centroid_index)
  
  def __set_child(self, node, lower_node, label, centroid_index):
    if centroid_index == label:
      lower_node.parent = node 
      node.children.append(lower_node)

  def nn_traversal(self, query: ArrayLike):
    '''Retrieve full stratum path for query point. Datapoints beloging to the final cluster in the list are the approximate N Nearest Neighbors

    Parameters
    ----------
    query : ArrayLike
      query vector

    Returns
    -------
    indecies : ArrayLike
      list containing an index of the nearest cluster at each strata
    '''
    query: NDArray = np.array(query)
    query.reshape(-1, 1)
    clusters = self.root.children
    indecies = []
    at_bottom = False

    return self.__nn_strata(clusters, indecies, query, at_bottom)
  
  def __nn_strata(self, clusters, indecies, query, at_bottom):
    neighborhood = NearestNeighbors(n_neighbors=1)
    neighborhood.fit(np.array([x.centroid for x in clusters]).reshape(-1, 1))
    nearest = neighborhood.kneighbors(np.array([query]).reshape(-1, 1), return_distance=False)
    clusters = clusters[nearest[0][0]].children
    indecies.append(nearest[0][0])
    if at_bottom: return indecies
    if clusters[0].children == []: at_bottom = True
    return self.__nn_strata(clusters, indecies, query, at_bottom)

 
class HierarchicalKMeans():
  '''Hierarchical K-Means Clustering

  Parameters
  ----------

  n_strata : int, default=2
    The number of strata that the cluster hierarchy will be composed of.
    Each additional strata will cluster the centroids of the previous stratum

  The following parameters will be uniformly applied to the sci-kit learn KMeans clustering model that fits each strata.
  The documentation for these parameters are copied directly.

  init : {'k-means++', 'random'}, callable or array-like of shape \
            (n_clusters, n_features), default='k-means++'
        Method for initialization:

        * 'k-means++' : selects initial cluster centroids using sampling \
            based on an empirical probability distribution of the points' \
            contribution to the overall inertia. This technique speeds up \
            convergence. The algorithm implemented is "greedy k-means++". It \
            differs from the vanilla k-means++ by making several trials at \
            each sampling step and choosing the best centroid among them.

        * 'random': choose `n_clusters` observations (rows) at random from \
        data for the initial centroids.

        * If an array is passed, it should be of shape (n_clusters, n_features)\
        and gives the initial centers.

        * If a callable is passed, it should take arguments X, n_clusters and a\
        random state and return an initialization.

        For an example of how to use the different `init` strategy, see the example
        entitled :ref:`sphx_glr_auto_examples_cluster_plot_kmeans_digits.py`.

    n_init : 'auto' or int, default='auto'
        Number of times the k-means algorithm is run with different centroid
        seeds. The final results is the best output of `n_init` consecutive runs
        in terms of inertia. Several runs are recommended for sparse
        high-dimensional problems (see :ref:`kmeans_sparse_high_dim`).

        When `n_init='auto'`, the number of runs depends on the value of init:
        10 if using `init='random'` or `init` is a callable;
        1 if using `init='k-means++'` or `init` is an array-like.

        .. versionadded:: 1.2
           Added 'auto' option for `n_init`.

        .. versionchanged:: 1.4
           Default value for `n_init` changed to `'auto'`.

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a
        single run.

    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.

    verbose : int, default=0
        Verbosity mode.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    copy_x : bool, default=True
        When pre-computing distances it is more numerically accurate to center
        the data first. If copy_x is True (default), then the original data is
        not modified. If False, the original data is modified, and put back
        before the function returns, but small numerical differences may be
        introduced by subtracting and then adding the data mean. Note that if
        the original data is not C-contiguous, a copy will be made even if
        copy_x is False. If the original data is sparse, but not in CSR format,
        a copy will be made even if copy_x is False.

    algorithm : {"lloyd", "elkan"}, default="lloyd"
        K-means algorithm to use. The classical EM-style algorithm is `"lloyd"`.
        The `"elkan"` variation can be more efficient on some datasets with
        well-defined clusters, by using the triangle inequality. However it's
        more memory intensive due to the allocation of an extra array of shape
        `(n_samples, n_clusters)`.

    Attributes
    ----------
    strata : dict[str, KMeans]
      A dictionary generated by method build_stratum().
      Each key is in the form of 'tn' where 'n' is the index of the strata
      Each value is a fitted KMeans object

    tree : ClusterTree
      A ClusterTree object generated upon fitting the model, that allows top down querying of a query point into a cluster at the lowest stratum.
      Allowing for Approximate Nearest Neighbors search.
      returning the full strata path.
  '''

  def __init__(self, n_strata=2, init='k-means++', n_init='warn', max_iter=300, tol=0.0001, verbose=0, random_state=None, copy_x=True, algorithm='lloyd'):
    self.n_strata = n_strata
    self.init = init
    self.n_init = n_init
    self.max_iter = max_iter
    self.tol = tol
    self.verbose = verbose
    self.random_state = random_state
    self.copy_x = copy_x
    self.algorithm = algorithm
    self.strata: dict = {}
    self.tree: ClusterTree = ClusterTree()

  def fit_stratum(self, stratum_data, n_clusters) -> KMeans:
    '''Compute K-Means clustering for a single stratum of the hierarchical clustering'''
    clustering: KMeans = KMeans(n_clusters=n_clusters, init=self.init, n_init=self.n_init, max_iter=self.max_iter, tol=self.tol, verbose=self.verbose, random_state=self.random_state, copy_x=self.copy_x, algorithm=self.algorithm)
    clustering.fit(stratum_data)
    return clustering

  def determine_k(self, n: int, max_comparisons: int) -> int:
    '''determine and compute number of clusters for a given stratum based on max_comparisons'''
    return round(n / max_comparisons)

  def build_stratum(self, peak_comparisons, max_comparisons, cluster_space, tier):
    '''Determine K-Means clustering for a single stratum of the hierarchical clustering'''
    num_clusters: int = self.determine_k(peak_comparisons, max_comparisons)
    current_stratum: KMeans = self.fit_stratum(cluster_space, num_clusters)
    self.strata[f't{tier}'] = current_stratum
    return current_stratum

  def fit(self, data: ArrayLike, max_comparisons: int):
    '''Compute the hierarchical K-Means clustering

    Parameters
    ----------
    data : array-like
      Training data to cluster.

    max_comparisons : int
      misnomer. Expected number of nearest-neighbor comparisons per strata
    '''
    data: NDArray = np.array(data)
    peak_comparisons: int = data.size
    cluster_space: NDArray = data
    tier: int = 0
    while peak_comparisons > max_comparisons:
      current_stratum = self.build_stratum(peak_comparisons, max_comparisons, cluster_space, tier)
      peak_comparisons = current_stratum.cluster_centers_.shape[0]
      tier += 1
      cluster_space = current_stratum.cluster_centers_
      self.tree.insert_clusters(cluster_space, current_stratum.labels_)
    return

  def predict(self, data: ArrayLike) -> list:
    '''Compute the full strata path of cluster membership for a new dataset

    Parameters
    ----------
    data : array-like
      data to classify
    '''
    results = []
    for point in data:
      strata_path = self.tree.nn_traversal(point)
      results.append(strata_path)
    return results