import numpy as np
from sklearn.cluster import KMeans

class KTree():
  def __init__(self, depth):
    return

class HierarchicalKMeans():
  def __init__(self, n_strata=2, init='k-means++', n_init='warn', max_iter='300', tol='0.0001', verbose=0, random_state=None, copy_x=True, algorithm='lloyd'):
    self.n_strata = n_strata
    self.init = init
    self.n_init = n_init
    self.max_iter = max_iter
    self.tol = tol
    self.verbose = verbose
    self.random_state = random_state
    self.copy_x = copy_x
    self.algorithm = algorithm

  strata: {}
  cluster_scheme: list[int] = []

  def hierarchical_division(self, n, scheme) -> int:
    result = n
    for cluster_set in scheme:
      result = result / cluster_set
    return result

  def determine_schema(self, data, max_comparisons=100) -> list[int]:
    scheme = [2]
    scheme_index = 0
    while self.hierarchical_division(np.shape(data)[0], scheme) > max_comparisons:
      scheme[0] += 1
      if scheme[0] >= max_comparisons:
        scheme.append(0)
        scheme_index += 1
    self.cluster_scheme = scheme
    return self.cluster_scheme

  def fit_strata(self, strata_data, n_clusters) -> KMeans:
    clustering: KMeans = KMeans(n_clusters, self.init, self.n_init, self.max_iter, self.tol, self.verbose, self.random_state, self.copy_x, self.algorithm)
    clustering.fit(strata_data)
    return clustering

  def get_cluster():
    return

  def grow_hierarchy(self, data, cluster_scheme: list[int], max_comparisons):
    self.determine_schema(data, max_comparisons=max_comparisons)
    cluster_targets = [data]
    num_datasets = 1
    index = 0
    while index < len(cluster_scheme):
      fits = []
      for target in cluster_targets:
        fit: KMeans = self.fit_strata(target, cluster_scheme[index])
        fits.append(fit)
      self.strata.append(fits)
      num_datasets = cluster_scheme[index]
      index += 1
      cluster_targets = []
      for cluster



    fit: KMeans = self.fit_strata(data, cluster_scheme[0])
    self.strata.append(fit)
    if len(cluster_scheme) > 1:
      for i in range():
        #do stuff