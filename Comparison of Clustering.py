"""
Analyze the performance of these two methods, hierarchical and Kmeans clustering, on various subsets of
the county-level cancer risk data set.
"""

__author__ = 'liuyincheng'

import math
import alg_cluster
import random
import time
import matplotlib.pyplot as plt
def pair_distance(cluster_list, idx1, idx2):
    """
    Helper function to compute Euclidean distance between two clusters
    in cluster_list with indices idx1 and idx2

    Returns tuple (dist, idx1, idx2) with idx1 < idx2 where dist is distance between
    cluster_list[idx1] and cluster_list[idx2]
    """
    return (cluster_list[idx1].distance(cluster_list[idx2]), min(idx1, idx2), max(idx1, idx2))


def slow_closest_pairs(cluster_list):
    """
    Compute the set of closest pairs of cluster in list of clusters using brute force algorithm
    """
    [min_dist, ixd1, ixd2] = [float("inf"), -1, -1]
    num_clusters = len(cluster_list)
    closest_pairs = set([tuple([min_dist, ixd1, ixd2])])
    for ixdu in range(num_clusters - 1):
        for ixdv in range(ixdu + 1, num_clusters):
            dist = pair_distance(cluster_list, ixdu, ixdv)[0]
            # dist = cluster_list[ixdu].distance(cluster_list[ixdv])
            if dist < min_dist:
                min_dist = dist
                closest_pairs = set([tuple([dist, ixdu, ixdv])])
            elif dist == min_dist:
                closest_pairs.add(tuple([dist, ixdu, ixdv]))
    return closest_pairs


def fast_closest_pair(cluster_list):
    """
    Compute a closest pair of clusters in cluster_list
    using O(n log(n)) divide and conquer algorithm

    Returns a tuple (distance, idx1, idx2) with idx1 < idx 2 where
    cluster_list[idx1] and cluster_list[idx2]
    have the smallest distance dist of any pair of clusters
    """

    def fast_helper(cluster_list, horiz_order, vert_order):
        """
        Divide and conquer method for computing distance between closest pair of points
        Running time is O(n * log(n))

        horiz_order and vert_order are lists of indices for clusters
        ordered horizontally and vertically

        Returns a tuple (distance, idx1, idx2) with idx1 < idx 2 where
        cluster_list[idx1] and cluster_list[idx2]
        have the smallest distance dist of any pair of clusters

        """
        num_clusters = len(horiz_order)
        closest_pair = tuple()
        min_dist = float("inf")
        # base case
        if num_clusters <= 3:
            for idxu in horiz_order:
                for idxv in horiz_order:
                    if idxu != idxv:
                        (dist, ixd1, ixd2) = pair_distance(cluster_list, idxu, idxv)
                        if dist < min_dist:
                            min_dist = dist
                            closest_pair = (dist, ixd1, ixd2)
        # divide
        else:

            mid_idx = num_clusters / 2
            mid_value = 0.5 * (cluster_list[horiz_order[mid_idx - 1]].horiz_center() +
                               cluster_list[horiz_order[mid_idx]].horiz_center())

            [horiz_left, horiz_right] = [horiz_order[0: mid_idx], horiz_order[mid_idx:]]
            [vert_left, vert_right] = [[], []]
            for idx in vert_order:
                if idx in horiz_left:
                    vert_left.append(idx)
                elif idx in horiz_right:
                    vert_right.append(idx)
            left_closest_pairs = fast_helper(cluster_list, horiz_left, vert_left)
            right_closest_pairs = fast_helper(cluster_list, horiz_right, vert_right)
            if left_closest_pairs[0] <= right_closest_pairs[0]:
                closest_pair = left_closest_pairs
                min_dist = left_closest_pairs[0]
            else:
                closest_pair= right_closest_pairs
                min_dist = right_closest_pairs[0]
        # conquer
            mid_seq = []
            for idx in vert_order:
                if abs(cluster_list[idx].horiz_center() - mid_value) < min_dist:
                    mid_seq.append(idx)
            for idxu in range(0, len(mid_seq) -1):
                for idxv in range(idxu + 1, min(idxu + 4, len(mid_seq))):
                    (dist, idx1, idx2) = pair_distance(cluster_list, mid_seq[idxu], mid_seq[idxv])
                    if dist < min_dist:
                        min_dist = dist
                        closest_pair = tuple([dist, idx1, idx2])
        return closest_pair

    # compute list of indices for the clusters ordered in the horizontal direction
    hcoord_and_index = [(cluster_list[idx].horiz_center(), idx)
                        for idx in range(len(cluster_list))]
    hcoord_and_index.sort()
    horiz_order = [hcoord_and_index[idx][1] for idx in range(len(hcoord_and_index))]

    # compute list of indices for the clusters ordered in vertical direction
    vcoord_and_index = [(cluster_list[idx].vert_center(), idx)
                        for idx in range(len(cluster_list))]
    vcoord_and_index.sort()
    vert_order = [vcoord_and_index[idx][1] for idx in range(len(vcoord_and_index))]

    # compute answer recursively
    answer = fast_helper(cluster_list, horiz_order, vert_order)
    return (answer[0], min(answer[1:]), max(answer[1:]))


def hierarchical_clustering(cluster_list, num_clusters):
    """
    Compute a hierarchical clustering of a set of clusters
    Note: the function mutates cluster_list

    Input: List of clusters, number of clusters
    Output: List of clusters whose length is num_clusters
    """
    clusters = []       # Copy cluster_list to a new list named clusters
    for cluster in cluster_list:
        clusters.append(cluster.copy())
    while len(clusters) > num_clusters:
        closest_pair = fast_closest_pair(clusters)
        clusters[closest_pair[1]].merge_clusters(clusters[closest_pair[2]])
        clusters.pop(closest_pair[2])
    return clusters


def kmeans_clustering(cluster_list, num_clusters, num_iterations):
    """
    Compute the k-means clustering of a set of clusters

    Input: List of clusters, number of clusters, number of iterations
    Output: List of clusters whose length is num_clusters
    """

    # initialize k-means clusters to be initial clusters with largest populations
    clusters = []       # Copy cluster_list to a new list named clusters
    for cluster in cluster_list:
        clusters.append(cluster.copy())
    centers = sorted(clusters, key = lambda cluster: cluster.total_population(), reverse = True)[0: num_clusters]
    print centers
    for _idx in range(num_iterations):
        kmeans_clusters = [[] for _i in range(num_clusters)]
        for clus in clusters:
            min_dist = float('inf')
            classified_cent = 0
            for center_idx in range(len(centers)):
                dist = clus.distance(centers[center_idx])
                if dist < min_dist:
                    classified_cent = center_idx
                    min_dist = dist
            kmeans_clusters[classified_cent].append(clus.copy())
        print kmeans_clusters
        centers = []
        for new_clusters in kmeans_clusters:
            assert len(new_clusters) > 0, "Empty new cluster"
            center = new_clusters[0]
            if len(new_clusters) > 1:
                for clus_idx in range(1, len(new_clusters)):
                    center.merge_clusters(new_clusters[clus_idx])
            centers.append(center)
    return centers


def gen_random_clusters(num_clusters):
    clusters = [alg_cluster.Cluster(set([]), random.random() * 2 - 1, random.random() * 2 - 1, 1, 0)
                for _i in range(num_clusters)]
    return clusters


"""
Question #1, compare the efficiency of two closest distance calculators, slow_closet_pairs and fast_closet_pair
"""
slow_time = []
fast_time = []
for clusters_size in range(2, 201):
    clusters = gen_random_clusters(clusters_size)
    time0 = time.time()
    slow_closest_pairs(clusters)
    time1 = time.time()
    fast_closest_pair(clusters)
    time2 = time.time()
    slow_time.append(time1 - time0)
    fast_time.append(time2 - time1)
plt.plot(range(2, 201), slow_time, 'r', lw = 2, label = "slow_closest_pairs")
plt.plot(range(2, 201), fast_time, 'g', lw = 2, label = "fast_closest_pair")
plt.legend(loc = 'upper left')
plt.xlabel('Number of initial clusters')
plt.ylabel('Running time (unit = second)')
plt.title('Comparison of Running time between two closest pair functions \n (Desktop Python)')
plt.grid()
plt.show()

