"""
Analyze the connectivity of a computer network as it undergoes a cyber-attack.
In particular, we will simulate an attack on this network in which an increasing number of
servers are disabled.  In computational terms, we will model the network by an undirected
graph and repeatedly delete nodes from this graph. We will then measure the resilience of
the graph in terms of the size of the largest remaining connected component as a function of
the number of nodes deleted.
"""

__author__ = 'liuyincheng'

import urllib2
import random
import time
import math
import matplotlib.pyplot as plt

NETWORK_URL = "http://storage.googleapis.com/codeskulptor-alg/alg_rf7.txt"


class UPATrial:
    """
    Simple class to encapsulate optimized trials for the UPA algorithm

    Maintains a list of node numbers with multiple instance of each number.
    The number of instances of each node number are
    in the same proportion as the desired probabilities

    Uses random.choice() to select a node number from this list for each trial.
    """

    def __init__(self, num_nodes):
        """
        Initialize a UPATrial object corresponding to a
        complete graph with num_nodes nodes

        Note the initial list of node numbers has num_nodes copies of
        each node number
        """
        self._num_nodes = num_nodes
        self._node_numbers = [node for node in range(num_nodes) for dummy_idx in range(num_nodes)]


    def run_trial(self, num_nodes):
        """
        Conduct num_nodes trials using by applying random.choice()
        to the list of node numbers

        Updates the list of node numbers so that each node number
        appears in correct ratio

        Returns:
        Set of nodes
        """

        # compute the neighbors for the newly-created node
        new_node_neighbors = set()
        for _ in range(num_nodes):
            new_node_neighbors.add(random.choice(self._node_numbers))

        # update the list of node numbers so that each node number
        # appears in the correct ratio
        self._node_numbers.append(self._num_nodes)
        for dummy_idx in range(len(new_node_neighbors)):
            self._node_numbers.append(self._num_nodes)
        self._node_numbers.extend(list(new_node_neighbors))

        #update the number of nodes
        self._num_nodes += 1
        return new_node_neighbors


def copy_graph(graph):
    """
    Make a copy of a graph
    """
    new_graph = {}
    for node in graph:
        new_graph[node] = set(graph[node])
    return new_graph


def delete_node(ugraph, node):
    """
    Delete a node from an undirected graph
    """
    neighbors = ugraph[node]
    ugraph.pop(node)
    for neighbor in neighbors:
        ugraph[neighbor].remove(node)


def targeted_order(ugraph):
    """
    Compute a targeted attack order consisting
    of nodes of maximal degree

    Returns:
    A list of nodes
    """
    # copy the graph
    new_graph = copy_graph(ugraph)

    order = []
    while len(new_graph) > 0:
        max_degree = -1
        for node in new_graph:
            if len(new_graph[node]) > max_degree:
                max_degree = len(new_graph[node])
                max_degree_node = node

        neighbors = new_graph[max_degree_node]
        new_graph.pop(max_degree_node)
        for neighbor in neighbors:
            new_graph[neighbor].remove(max_degree_node)

        order.append(max_degree_node)
    return order


def load_graph(graph_url):
    """
    Function that loads a graph given the URL
    for a text representation of the graph

    Returns a dictionary that models a graph
    """
    graph_file = urllib2.urlopen(graph_url)
    graph_text = graph_file.read()
    graph_lines = graph_text.split('\n')
    graph_lines = graph_lines[ : -1]

    print "Loaded graph with", len(graph_lines), "nodes"

    answer_graph = {}
    for line in graph_lines:
        neighbors = line.split(' ')
        node = int(neighbors[0])
        answer_graph[node] = set([])
        for neighbor in neighbors[1 : -1]:
            answer_graph[node].add(int(neighbor))

    return answer_graph


def UER_graph(num_nodes, prob):
    """
    Randomly generate an undirected graph, there is an undirected edge between node i and node j with probability of p
    :param num_nodes: number of nodes (int)
    :param prob: probability that there is a edge from i to j, (float, range = [0, 1])
    :return: a random directed graph dictionary
    """
    initial_nodes = range(num_nodes)
    terminal_nodes = range(num_nodes)
    random_graph = {}
    for node in range(num_nodes):   # Initial an empty neighbor list for evey node in graph
        random_graph[node] = set([])
    for tail in initial_nodes:      # Randomly add an undirected edge
        terminal_nodes.pop(0)
        for head in terminal_nodes:
            if tail != head and random.random() <= prob:
                random_graph[tail].add(head)
                random_graph[head].add(tail)
    return random_graph


def UPA_graph(final_num, initial_num):
    """
    The algorithm starts by creating a complete undirected graph on m nodes.
    Then, the algorithm grows the graph by adding n-m nodes, where each new node is connected to
    m nodes randomly chosen from the set of existing nodes.
    :param final_num: n
    :param initial_num: m
    :return: random undirected graph by UPA algorithm
    """
    ugraph = UER_graph(initial_num, 1)      # Initial a complete undirected graph
    tail = UPATrial(initial_num)
    for node in range(initial_num, final_num):
        neighbors = tail.run_trial(initial_num)
        ugraph[node] = neighbors
        for neighbor in neighbors:
            ugraph[neighbor].add(node)
    return ugraph



# network_graph = load_graph(NETWORK_URL)
# uer_graph = UER_graph(1347, 0.0034)
# upa_graph = UPA_graph(1347, 2)

# total = 0
# for node in upa_graph:
#     total += len(upa_graph[node])
#     neighbors = upa_graph[node]
# print total
# print UER_graph(3, 0.5)