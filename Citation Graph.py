"""
Provided code for Application portion of Module 1

Imports physics citation graph
"""

__author__ = 'liuyincheng'

# general imports
import urllib2
import random


# Set timeout for CodeSkulptor if necessary
#import codeskulptor
#codeskulptor.set_timeout(20)


###################################
# Code for loading citation graph

CITATION_URL = "http://storage.googleapis.com/codeskulptor-alg/alg_phys-cite.txt"

class DPATrial:
    """
    Simple class to encapsulate optimized trials for DPA algorithm

    Maintains a list of node numbers with multiple instances of each number.
    The number of instances of each node number are
    in the same proportion as the desired probabilities

    Uses random.choice() to select a node number from this list for each trial.
    """

    def __init__(self, num_nodes):
        """
        Initialize a DPATrial object corresponding to a
        complete graph with num_nodes nodes

        Note the initial list of node numbers has num_nodes copies of
        each node number
        """
        self._num_nodes = num_nodes
        self._node_numbers = [node for node in range(num_nodes) for dummy_idx in range(num_nodes)]


    def run_trial(self, num_nodes):
        """
        Conduct num_node trials using by applying random.choice()
        to the list of node numbers

        Updates the list of node numbers so that the number of instances of
        each node number is in the same ratio as the desired probabilities

        Returns:
        Set of nodes
        """

        # compute the neighbors for the newly-created node
        new_node_neighbors = set()
        for dummy_idx in range(num_nodes):
            new_node_neighbors.add(random.choice(self._node_numbers))

        # update the list of node numbers so that each node number
        # appears in the correct ratio
        self._node_numbers.append(self._num_nodes)
        self._node_numbers.extend(list(new_node_neighbors))

        #update the number of nodes
        self._num_nodes += 1
        return new_node_neighbors


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


def make_complete_graph(num_nodes):
    """
    Takes the number of nodes num_nodes and returns a dictionary corresponding to a complete directed graph
    with the specified number of nodes.
    """
    graph = {}
    if num_nodes <= 0:  # Return empty graph if num_nodes is not positive
        return graph
    nodes = []
    for node_id in range(num_nodes):
        nodes.append(node_id)
    for node_id in range(num_nodes):
        out_nodes = list(nodes)
        out_nodes.remove(node_id)
        graph[node_id] = set(out_nodes)
    return graph


def compute_in_degrees(digraph):
    """
    Takes a directed graph digraph (represented as a dictionary) and computes the in-degrees for the nodes in the graph.
    """
    in_degrees_graph = {}
    for key in digraph.keys():
        in_degrees_graph[key] = 0
    for node_id in digraph.keys():
        for out_node in digraph[node_id]:
            in_degrees_graph[out_node] += 1
    return in_degrees_graph


def in_degree_distribution(digraph):
    """
    Takes a directed graph digraph (represented as a dictionary) and computes the normalized distribution
    of the in-degrees of the graph.
    """
    in_degree_count = compute_in_degrees(digraph)
    in_degree_dist = {}
    for num in in_degree_count.values():
        if num in in_degree_dist.keys():
            in_degree_dist[num] += 1
        else:
            in_degree_dist[num] = 1
    return in_degree_dist


def in_degrees_dist_plot(in_degrees_dist, num_nodes):
    import matplotlib.pylab as plt
    x_axis = []
    y_axis = []
    for node, degree in in_degrees_dist.items():
        if node != 0:
            distribution = float(degree) / float(num_nodes)
            x_axis.append(node)
            y_axis.append(distribution)
    plt.loglog(x_axis, y_axis, 'ro')
    plt.xlabel('In-degrees')
    plt.ylabel('Distribution')
    plt.title('In degrees Distribution (log/log Plot)')
    plt.show()


def ER_random_directed_graph(num_nodes, prob):
    """
    Randomly generate a directed graph, there is a directed edge from node i to node j with probability of p
    :param num_nodes: number of nodes (int)
    :param prob: probability that there is a edge from i to j, (float, range = [0, 1])
    :return: a random directed graph dictionary
    """
    initial_nodes = range(num_nodes)
    terminal_nodes = range(num_nodes)
    random_graph = {}
    for tail in initial_nodes:
        out_nodes = set([])
        for head in terminal_nodes:
            if tail != head and random.random() <= prob:
                out_nodes.add(head)
        random_graph[tail] = out_nodes
    return random_graph


def DPA_random_directed_graph(final_num_nodes, initial_num_nodes):
    """
    The algorithm starts by creating a complete directed graph on m nodes.
    Then, the algorithm grows the graph by adding n-m nodes, where each new node is connected to
    m nodes randomly chosen from the set of existing nodes.
    :param final_num_nodes :n
    :param initial_num_nodes: m
    :return: directed graph
    """
    digraph = make_complete_graph(initial_num_nodes)
    tail = DPATrial(initial_num_nodes)
    for node in range(initial_num_nodes, final_num_nodes):
        neighbors = tail.run_trial(initial_num_nodes)
        digraph[node] = neighbors
    return digraph

# Question 1, normalized citation in-degrees distribution plotting.
# citation_graph = load_graph(CITATION_URL)
# citation_papers = 27770
# citation_in_degrees_dist = in_degree_distribution(citation_graph)
# in_degrees_dist_plot(citation_in_degrees_dist, citation_papers)

# Question 2, using ER algorithm provided in homework to generate a random directed graph
# and plot it.
# num_nodes = 1000
# random_graph = ER_random_directed_graph(num_nodes, 0.6)
# random_in_degrees_dist = in_degree_distribution(random_graph)
# in_degrees_dist_plot(random_in_degrees_dist, num_nodes)

# Question 4, implement the DPA algorithm into the graph with almost size with citation graph.
# final_nodes = 27770
# initial_nodes = 13
# DPA_digraph = DPA_random_directed_graph(final_nodes, initial_nodes)
# DPA_in_degrees_dist = in_degree_distribution(DPA_digraph)
# in_degrees_dist_plot(DPA_in_degrees_dist, final_nodes)