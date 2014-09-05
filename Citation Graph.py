"""
Provided code for Application portion of Module 1

Imports physics citation graph
"""

__author__ = 'liuyincheng'

# general imports
import urllib2


# Set timeout for CodeSkulptor if necessary
#import codeskulptor
#codeskulptor.set_timeout(20)


###################################
# Code for loading citation graph

CITATION_URL = "http://storage.googleapis.com/codeskulptor-alg/alg_phys-cite.txt"
NUM_NODES = 27770

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


def in_degrees_dist_plot(in_degrees_dist):
    import matplotlib.pylab as plt
    x_axis = []
    y_axis = []
    for node, degree in in_degrees_dist.items():
        if node != 0:
            distribution = float(degree) / float(NUM_NODES)
            x_axis.append(node)
            y_axis.append(distribution)
    plt.loglog(x_axis, y_axis, 'ro')
    plt.xlabel('In-degrees')
    plt.ylabel('Distribution')
    plt.title('In degrees Distribution (log/log Plot)')
    plt.show()

citation_graph = load_graph(CITATION_URL)

# Question 1, normalized citation in-degrees distribution plotting.
citation_in_degrees_dist = in_degree_distribution(citation_graph)
in_degrees_dist_plot(citation_in_degrees_dist)

