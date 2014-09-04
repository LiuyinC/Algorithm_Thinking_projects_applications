"""
To gain a more tangible feel for how directed graphs are represented as dictionaries
in Python, you will create three specific graphs (defined as constants) and implement
a function that returns dictionaries corresponding to a simple type of directed graphs.
"""

EX_GRAPH0 = {0: set([1, 2]), 1: set([]), 2: set([])}

EX_GRAPH1 = {0: set([1, 4, 5]), 1: set([2, 6]), 2: set([3]), 3: set([0]), 4: set([1]), 5: set([2]), 6: set([])}

EX_GRAPH2 = {0: set([1, 4, 5]), 1: set([2, 6]), 2: set([3, 7]), 3: set([7]), 4: set([1]), 5: set([2]), 6: set([]),
             7: set([3]), 8: set([1, 2]), 9: set([0, 3, 4, 5, 6, 7])}


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
    Takes a directed graph digraph (represented as a dictionary) and computes the un-normalized distribution
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

EX_GRAPH4 = {'cat': set(['dog', 'm']), 'm': set(['dog','b']), 'b': set([]), 'dog': set([])}
print compute_in_degrees(EX_GRAPH4)
print in_degree_distribution(EX_GRAPH4)