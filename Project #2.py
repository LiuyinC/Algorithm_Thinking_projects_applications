"""
First write a Python code that implements breadth-first search, then use this function to compute
the set of connected components (CCs) of an undirected graph as well as determine the size of the
largest CC. Finally write a function that compute the resilience of a graph.
"""

__author__ = 'liuyincheng'


from collections import deque

EX_GRAPH0 = {0: set([1]), 1: set([0, 2]), 2: set([1]), 3: set([])}

EX_GRAPH1 = {0: set([1, 4, 5]), 1: set([2, 6]), 2: set([3]), 3: set([0]), 4: set([1]), 5: set([2]), 6: set([])}

EX_GRAPH2 = {0: set([1, 4, 5]), 1: set([2, 6]), 2: set([3, 7]), 3: set([7]), 4: set([1]), 5: set([2]), 6: set([]),
             7: set([3]), 8: set([1, 2]), 9: set([0, 3, 4, 5, 6, 7])}


def bfs_visited(ugraph, start_node):
    """
    Takes the undirected graph ugraph and the node start_node and returns the set consisting of
    all nodes that are visited by a breadth-first search that starts at start_node.
    """
    visited_nodes = set([start_node])
    queue = deque([start_node])
    while len(queue) != 0:
        tail = queue[0]
        queue.popleft()
        for head in ugraph[tail]:
            if head not in visited_nodes:
                visited_nodes.add(head)
                queue.append(head)
    return visited_nodes


def cc_visited(ugraph):
    """
    Takes the undirected graph ugraph and returns a list of sets, where each set consists of
    all the nodes (and nothing else) in a connected component, and there is exactly one set in
    the list for each connected component in ugraph and nothing else.
    """
    connected_components = []
    remaining_nodes = ugraph.keys()
    while len(remaining_nodes) != 0:
        tail = remaining_nodes[0]
        heads = bfs_visited(ugraph, tail)
        connected_components.append(heads)
        for node in heads:
            remaining_nodes.remove(node)
    return connected_components


def largest_cc_size(ugraph):
    """
    Takes the undirected graph ugraph and returns the size (an integer) of the largest
    connected component in ugraph.
    """
    connected_components = cc_visited(ugraph)
    if len(connected_components) == 0:
        return 0
    max_size = max(map(len, connected_components))
    return max_size


def compute_resilience(ugraph, attack_order):
    """
    Takes the undirected graph ugraph, a list of nodes attack_order and iterates through
    the nodes in attack_order. For each node in the list, the function removes the given node
    and its edges from the graph and then computes the size of the largest connected component
    for the resulting graph.
    """
    original_max_cc = largest_cc_size(ugraph)
    resilience = [original_max_cc]
    for node in attack_order:
        for head in ugraph[node]:       # Delete all edges
            if node in ugraph[head]:
                ugraph[head].remove(node)
        ugraph.pop(node)     # Delete the node
        max_cc = largest_cc_size(ugraph)
        resilience.append(max_cc)
    return resilience

print compute_resilience(EX_GRAPH0, [3, 2, 1, 0])