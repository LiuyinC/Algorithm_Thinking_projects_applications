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
from collections import deque
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


def random_order(ugraph):
    """
    Take a graph and return a list of the nodes in the graph in some random order
    """
    nodes = ugraph.keys()
    random.seed(1)      # Set a random seed to guarantee the same results
    random.shuffle(nodes)
    return nodes


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
    temp_ugraph = copy_graph(ugraph)
    original_max_cc = largest_cc_size(temp_ugraph)
    resilience = [original_max_cc]
    for node in attack_order:
        for head in temp_ugraph[node]:       # Delete all edges
            if node in temp_ugraph[head]:
                temp_ugraph[head].remove(node)
        temp_ugraph.pop(node)     # Delete the node
        max_cc = largest_cc_size(temp_ugraph)
        resilience.append(max_cc)
    return resilience


def fast_targeted_order(ugraph):
    """
    Quickly compute a targeted attack order consisting of nodes of maximal degree
    """
    temp_ugraph = copy_graph(ugraph)
    num_nodes = len(temp_ugraph)
    order = []
    degree_sets = [ set([]) for _i in range(num_nodes)]     # Initial an empty degree sets
    for node in temp_ugraph.keys():      # Create the degree_set
        degree = len(temp_ugraph[node])
        degree_sets[degree].add(node)
    for degree in range(num_nodes- 1, -1, -1):
        while len(degree_sets[degree]) != 0:
            tail = degree_sets[degree].pop()
            for head in temp_ugraph[tail]:
                head_degree = len(temp_ugraph[head])
                degree_sets[head_degree].remove(head)
                degree_sets[head_degree - 1].add(head)
                temp_ugraph[head].remove(tail)
            temp_ugraph.pop(tail)
            order.append(tail)
    return order


# test_graph = UER_graph(1000, 1)
# # test_graph = UPA_graph(1000, 5)
# time0 = time.time()
# order1 = targeted_order(test_graph)
# time1 = time.time()
# order2 = fast_targeted_order(test_graph)
# time2 = time.time()
# print "target order time = ", time1- time0
# print "faster target order time = ", time2 - time1


# network_graph = load_graph(NETWORK_URL)
# uer_graph = UER_graph(1347, 0.0034)
# upa_graph = UPA_graph(1347, 2)
#
#
"""
Question 1, plot the resilience for all these graph versus number of nodes removed.
"""
# attack_order = random_order(uer_graph)
# x_value = range(len(attack_order) + 1)
# network_res = compute_resilience(network_graph, attack_order)
# uer_res = compute_resilience(uer_graph, attack_order)
# upa_res = compute_resilience(upa_graph, attack_order)
# plt.grid()
# plt.plot(x_value, network_res, '-b', lw = 2, label = 'Network Graph')
# plt.plot(x_value, uer_res, '-r', lw = 2, label='ER Graph (p = 0.0034)')
# plt.plot(x_value, upa_res, '-g', lw = 2, label = 'UPA Graph (m = 2)')
# plt.xlabel('The number of nodes removed')
# plt.ylabel('Size of largest connected component after node removal')
# plt.title('The resilience of Network, ER and UPA Graphs')
# plt.legend()
# plt.show()

"""
Question 3, analyze the running time of these two methods on UPA graphs of size n with m=5.
"""
slow_running_times = []
fast_running_times = []
for final_num in range(10, 1000, 10):
    upa_graph = UPA_graph(final_num, 5)
    time0 = time.time()
    slow_order = targeted_order(upa_graph)
    time1 = time.time()
    fast_order = fast_targeted_order(upa_graph)
    time2 = time.time()
    slow_running_times.append(time1 - time0)
    fast_running_times.append(time2 - time1)
print 'slow', slow_running_times
print 'fast', fast_running_times
plt.plot(range(10, 1000, 10), slow_running_times, 'r', lw = 2, label = 'Provided target order generation method')
plt.plot(range(10, 1000, 10), fast_running_times, 'g', lw = 2, label = 'Fast target order generation method')
plt.legend(loc = 'upper left')
plt.xlabel('Size of UPA graph n')
plt.ylabel('Running time (unit = second)')
plt.title('Comparison of Running time \n for generating target order from UPA graphs  (Desktop Python)')
plt.grid()
plt.show()


"""
Question 4, continue the analysis of the computer network, and exam its resilience under an attack
 in which servers are chosen based on their connectivity. Then again compare the resilience of
 the computer network to that of UER and UPA random graphs of similar size.
"""
# network_order = fast_targeted_order(network_graph)
# uer_order = fast_targeted_order(uer_graph)
# upa_order = fast_targeted_order(upa_graph)
# network_res = compute_resilience(network_graph, network_order)
# uer_res = compute_resilience(uer_graph, uer_order)
# upa_res = compute_resilience(upa_graph, upa_order)
# plt.plot(network_res, '-b', lw = 2, label = 'Network Graph')
# plt.plot(uer_res, '-r', lw = 2, label='ER Graph (p = 0.0034)')
# plt.plot(upa_res, '-g', lw = 2, label = 'UPA Graph (m = 2)')
# plt.xlabel('The number of nodes removed')
# plt.ylabel('Size of largest connected component after node removal')
# plt.title('The resilience of Network, ER and UPA Graphs \n under a specific attack order')
# plt.grid()
# plt.legend()
# plt.show()


# # Compare resilience of UPA graph under different attacks
# random_attack = random_order(upa_graph)
# upa_res_random = compute_resilience(upa_graph,random_attack)
# plt.plot(upa_res_random, 'r', lw = 2, label = 'UPA under random attack')
# plt.xlabel('The number of nodes removed')
# plt.ylabel('Size of largest connected component after node removal')
# plt.title('The resilience of UPA graphs under different attack orders')
# plt.grid()
# plt.legend()
# plt.show()

# Compare resilience of computer network under vulnerable attack by different order generation method
# network_order_slow = targeted_order(network_graph)
# network_res_slow = compute_resilience(network_graph, network_order_slow)
# plt.plot(network_res_slow, '-r', lw = 2, label = "Slow attack order generator")
# plt.legend()
# plt.show()

# test_graph = UER_graph(40, 0.1)
# print test_graph
# print test_graph.keys()
# print map(len, test_graph.values())
#
# order1 = targeted_order(test_graph)
# order2 = fast_targeted_order(test_graph)
# print len(order1), len(order2)
# print 'slow:', order1
# print 'fast:', order2
# print order1==order2