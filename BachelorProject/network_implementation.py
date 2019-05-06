import networkx as nx
import matplotlib.pyplot as plt
import math
import random
import sys
import numpy as np
import copy
import LeoSatellites as sat


#This is the way to put labels in them
NUMBER_OF_NODES = 5
NUMBER_OF_LEVELS = 3
MAX_DISTANCE_BETWEEN_SATELLITES = sat.MAX_DISTANCE_BETWEEN_SATS #TODO : change it with max distance btw sats

def create_full_mesh(number_of_nodes):
    created_graph = nx.Graph()
    created_graph.add_nodes_from([i for i in range(0, number_of_nodes)])
    for i in range (0, number_of_nodes):
        for j in range (0, number_of_nodes):
            if (i != j):
                random_int = np.random.randint(low = 1, high = 2)
                created_graph.add_edge(i, j, weight=1)
    return created_graph



def create_max_links(number_of_nodes, number_of_links):
    created_graph = nx.Graph()
    created_graph.add_nodes_from([i for i in range(0, number_of_nodes)])
    number_of_neighbours = [0 for _ in range (0, number_of_nodes)]

    for node1 in range (0, number_of_nodes):
        possible_neighbours = list(created_graph.nodes())
        for link in range (0, number_of_links):
            if number_of_neighbours[node1] < number_of_links:
                node2 = random.choice(possible_neighbours)
                possible_neighbours.remove(node2)
                random_int = np.random.randint(low=1, high=5)
                created_graph.add_edge(node1, node2, weight=random_int)
                number_of_neighbours[node1] += 1
                number_of_neighbours[node2] += 1

    return created_graph




def create_a_levels(graph, k):
    nodes = graph.nodes
    a_list = [list(nodes)]
    n = graph.number_of_nodes()

    # Creates the A set of different k levels
    prob = 1.0 / math.pow(float(n), 1 / float(k))
    for i in range(1, k):
        a_i = list()
        for node in a_list[i - 1]:
            if random.uniform(0, 1) < prob:
                a_i.append(node)
        a_list.append(a_i)

    a_list.append([])

    return a_list

# USED PREVIOUSLY FOR FULL MESH GRAPHS ------------------------------------------------------------------------------------------------------
def dist_Ai_v_and_pi_v(graph, a_list):
    nodes = graph.nodes
    dist_v_a = list() # this is a matrix with vertices as lines and a_i as columns
    p_v_list = list() # this is a matrix with vertices as lines and pi(v) as columns, corresponding to shortest path to a_i
    for node1 in nodes:
        dist_node1_a_i = [0]
        p_node1_list = [node1]
        for a_i in a_list[1:]:
            min_dist = sys.float_info.max
            closest_node = None
            for node2 in a_i:
                if node1 == node2:
                    min_dist = 0
                    closest_node = node2
                elif graph[node1][node2]['weight'] < min_dist:
                    min_dist = graph[node1][node2]['weight']
                    closest_node = node2
            dist_node1_a_i.append(min_dist)
            p_node1_list.append(closest_node)
        dist_v_a.append(dist_node1_a_i)
        p_v_list.append(p_node1_list)

    return dist_v_a, p_v_list

def create_bunches(graph, a_list, dist_v_a):
    levels = len(a_list)
    nodes = graph.nodes()
    b_v_list = list()
    for crit_node in nodes:
        b_v_i = list()
        for i in range (0, levels):
            difference_set = [node for node in a_list[i] if node not in a_list[i+1]]
            for node2 in difference_set:
                if crit_node == node2 or graph[crit_node][node2]['weight'] < dist_v_a[crit_node][i+1]:
                    b_v_i.append(node2)
        b_v_list.append(b_v_i)
    return b_v_list


def dist(node1, node2, bunches, p_v_list, graph):
    if node1 < 0 or node1 >= NUMBER_OF_NODES or node2 < 0 or node2 >= NUMBER_OF_NODES:
        raise Exception("One of the nodes is out of bounds")

    next_node = node1
    i = 0
    path = []
    while next_node not in bunches[node2]:
        i = i+1
        (node1, node2) = (node2, node1)
        next_node = p_v_list[node1][i]

    if next_node != node1 and next_node != node2:
        path.append(next_node)

    if next_node == node1:
        dist_next_node_node1 = 0
    else:
        print('next_node : ', next_node, 'node1 :', node1)
        dist_next_node_node1 = graph[next_node][node1]['weight']

    if next_node == node2:
        dist_next_node_node2 = 0
    else:
        dist_next_node_node2 = graph[next_node][node2]['weight']

    path.reverse()

    return dist_next_node_node1 + dist_next_node_node2, path



def main_full_mesh():
    small_graph=create_full_mesh(NUMBER_OF_NODES)
    a_list = create_a_levels(small_graph, NUMBER_OF_LEVELS)
    (dist_v_a, p_v_list) = dist_Ai_v_and_pi_v(small_graph, a_list)
    print(a_list)
    print(dist_v_a)
    print(p_v_list)
    bunches = create_bunches(small_graph, a_list, dist_v_a)
    print(bunches)
    (distance, path) = dist(0, 5, bunches, p_v_list, small_graph)
    print(distance, " ", path)

#

#-----------------------------------------------------------------------------------------------------------------------

# USED FOR SATELLITE GRAPHS ------------------------------------------------------------------------------------------------------


def add_level_to_nodes(graph, a_list):
    for level, set in enumerate(a_list):
        for node in set:
            graph.nodes[node]['level'] = level


def bunches_sat_graph(graph, a_list):
    level = len(a_list)-1
    nodes = graph.nodes

    dij_dist = []
    for node in nodes:
        distances_to_others = [MAX_DISTANCE_BETWEEN_SATELLITES for _ in nodes]
        distances_to_others[node] = 0

        for neighbor in graph.neighbors(node):
            if neighbor != node:
                distances_to_others[neighbor] = graph[node][neighbor]['weight']

        q_distances = {node: distance for node, distance in enumerate(distances_to_others)}
        del q_distances[node]

        distance_to_levels = {lvl: MAX_DISTANCE_BETWEEN_SATELLITES for lvl in range(0, level+1)} #contains len(a_list) elements

        next_level = 0
        while len(a_list[next_level]) != 0 and graph.nodes[node]['level'] >= next_level:
            distance_to_levels[next_level] = 0
            next_level += 1


        print(node, ' has distance to levels ', distance_to_levels)

        bunch = set()
        bunch.add(node)

        while len(q_distances) > 0 and len(a_list[next_level]) > 0:
            u, dist_u = min(q_distances.items(), key=lambda x:x[1])

            while graph.nodes[u]['level'] >= next_level:
                distance_to_levels[next_level] = dist_u
                next_level += 1

            if graph.nodes[u]['level'] >= next_level-1 and distance_to_levels[next_level]>=MAX_DISTANCE_BETWEEN_SATELLITES:
                bunch.add(u)

            del q_distances[u]
            neighbors = list(graph.neighbors(u))
            u_neighbors = [node for node in neighbors if node in list(map(lambda x: x[0], q_distances.items()))]
            for v in u_neighbors:
                alt = dist_u + graph[u][v]['weight']

                if alt < distances_to_others[v]:
                    distances_to_others[v] = alt
                    q_distances[v] = alt

        # a node adds all nodes at level max to its bunch
        bunch = list(bunch | set(a_list[next_level-1]))

        graph.nodes[node]['bunch'] = bunch
        dij_dist.append({node: distance for node, distance in enumerate(distances_to_others)})

    return dij_dist


#-----------------------------------------------------------------------------------------------------------------------

# HERE ARE DIFFERENT BASIC GRAPH FONCTIONS FOR TESTS

# Node_0---(1)---Node_1----(3)---Node_2-----(6)----Node_3
def test_path_graph():
    #level == 4
    links_graph = nx.Graph()
    links_graph.add_nodes_from([0, 1, 2 ,3])
    links_graph.add_edge(0, 1, weight=1)
    links_graph.add_edge(1, 2, weight=3)
    links_graph.add_edge(2, 3, weight=6)
    print(links_graph.edges())

    a_list = [[0, 1, 2, 3], [1, 2, 3], [2, 3], [3], []]
    add_level_to_nodes(links_graph, a_list)

    for node in links_graph.nodes:
        print(node, 'has level ', links_graph.nodes[node]['level'])

    add_level_to_nodes(links_graph, a_list)
    bunches_sat_graph(links_graph, a_list)
    for node in links_graph.nodes:
        print(node, ' has bunch ', links_graph.nodes[node]['bunch'])

def test_triangle_graph():
    #level == 3
    triangle_graph = nx.Graph()
    triangle_graph.add_nodes_from([0, 1, 2])
    triangle_graph.add_edge(0, 1, weight = 1)
    triangle_graph.add_edge(1, 2, weight = 6)
    triangle_graph.add_edge(2, 0, weight=3)

    a_list = [[0, 1, 2], [1, 2], [2], []]
    add_level_to_nodes(triangle_graph, a_list)
    bunches_sat_graph(triangle_graph, a_list)
    nodes = triangle_graph.nodes
    for node in triangle_graph:
        print(node, ' has bunch ', nodes[node]['bunch'])

#This is a simulation of a small constellation. It has been constructed in LeoSatellites.py
def test_small_sat_graph():
    # level == 3
    small_graph = sat.create_small_sat_graph()
    print(small_graph.edges(data=True))
    a_list = create_a_levels(small_graph, 3)
    print(a_list)
    add_level_to_nodes(small_graph, a_list)
    bunches_sat_graph(small_graph, a_list)
    nodes = small_graph.nodes
    for node in small_graph:
        print(node, ' has bunch ', nodes[node]['bunch'])

test_small_sat_graph()



