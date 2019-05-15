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
MAX_DISTANCE_BETWEEN_SATELLITES = sys.float_info.max
# todo : see if this would be better sat.MAX_DISTANCE_BETWEEN_SAT

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

#Creates for each node a bunch on the following form:
#dictionnary{node_in_bunch: distance_to_node_in_bunch, previous_node_before_reaching_this_node}
#Keeps track of the path to the next level following the same format

def bunches_sat_graph(graph, a_list):
    level = len(a_list)-1
    nodes = graph.nodes

    for node in nodes:
        distances_to_others = {node: (MAX_DISTANCE_BETWEEN_SATELLITES, None) for node in nodes}
        distances_to_others[node] = 0, None

        for neighbor in graph.neighbors(node):
            if neighbor != node:
                distances_to_others[neighbor] = graph[node][neighbor]['weight'], node

        q_distances = {node: distance[0] for node, distance in distances_to_others.items()}
        del q_distances[node]

        distance_to_levels = {lvl: (MAX_DISTANCE_BETWEEN_SATELLITES, None) for lvl in range(0, level+1)} #contains len(a_list) elements

        next_level = 0
        while len(a_list[next_level]) != 0 and graph.nodes[node]['level'] >= next_level:
            distance_to_levels[next_level] = 0, node
            next_level += 1

        bunch = set()
        bunch.add(node)

        while len(q_distances) > 0 and len(a_list[next_level]) > 0:
            u, dist_u = min(q_distances.items(), key=lambda x:x[1])

            while graph.nodes[u]['level'] >= next_level:
                distance_to_levels[next_level] = dist_u, u
                next_level += 1


            if graph.nodes[u]['level'] == next_level - 1:
                assert(distance_to_levels[next_level][0]>=MAX_DISTANCE_BETWEEN_SATELLITES)

            if graph.nodes[u]['level'] >= next_level-1:
                bunch.add(u)

            del q_distances[u]
            neighbors = list(graph.neighbors(u))
            u_neighbors = [node for node in neighbors if node in list(map(lambda x: x[0], q_distances.items()))]

            for v in u_neighbors:
                alt = dist_u + graph[u][v]['weight']
                #print(distances_to_others[v][0], alt)
                if alt < distances_to_others[v][0]:
                    #print('Im here', distances_to_others[v][0], alt)
                    distances_to_others[v] = alt, u
                    q_distances[v] = alt

        # a node adds all nodes at level max to its bunch
        bunch = list(bunch | set(a_list[next_level-1]))

        #keeps track of all the useful distances
        bunch = {node: distances_to_others[node] for node in bunch}

        graph.nodes[node]['bunch'] = bunch
        graph.nodes[node]['next_levels'] = distance_to_levels
        print('for node', node, 'the bunch is ', bunch, '\ndistance_to_levels is ', distance_to_levels)


    return


#Creates for each node a cluster on the following form:
#dictionnary{node_in_cluster: distance_to_node_in_cluster, next_step_to_reach_this_node}
#TODO : FIND ANOTHER WAY TO COMPUTE IT (or don't use it), it's really slow
#TODO : Avoid exponential time...
def clusters_from_bunches(graph):
    nodes = graph.nodes
    for node1 in nodes:
        cluster = {node2: nodes[node2]['bunch'][node1] for node2 in nodes if node1 in nodes[node2]['bunch'].keys()}
        nodes[node1]['cluster'] = cluster


def dist_u_v(graph, u, v):
    w = u
    next_level = 0
    while w not in set(graph.nodes[v]['bunch'].keys()):
        next_level += 1
        u, v = v, u
        w = graph.nodes[u]['next_levels'][next_level][1]

    return graph.nodes[u]['bunch'][w][0] + graph.nodes[v]['bunch'][w][0]


#Creates the routing table for each node (containing distances and next hop to Pi_v and to the nodes in Cluster)
def routing_tables(graph):
    nodes = graph.nodes
    for node1 in nodes:
        #Adds the cluster to the routing tables
        #TODO : Verify the next_hop computed here is the correct next_hope to follow
        routing_table = nodes[node1]['cluster']

        #Get the distance and node to the next levels, then computes the path to arrive there
        for _, dist_and_node in nodes[node1]['next_levels'].items():
            dist, node2 = dist_and_node
            if node2 != None:
                next_node = None
                after_node = node2
                bunch_u = graph.nodes[node1]['bunch']
                while after_node != node1:
                    next_node = after_node
                    after_node = bunch_u[after_node][1]
                routing_table[node2] = (dist, next_node)
        nodes[node1]['routing_table'] = routing_table

# NB : Constructs all the required parameters for the graph EXCEPT CLUSTERS AND ROUTING TABLES
def initialize(graph):
    a_list = create_a_levels(graph, NUMBER_OF_LEVELS)
    add_level_to_nodes(graph, a_list)
    bunches_sat_graph(graph, a_list)

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

    # levels
    a_list = [[0, 1, 2, 3], [1, 2, 3], [2, 3], [3], []]
    add_level_to_nodes(links_graph, a_list)

    for node in links_graph.nodes:
        print(node, 'has level ', links_graph.nodes[node]['level'])

    bunches_sat_graph(links_graph, a_list)
    clusters_from_bunches(links_graph)
    for node in links_graph.nodes:
        print(node, ' has bunch ', links_graph.nodes[node]['bunch'])
    for node in links_graph.nodes:
        print(node, ' has cluster', links_graph.nodes[node]['cluster'])
    for node in links_graph.nodes:
        print(node, ' has next levels ', links_graph.nodes[node]['next_levels'])
    for node1 in links_graph.nodes:
        for node2 in links_graph.nodes:
            print('node ', node1, ' to node ', node2, ' : ', dist_u_v(links_graph, node1, node2))
    routing_tables(links_graph)
    for node in links_graph:
        print(links_graph.nodes[node]['routing_table'])


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
    for node in triangle_graph:
        print(node, 'has next level', nodes[node]['next_level'])
    for node1 in triangle_graph.nodes:
        for node2 in triangle_graph.nodes:
            print('node ', node1, ' to node ', node2, ' : ', dist_u_v(triangle_graph, node1, node2))


def test_rnd_graph():
    triangle_graph = nx.Graph()
    triangle_graph.add_nodes_from([0, 1, 2, 3, 4, 5])
    triangle_graph.add_edge(0, 1, weight=146.86047800548656)
    triangle_graph.add_edge(0, 2, weight=122.18837915284743)
    triangle_graph.add_edge(0, 3, weight=17)
    triangle_graph.add_edge(0, 4, weight=10.63014581273465)
    triangle_graph.add_edge(0, 5, weight=124.90796611905904)
    triangle_graph.add_edge(1, 2, weight=77.3692445355388)
    triangle_graph.add_edge(1, 3, weight=161.8919392681427)
    triangle_graph.add_edge(1, 4, weight=136.4734406395618)
    triangle_graph.add_edge(1, 5, weight=41.593268686170845)
    triangle_graph.add_edge(2, 3, weight=131.24404748406687)
    triangle_graph.add_edge(2, 4, weight=112.25417586887359)
    triangle_graph.add_edge(2, 5, weight=98.08159868191383)
    triangle_graph.add_edge(3, 4, weight=25.96150997149434)
    triangle_graph.add_edge(3, 5, weight=141.4390328021229)
    triangle_graph.add_edge(4, 5, weight=115.52056094046635)
    a_list = [[0, 3, 5], [4], [1, 2], []]
    add_level_to_nodes(triangle_graph, a_list)
    bunches_sat_graph(triangle_graph, a_list)
    nodes = triangle_graph.nodes
    for node in triangle_graph:
        print(node, ' has bunch ', nodes[node]['bunch'])
    for node in triangle_graph:
        print(node, ' has next levels ', nodes[node]['next_levels'])


#This is a simulation of a small constellation. It has been constructed in LeoSatellites.py
def test_small_sat_graph():
    # level == 3
    small_graph, _ = sat.create_small_sat_graph()
    print(small_graph.edges(data=True))
    a_list = create_a_levels(small_graph, 3)
    print(a_list)
    add_level_to_nodes(small_graph, a_list)
    bunches_sat_graph(small_graph, a_list)
    #clusters_from_bunches(small_graph)
    #routing_tables(small_graph)
    nodes = small_graph.nodes
    for node in small_graph:
        print(node, ' has bunch ', nodes[node]['bunch'])
    #for node in small_graph:
        #print(node, ' has cluster', nodes[node]['cluster'])
    print(dist_u_v(small_graph, 0, 2))
    #for node in small_graph:
        #print(small_graph.nodes[node]['routing_table'])


def test_big_graph():
    graph, _ = sat.create_spaceX_graph()
    a_list = [[range(0, 5999)], [range(0, 1200)], [range(0, 400)]]
    add_level_to_nodes(graph, a_list)
    bunches_sat_graph(graph, a_list)


test_big_graph()


#-----------------------------------------------------------------------------------------------------------------------
# True computations happens here

def compact_rd(graph):
    a_list = create_a_levels(graph, NUMBER_OF_LEVELS)
    add_level_to_nodes(graph, a_list)
    bunches_sat_graph(graph, a_list)
    nodes = graph.nodes
    distances = {}
    for node1 in nodes:
        dist_for_node1 = {}
        for node2 in nodes:
            dist_for_node1[node2] = dist_u_v(graph, node1, node2)
        distances[node1] = dist_for_node1
    return distances

def plot_dij_vs_compact():
    sat_graph, distances_matrix = sat.create_small_sat_graph()
    number_of_nodes = sat_graph.number_of_nodes()

    #Puts all the direct sight values into a single dimension list
    direct_list = []
    for dim1 in distances_matrix:
        for dim2 in dim1:
            for dim3 in dim2:
                for dim4 in dim3:
                    direct_list.append(dim4)

    dij_list = []
    dij_distance = dict(nx.all_pairs_dijkstra_path_length(sat_graph))

    print(dij_distance[0])

    compact_list = []
    compact_distance = compact_rd(sat_graph)
    print(compact_distance[0])


    for node1 in range(0, number_of_nodes):
        dist_dict = dij_distance[node1]
        compact_dict = compact_distance[node1]
        for node2 in range(0, number_of_nodes):
            dij_list.append(dist_dict[node2])
            compact_list.append(compact_dict[node2])





