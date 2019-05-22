import networkx as nx
import matplotlib.pyplot as plt
import math
import random
import sys
import numpy as np
import copy
import LeoSatellites as sat
import pickle
from bokeh.plotting import figure, output_file, show


#This is the way to put labels in them
NUMBER_OF_NODES = 5
NUMBER_OF_LEVELS = 3
MAX_DISTANCE_BETWEEN_SATELLITES = sys.float_info.max
MAX_HOPS = sys.float_info.max
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

#TODO : Try to use a heap
#
def bunches_and_clusters(graph, a_list):
    level = len(a_list)-1
    nodes = graph.nodes

    #Calculate the level where a_list[level] is empty. One should never reach this level
    max_level = 0
    while len(a_list[max_level]) > 0:
        max_level+=1

    LAST_LEVEL_NODES = set(a_list[max_level-1])
    for node in nodes:
        nodes[node]['bunch'] = {}
        nodes[node]['cluster'] = {}

    for node in nodes:
        last_level_nodes = copy.deepcopy(LAST_LEVEL_NODES)

        # this will contain : the distance to the other elements, the routing node.
        # the last node before arrival and the length of the path to arrive there
        distances_to_others = {node: (MAX_DISTANCE_BETWEEN_SATELLITES, None, None, MAX_HOPS) for node in nodes}
        distances_to_others[node] = 0, node, node, 0

        for neighbor in graph.neighbors(node):
            if neighbor != node:
                distances_to_others[neighbor] = graph[node][neighbor]['weight'], neighbor, node, 1

        q_distances = {node: distance[0] for node, distance in distances_to_others.items()}
        del q_distances[node]

        #this list constains : the distance to other levels and the last node to arrive there (todo: is the second param useful?)
        distance_to_levels = {lvl: (MAX_DISTANCE_BETWEEN_SATELLITES, None) for lvl in range(0, level+1)}

        next_level = 0
        while next_level < max_level and graph.nodes[node]['level'] >= next_level:
            distance_to_levels[next_level] = 0, node
            next_level += 1

        bunch = set()
        bunch.add(node)
        last_level_nodes.discard(node)
        while (len(q_distances) > 0 and next_level < max_level) or len(last_level_nodes) > 0:
            u, dist_u = min(q_distances.items(), key=lambda x:x[1])

            while graph.nodes[u]['level'] >= next_level:
                distance_to_levels[next_level] = dist_u, u
                next_level += 1

            if graph.nodes[u]['level'] == next_level - 1:
                assert(distance_to_levels[next_level][0]>=MAX_DISTANCE_BETWEEN_SATELLITES)

            if graph.nodes[u]['level'] >= next_level-1:
                bunch.add(u)

            if u in last_level_nodes:
                bunch.add(u)
                last_level_nodes.discard(u)

            del q_distances[u]
            neighbors = list(graph.neighbors(u))
            u_neighbors = [node for node in neighbors if node in list(map(lambda x: x[0], q_distances.items()))]

            for v in u_neighbors:
                alt = dist_u + graph[u][v]['weight']
                current_distance = distances_to_others[v][0]
                if alt < current_distance:
                    _, routing_node, _, hops_to_u, = distances_to_others[u]
                    distances_to_others[v] = alt, routing_node, u, hops_to_u+1
                    q_distances[v] = alt


        #keeps track of all the useful distances
        for node2 in bunch:
            dist, next_node, last_node, hops = distances_to_others[node2]
            graph.nodes[node]['bunch'][node2] = dist, next_node, hops
            graph.nodes[node2]['cluster'][node] = dist, last_node, hops

        graph.nodes[node]['next_levels'] = distance_to_levels
        #print('for node', node, 'the bunch is ', bunch, '\ndistance_to_levels is ', distance_to_levels)


    return


#Creates for each node a cluster on the following form:
#dictionnary{node_in_cluster: distance_to_node_in_cluster, next_step_to_reach_this_node}
#TODO : Verify this work or drop clusters
def clusters_from_bunches(graph):
    nodes = graph.nodes
    for node in nodes :
        nodes[node]['cluster'] = []
    for node1 in nodes:
        for node2, value in nodes[node1]['bunch'].items():
            nodes[node2]['cluster'].append(node1)
    return

def dist_u_v(graph, u, v):
    w = u
    next_level = 0
    while w not in set(graph.nodes[v]['bunch'].keys()):
        next_level += 1
        u, v = v, u
        w = graph.nodes[u]['next_levels'][next_level][1]

    return graph.nodes[u]['bunch'][w][0] + graph.nodes[v]['bunch'][w][0]

#TODO : add routing here
def dist_with_routing(graph, u, v):

    w = u
    next_level = 0
    while w not in set(graph.nodes[v]['bunch'].keys()):
        next_level += 1
        u, v = v, u
        w = graph.nodes[u]['next_levels'][next_level][1]

    dist_u_w, _, hops_u_w = graph.nodes[u]['bunch'][w]
    dist_v_w, _, hops_v_w = graph.nodes[v]['bunch'][w]

    return dist_u_w + dist_v_w, _, hops_u_w + hops_v_w



# NB : Constructs all the required parameters for the graph EXCEPT CLUSTERS AND ROUTING TABLES
def initialize(graph):
    a_list = create_a_levels(graph, NUMBER_OF_LEVELS)
    add_level_to_nodes(graph, a_list)
    bunches_and_clusters(graph, a_list)

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

    bunches_and_clusters(links_graph, a_list)
    for node in links_graph.nodes:
        print(node, ' has bunch ', links_graph.nodes[node]['bunch'])
    for node in links_graph.nodes:
        print(node, ' has next levels ', links_graph.nodes[node]['next_levels'])
    for node in links_graph.nodes:
        print(node, ' has cluster', links_graph.nodes[node]['cluster'])

    '''for node1 in links_graph:
        for node2 in links_graph:
            print(node1, 'to', node2, dist_with_routing(links_graph, node1, node2))'''


def test_triangle_graph():
    #level == 3
    triangle_graph = nx.Graph()
    triangle_graph.add_nodes_from([0, 1, 2])
    triangle_graph.add_edge(0, 1, weight = 1)
    triangle_graph.add_edge(1, 2, weight = 6)
    triangle_graph.add_edge(2, 0, weight=3)

    a_list = [[0, 1, 2], [1, 2], [2], []]
    add_level_to_nodes(triangle_graph, a_list)
    bunches_and_clusters(triangle_graph, a_list)
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
    bunches_and_clusters(triangle_graph, a_list)
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
    bunches_and_clusters(small_graph, a_list)
    nodes = small_graph.nodes
    for node in small_graph:
        print(node, ' has bunch ', nodes[node]['bunch'])
    for node in small_graph:
        print(node, ' has cluster', nodes[node]['cluster'])
    print(dist_u_v(small_graph, 0, 2))
    for node in small_graph:
        print(small_graph.nodes[node]['routing_table'])


def test_big_graph():
    graph, _ = sat.create_spaceX_graph()
    print(graph.number_of_nodes())
    #a_list = [list(range(0, 1600)), list(range(0, 800)), list(range(0, 400)), []]
    a_list = create_a_levels(graph, NUMBER_OF_LEVELS)
    print(a_list)
    add_level_to_nodes(graph, a_list)
    bunches_and_clusters(graph, a_list)
    for node in graph.nodes:
        for _, dist_and_node in graph.nodes[node]['bunch'].items():
            dist1, _ = dist_and_node



#-----------------------------------------------------------------------------------------------------------------------
# True computations happens here

def compact_rd(graph):
    a_list = create_a_levels(graph, NUMBER_OF_LEVELS)
    add_level_to_nodes(graph, a_list)
    bunches_and_clusters(graph, a_list)
    nodes = graph.nodes
    distances = []
    for node1 in nodes:
        for node2 in nodes:
            distances.append(dist_u_v(graph, node1, node2))

    return distances


#Computes all the datas needed and stores them into a pickle file
def dij_vs_compact():
    sat_graph, distances_matrix = sat.create_spaceX_graph()
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

    compact_list = compact_rd(sat_graph)

    bound_list = []

    for node1 in range(0, number_of_nodes):
        dist_dict = dij_distance[node1]
        for node2 in range(0, number_of_nodes):
            dij_list.append(dist_dict[node2])
            bound_list.append((2*NUMBER_OF_LEVELS-1)*dist_dict[node2])

    pickle_compact = open('compact_list.pickle', 'ab')
    pickle.dump((direct_list, dij_list, compact_list, bound_list), pickle_compact)

    print(direct_list)
    print(dij_list)
    print(compact_list)
    print(bound_list)

def plot_lines():
    pickle_in = open('compact_list.pickle', "rb")
    direct_list, dij_list, compact_list, bound_list = pickle.load(pickle_in)
    print(bound_list)
    # Draw point based on above x, y axis values.
    plt.plot(direct_list, dij_list, 'bs', direct_list, compact_list, 'g^', direct_list, bound_list, 'r--')

    # Set chart title.
    plt.title("Distances for different routing algorithms")

    # Set x, y label text.
    plt.xlabel('Line of sight')
    plt.ylabel('Bound - red \n Dijkstra dist - blue \n Compact dist - green')
    plt.show()

def plot_multiple():
    pickle_in = open('compact_list.pickle', "rb")
    direct_list, dij_list, compact_list, bound_list = pickle.load(pickle_in)

    direct_list = direct_list[:1000]
    dij_list = dij_list[:1000]
    compact_list = compact_list[:1000]
    bound_list = bound_list[:1000]

    #direct_list = [1, 2, 3, 4, 5]
    #compact_list = [6, 7, 2, 4, 5]

    # output to static HTML file
    output_file("line.html")

    p = figure(plot_width=800, plot_height=800, title="Algorithms depending on the line of sight")

    # add a circle renderer with a size, color, and alpha
    p.circle(direct_list, dij_list, size=5, color="navy", alpha=0.5)
    p.circle(direct_list, compact_list, size=5, color="orange", alpha=0.5)
    p.circle(direct_list, bound_list, size=5, color="red", alpha=0.5)

    # show the results
    show(p)



plot_multiple()








