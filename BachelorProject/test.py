def inversion(x):
    x = not(x)

x = True
inversion(x)
print(x)

#draft function
def dist_u_v(graph, u, v):
    basic_u = u
    w = u
    next_level = 0

    is_in_bunch = True
    while w not in set(graph.nodes[v]['bunch'].keys()):
        next_level += 1
        u, v = v, u
        w = graph.nodes[u]['next_levels'][next_level][1]
        is_in_bunch = False

    next_node = None
    if not(is_in_bunch):
        after_node = w
        bunch_u = graph.nodes[basic_u]['bunch']
        while after_node != basic_u:
            next_node = after_node
            after_node = bunch_u[after_node][1]
    else:
        next_node = graph.nodes[v]['bunch'][w][1]

    return graph.nodes[u]['bunch'][w][0] + graph.nodes[v]['bunch'][w][0], next_node


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

            # TODO sanity check: graph.nodes[u]['level'] == next_level-1
            # assert(graph.nodes[u]['level'] == next_level-1)

            if graph.nodes[u]['level'] == next_level - 1:
                assert(distance_to_levels[next_level][0]>=MAX_DISTANCE_BETWEEN_SATELLITES)

            if graph.nodes[u]['level'] >= next_level-1 and distance_to_levels[next_level][0] >= \
                    MAX_DISTANCE_BETWEEN_SATELLITES:
                bunch.add(u)

            del q_distances[u]
            neighbors = list(graph.neighbors(u))
            u_neighbors = [node for node in neighbors if node in list(map(lambda x: x[0], q_distances.items()))]

            for v in u_neighbors:
                alt = dist_u + graph[u][v]['weight']
                if alt < distances_to_others[v][0]:
                    distances_to_others[v] = alt, u
                    q_distances[v] = alt

        # a node adds all nodes at level max to its bunch
        bunch = list(bunch | set(a_list[next_level-1]))

        #keeps track of all the useful distances
        bunch = {node: distances_to_others[node] for node in bunch}

        graph.nodes[node]['bunch'] = bunch
        graph.nodes[node]['next_levels'] = distance_to_levels

    return


TODO : add only the nodes we want
def routing_tables(graph):
    nodes = graph.nodes
    for node1 in nodes:
        routing_node_1 = {}
        #First we add to the routing table all the elements in the cluster
        for node2 in nodes[node1]['cluster']:
            routing_node_1[node2] = dist_with_routing(graph, node1, node2)
        #Then we add to it all the elements leading to the next level
        for _, dist_and_node in nodes[node1]['next_levels'].items():
            _, node2 = dist_and_node
            if node2 is not None:
                routing_node_1[node2] = nodes[node1]['bunch'][node2]
        nodes[node1]['routing_table'] = routing_node_1


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
        #TODO: USE HEAP QUEUE instead of that!
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
            #in case it doesn't work: add list() here
            u_neighbors = [node for node in neighbors if node in q_distances.items()]

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
