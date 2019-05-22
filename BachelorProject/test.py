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


