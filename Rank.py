# -*- coding: UTF-8 -*-

import networkx as nx
import numpy as np
import random
from collections import Counter


def edge_rank(graph):
    '''
    :return: edge selected ranked by edge betweenness centrality
    '''
    d = nx.edge_betweenness_centrality(graph)
    rank_list = Counter(d).most_common()
    edge_select = rank_list[0][0]
    return edge_select


def k_shell_rank(graph):
    '''
    :param graph:
    :return: the selection node
    '''
    nodes = []
    node_core_num_dict = nx.core_number(graph)
    max_core_k = max(node_core_num_dict.values())
    for v in node_core_num_dict:
        if node_core_num_dict[v] >= max_core_k:
            nodes.append(v)
    return random.choice(nodes)



def leader_rank(graph):
    """
    :param graph:
    :return: 
    """
    num_nodes = graph.number_of_nodes()
    nodes = graph.nodes()
    graph.add_node('g')
    for node in nodes:
        graph.add_edge('g', node)
    LR = dict.fromkeys(nodes, 1.0)
    LR['g'] = 0.0
    while True:
        tempLR = {}
        for node1 in graph.nodes():
            s = 0.0
            for node2 in graph.nodes():
                if node2 in graph.neighbors(node1):
                    s += 1.0 / graph.degree([node2])[node2] * LR[node2]
            tempLR[node1] = s
        error = 0.0
        for n in tempLR.keys():
            error += abs(tempLR[n] - LR[n])
        if error == 0.0:
            break
        LR = tempLR
		
    avg = LR['g'] / num_nodes
    LR.pop('g')
    for k in LR.keys():
        LR[k] += avg
    graph.remove_node('g')
    print('graph.edge:', graph.edges())
    # print('LR:', sorted(LR.items(), key=lambda item: item[1]))
    return sorted(LR.items(), key=lambda item: item[1])[-1][0]
