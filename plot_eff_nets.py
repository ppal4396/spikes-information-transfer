import networkx as nx
import glob
import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys
from math import sqrt
from plot_results import plot_matrix_colour_map

# ---- config -----
mouse = 'mouse2probe8'
repeat_num = 0
# -----------------

def layer_name(target_label):
    return 'Layer ' + (target_label.split('_')[1])

def get_target_labels(mouse):
    '''returns list of strings. each string is a label containing layer and cell number.
    the index of these labels is the same index in `spikes` used during inference.'''
    target_labels = None
    with open(f'data/{mouse}/target_indices.txt') as f:
        target_labels = [label.strip() for label in f.readlines()]
    return target_labels

def draw_whole_network(nodes, edge_info):
    '''Draws effective network containing every node.'''
    network = nx.DiGraph()
    node_sizes = None #will be dependent on num parents
    node_colours = None #will be dependent on num targets

    for layer in nodes.keys():
        network.add_nodes_from(nodes[layer], layer=layer)

    node_sizes = [25+(d*50) for d in edge_info['in_degrees']]
    node_colours = edge_info['out_degrees'] #pyplot colormap will normalise.

    #hacky way to ensure colours and sizes are given to the right nodes when drawing.
    # this is because parameters `node_color` and `node_size` to `draw_networkx_nodes` 
    # don't have a mapping to nodes, they're just lists of colours/sizes. 
    # They're applied in the order of nodes in the graph that the nodes were added in.
    node_colours = [node_colours[v] for v in network.nodes]
    node_sizes = [node_sizes[v] for v in network.nodes]

    for parent_index, target_index in edge_info['edges']:
        network.add_edge(parent_index, target_index)
    
    pos = nx.multipartite_layout(network, subset_key='layer')
    plt.figure(figsize=(8, 8))

    #draw cross layer edges straight and in layer edges curved.
    nx.draw_networkx_edges(network, pos, edgelist=edge_info['cross_layer_edges'], alpha=0.2, node_size=node_sizes)
    nx.draw_networkx_edges(network, pos, edgelist=edge_info['in_layer_edges'], connectionstyle="arc3,rad=0.2", alpha=0.2, node_size=node_sizes)
    colour_mappable = nx.draw_networkx_nodes(network, pos, node_color=node_colours, node_size=node_sizes, cmap=plt.cm.cool)

    plt.xlim(-0.2, 0.2)
    plt.ylim(-1.2, 1.2)
    plt.axis('off')

    plt.colorbar(colour_mappable)

    # Make node size legend
    for n in range(min(node_sizes),max(node_sizes)+1, 150):
        plt.plot([], [], 'bo', markersize = sqrt(n), label = f"{int((n-25) / 50)}")
    plt.legend(labelspacing = 5, loc='center', bbox_to_anchor=(1, 0.5), frameon = False)

    savepath = f'results/{mouse}/effective_inference/effective_network_{repeat_num}.png'
    plt.savefig(savepath)
    print(f"network drawn in {savepath}")


def layer_dictionary(target_labels):
    '''Returns nodes inside dictionary where keys are layer names. 
    A node is an int containing the node's index as used during inference.'''
    nodes = {'Layer 23':[], 'Layer 4':[], 'Layer 5':[], 'Layer 6':[]}
    for target_idx, target_label in enumerate(target_labels):
        layer = layer_name(target_label)
        nodes[layer].append(target_idx)
    return nodes

def edges_and_degrees(target_labels, mouse=mouse, repeat_num=repeat_num):
    '''Returns dictionary of edges in the whole effective network and other helpful information,
    from pickled results.'''
    ret_dict = {
        'edges' : [],
        'cross_layer_edges' : [], #will be drawn straight
        'in_layer_edges' : [], #will be drawn curved,
        'in_degrees' : [0] * len(target_labels),
        'out_degrees' : [0] * len(target_labels)
    }

    #read through pickled results and generate graph edges and degrees
    for path in glob.glob(f'results/{mouse}/effective_inference/repeat_{repeat_num}/*.pk'):
        cond_set = None
        # surrogate_vals_at_each_round = None
        # TE_vals_at_each_round = None
        with open(path, 'rb') as f:
            cond_set = pickle.load(f)
            # surrogate_vals_at_each_round = pickle.load(f)
            # TE_vals_at_each_round = pickle.load(f)
        
        if cond_set is None:
            print("WARNING: Error reading conditional set from pickle file.")
            print(f"file: {path}")
            continue
        if len(cond_set) == 0:
            # print("Empty cond set.")
            # print(f"file: {path}")
            continue

        target_index = int(path.split('_')[-1].rstrip('.pk'))
        layer = layer_name(target_labels[target_index])
        ret_dict['in_degrees'][target_index] = len(cond_set)
        for parent_index in cond_set.keys():
            parent_index = int(parent_index) #numpy int originally
            ret_dict['out_degrees'][parent_index] += 1

            parent_layer = layer_name(target_labels[parent_index])
            if layer == parent_layer:
                ret_dict['in_layer_edges'].append((parent_index, target_index))
            else:
                ret_dict['cross_layer_edges'].append((parent_index, target_index))

            ret_dict['edges'].append((parent_index, target_index))

    return ret_dict

def edge_count_matrix(nodes, edge_info, target_labels):
    '''Returns matrix containing directed edges in the network
    Layer 23
    Layer  4
    Layer  5
    Layer  6
       Layer 23  4   5   6'''
    mat = np.zeros((4, 4))

    for source_index, target_index in edge_info['edges']:
        
        source_layer = layer_name(target_labels[source_index])
        target_layer = layer_name(target_labels[target_index])

        mat[ layer_name_to_matrix_index(source_layer, nodes) , layer_name_to_matrix_index(target_layer, nodes)] += 1
    
    return mat

def proportion_matrix(nodes, edge_count_mat):
    '''Returns matrix containing proportion of significant edges to possible edges across layers
    Layer 23
    Layer  4
    Layer  5
    Layer  6
       Layer 23  4   5   6'''
    mat = np.zeros((4, 4))
    
    for layer_a in nodes.keys():
        for layer_b in nodes.keys():
            n_possible_edges = len(nodes[layer_a]) * len(nodes[layer_b]) * 2
            mat[ layer_name_to_matrix_index(layer_a, nodes), layer_name_to_matrix_index(layer_b, nodes)] = n_possible_edges

    mat =  (edge_count_mat / mat).round(decimals=2)

    return mat

def layer_name_to_matrix_index(layer_name, nodes):
    mapping = {}
    for i, layer in enumerate(nodes.keys()):
        mapping[layer] = i
    return mapping[layer_name]

def main():

    target_labels = get_target_labels(mouse) 
    nodes = layer_dictionary(target_labels)
    edge_info = edges_and_degrees(target_labels)
    draw_whole_network(nodes, edge_info)
    
    n_edges_mat = edge_count_matrix(nodes, edge_info, target_labels)
    plot_matrix_colour_map(n_edges_mat, f'results/{mouse}/effective_inference/n_edges_{repeat_num}', n_edges_mat.shape[0])

    prop_edges_mat = proportion_matrix(nodes, n_edges_mat)
    plot_matrix_colour_map(prop_edges_mat, f'results/{mouse}/effective_inference/proportion_edges_{repeat_num}', n_edges_mat.shape[0])

if __name__ == '__main__':
    main()