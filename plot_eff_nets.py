import networkx as nx
import glob
import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys
from math import sqrt

def layer_name(target_label):
    return 'Layer ' + (target_label.split('_')[1])

mouse = 'mouse2probe8'

#target_labels contains layer and cell number. index is same index as used during inference.
target_labels = None
with open(f'data/{mouse}/target_indices.txt') as f:
    target_labels = [label.strip() for label in f.readlines()]

#for drawing
network = nx.DiGraph()
nodes = {'Layer 23':[], 'Layer 4':[], 'Layer 5':[], 'Layer 6':[]}
node_sizes = [25] * len(target_labels) #will be dependent on num parents
node_colours = [0] * len(target_labels) #will be dependent on num targets
curved_edges = [] #edges within layers should be drawn curved for visibility
straight_edges = [] #edges within

#generate graph nodes
for target_idx, target_label in enumerate(target_labels):
    layer = layer_name(target_label)
    nodes[layer].append(target_idx)
for layer in nodes.keys():
    network.add_nodes_from(nodes[layer], layer=layer)

#read through pickled results and generate graph edges
for path in glob.glob(f'results/{mouse}/effective_inference/*.pk'):
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
    node_sizes[target_index] = 25 + len(cond_set) * 50 #drawn node size dependent on number of sources.
    for parent_index in cond_set.keys():
        parent_index = int(parent_index) #numpy int originally
        node_colours[parent_index] += 1 #drawn node size dependent on number of targets.
        parent_layer = layer_name(target_labels[parent_index])
        if layer == parent_layer:
            curved_edges.append((parent_index, target_index))
        else:
            straight_edges.append((parent_index, target_index))
        network.add_edge(parent_index, target_index)

#hacky way to ensure colours and sizes are given to the right nodes when drawing.
# this is because parameters `node_color` and `node_size` to `draw_networkx_nodes` 
# don't have a mapping to nodes, they're just lists of colours/sizes. 
# They're applied in the order of nodes in the graph that the nodes were added in.
node_colours = [node_colours[v] for v in network.nodes]
node_sizes = [node_sizes[v] for v in network.nodes]

pos = nx.multipartite_layout(network, subset_key='layer')
plt.figure(figsize=(8, 8))
ax = plt.axes()
ax.set_xticks(np.arange(-0.2, 0.2, 0.1), labels=[i for i in nodes.keys()])
ax.set_yticks([])

nx.draw_networkx_edges(network, pos, edgelist=straight_edges, alpha=0.2, node_size=node_sizes)
nx.draw_networkx_edges(network, pos, edgelist=curved_edges, connectionstyle="arc3,rad=0.2", alpha=0.2, node_size=node_sizes)
plt_pathcollection = nx.draw_networkx_nodes(network, pos, node_color=node_colours, node_size=node_sizes, cmap=plt.cm.cool)

plt.xlim(-0.2, 0.2)
plt.ylim(-1.2, 1.2)
plt.colorbar(plt_pathcollection)
plt.axis('off')
# Make legend
for n in range(min(node_sizes),max(node_sizes)+1, 150):
    plt.plot([], [], 'bo', markersize = sqrt(n), label = f"{int((n-25) / 50)}")
plt.legend(labelspacing = 5, loc='center', bbox_to_anchor=(1, 0.5), frameon = False)

plt.savefig(f'results/{mouse}/effective_inference/effective_network.png')
