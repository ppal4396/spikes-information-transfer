
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import numpy as np
import sys

n_layers = 4

sig_links_mat = np.zeros((n_layers,n_layers))
with open('results/pairwise_summary.csv') as f:
    header = f.readline()
    lines = f.readlines()
i = 0
j = 0
for line in lines:
    sig_links_mat[i, j] = int(line.strip().split(',')[-1])
    j += 1
    if j == n_layers:
        j = 0
        i += 1
# index  0, 1, 2, 3
# layer 23, 4, 5, 6 


vmin = 0
vmax = np.amax(sig_links_mat)

# plot
blues = mpl.colormaps['Blues']
fig, axs = plt.subplots(1, 1, figsize=(4, 3),
                        constrained_layout=True, squeeze=False)
layer_names = ['Layer 23', 'Layer 4', 'Layer 5', 'Layer 6']
for [ax, cmap] in zip(axs.flat, [blues]):
    psm = ax.imshow(sig_links_mat, cmap=cmap, snap = True, rasterized=True, vmin=vmin, vmax=vmax)
    fig.colorbar(psm, ax=ax)
    ax.set_xticks(np.arange(n_layers), labels=layer_names)
    ax.set_yticks(np.arange(n_layers), labels=layer_names)

#add text annotations
for i in range(n_layers):
    for j in range(n_layers):
        if sig_links_mat[i,j] < 10:
            colour = 'black'
        else:
            colour = 'white'
        text = ax.text(j, i, sig_links_mat[i, j],
                       ha="center", va="center", color=colour)

#rotate x tick labels
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
plt.savefig('results/pairwise_summary.png')
print("saved in results/pairwise_summary.png")