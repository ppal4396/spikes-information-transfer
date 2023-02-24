
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import numpy as np
import sys
import glob

mouse = 'mouse1probe3'
n_layers = 4

sig_links_mat = np.zeros((n_layers,n_layers))
with open(f'results/{mouse}/pairwise_summary.csv') as f:
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

n_links_mat = np.zeros((n_layers, n_layers))
for filename in glob.glob(f'results/{mouse}/Layer*.csv'):
    source = filename.split('Layer ')[1][:2]
    if source != '23':
        source_idx = int(source[:1]) - 3
    else:
        source_idx = 0
    dest = filename.split('Layer ')[2][:2]
    if dest != '23':
        dest_idx = int(dest[:1]) - 3
    else:
        dest_idx = 0
    with open(filename) as f:
        n_links = len(f.readlines())
    n_links_mat[source_idx, dest_idx] = n_links


# plot number of links
#left axis is source, bottom axis is destination.
def plot_matrix_colour_map(mat, fname):
    vmin = 0
    vmax = np.amax(mat)
    blues = mpl.colormaps['Blues']
    fig, axs = plt.subplots(1, 1, figsize=(4, 3),
                            constrained_layout=True, squeeze=False)
    layer_names = ['Layer 23', 'Layer 4', 'Layer 5', 'Layer 6']
    for [ax, cmap] in zip(axs.flat, [blues]):
        psm = ax.imshow(mat, cmap=cmap, snap = True, rasterized=True, vmin=vmin, vmax=vmax)
        fig.colorbar(psm, ax=ax)
        ax.set_xticks(np.arange(n_layers), labels=layer_names)
        ax.set_yticks(np.arange(n_layers), labels=layer_names)

    #add text annotations
    for i in range(n_layers):
        for j in range(n_layers):
            if mat[i,j] < 0.33 * vmax:
                colour = 'black'
            else:
                colour = 'white'
            text = ax.text(j, i, mat[i, j],
                        ha="center", va="center", color=colour)

    #rotate x tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")
    plt.savefig(f'results/{mouse}/{fname}.png')
    print(f"saved in results/{mouse}/{fname}.png")

plot_matrix_colour_map(n_links_mat, 'n_links')

plot_matrix_colour_map(sig_links_mat, 'n_sig_links')

#plot proportion of significant links
sig_links_mat_normalised  = np.zeros((n_layers, n_layers))
i = 0
while i < n_layers:
    j = 0
    while j < n_layers:
        sig_links_mat_normalised[i,j] = round(sig_links_mat[i,j] / n_links_mat[i,j], 2)
        j += 1
    i += 1

print(sig_links_mat)
print(n_links_mat)
print(sig_links_mat_normalised)

plot_matrix_colour_map(sig_links_mat_normalised, 'pairwise_summary')
