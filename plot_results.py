
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import numpy as np
import sys
import glob

mouse = 'mouse2probe6'
n_layers = 4

#TODO: show in plots when no cells were recorded in the first place in certain layers.

def plot_matrix_colour_map(mat, fname):
    ''' plot matrices as colour map. left axis is source, bottom axis is dest.
    index  0, 1, 2, 3
    layer 23, 4, 5, 6'''
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

# count number of sig links
sig_links_mat = np.zeros((n_layers,n_layers))
with open(f'results/{mouse}/pairwise_summary.csv') as f:
    # header = f.readline()
    lines = f.readlines()
i = 0
j = 0
for line in lines:
    sig_links_mat[i, j] = int(line.strip().split(',')[4])
    j += 1
    if j == n_layers:
        j = 0
        i += 1

# count number of links and average TE rate + average TE per source spike
# + average TE per dest spike.
# note: calculate averages after zero'ing non-sig TEs

n_links_mat = np.zeros((n_layers, n_layers))
avg_te_rate_mat = np.zeros((n_layers, n_layers))
avg_te_per_source_spike_mat = np.zeros((n_layers, n_layers))
avg_te_per_dest_spike_mat = np.zeros((n_layers, n_layers))


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
        lines = f.readlines()
        n_links = len(lines)
    n_links_mat[source_idx, dest_idx] = n_links

    for line_no, line in enumerate(lines):
        te = float(line.split(',')[3]) #corrected te, i.e. after minusing surrogate mean
        te_per_src_spk = float(line.split(',')[7])
        te_per_dest_spk = float(line.split(',')[9])
        sig = float(line.split(',')[4].strip())
        if sig < 0.05 and te < 0:
            print("Negative 'significant' transfer found.")
            print(filename, f"line {line_no+1}")
            print(line, '\n')
        if sig > 0.05 or te < 0:
            te = 0
            te_per_src_spk = 0
            te_per_dest_spk = 0
        avg_te_rate_mat[source_idx, dest_idx] += te
        avg_te_per_source_spike_mat[source_idx, dest_idx] += te_per_src_spk
        avg_te_per_dest_spike_mat[source_idx, dest_idx] += te_per_dest_spk

i = 0
while i < n_layers:
    j = 0
    while j < n_layers:
        avg_te_rate_mat[i,j] = round(avg_te_rate_mat[i,j] / n_links_mat[i,j], 2)
        
        #note: for per spike, only using sig links in average.
        if not sig_links_mat[i,j]:
            avg_te_per_source_spike_mat[i,j] = 0
            avg_te_per_dest_spike_mat[i,j] = 0
        else:
            avg_te_per_source_spike_mat[i,j] = round(
                avg_te_per_source_spike_mat[i,j] / sig_links_mat[i,j], 2)

            #note: for per spike, only using sig links in average.
            avg_te_per_dest_spike_mat[i,j] = round(
                avg_te_per_dest_spike_mat[i,j] / sig_links_mat[i,j], 2)
        j+=1
    i+=1

#proportion of significant links
sig_links_mat_normalised  = np.zeros((n_layers, n_layers))
i = 0
while i < n_layers:
    j = 0
    while j < n_layers:
        sig_links_mat_normalised[i,j] = round(sig_links_mat[i,j] / n_links_mat[i,j], 2)
        j += 1
    i += 1

plot_matrix_colour_map(n_links_mat, 'n_links')
plot_matrix_colour_map(sig_links_mat, 'n_sig_links')
plot_matrix_colour_map(sig_links_mat_normalised, 'proportion_sig_links')
plot_matrix_colour_map(avg_te_rate_mat, 'avg_te_rate')
plot_matrix_colour_map(avg_te_per_source_spike_mat, 'avg_te_per_source')
plot_matrix_colour_map(avg_te_per_dest_spike_mat, 'avg_te_per_dest')


#