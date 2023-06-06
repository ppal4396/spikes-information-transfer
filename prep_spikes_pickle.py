'''Prepares pickle of all spike times for use in effective network inference script.
Saves a text file containing the cell names of each spike train in `spikes`, following the same index.'''

import glob
import numpy as np
import pickle
import sys

# ----- CONFIG -----
association = True #True if multiple probes
mice = ['mouse1probe4', 'mouse1probe8'] #single element if one probe.
# ------------------

#clear files.
if not association:
    mouse = mice[0]
    with open(f'data/{mouse}/target_indices.txt', 'w') as f:
        pass
    with open(f'data/{mouse}/spikes_LIF_{mouse}.pk', 'w') as f:
        pass
else:
    with open(f'data/{mice[0]}_{mice[1]}_target_indices.txt', 'w') as f:
        pass
    with open(f'data/spikes_LIF_{mice[0]}_{mice[1]}.pk', 'w') as f:
        pass

#read spikes
spikes = []
for mouse in mice:
    for path in glob.glob(f'data/{mouse}/mouse*.txt'):
        try:
            cell_name = 'layer_' + path.split('_')[4] + '_cell_' + path.split('_')[-1].rstrip('.txt')
        except IndexError:
            print("Index error when reading path. Stopping.")
            print(path)
            sys.exit(-1)
        if not association:
            with open(f'data/{mouse}/target_indices.txt', 'a+') as f:
                f.write(cell_name + '\n')
        else:
            with open(f'data/{mice[0]}_{mice[1]}_target_indices.txt', 'a+') as f:
                f.write(mouse + '_' + cell_name + '\n')
        numpy_array = np.loadtxt(path)
        spikes.append(numpy_array)
if not association:
    mouse = mice[0]
    with open(f'data/{mouse}/spikes_LIF_{mouse}.pk', 'wb') as f:
        pickle.dump(spikes, f)
else:
    with open(f'data/spikes_LIF_{mice[0]}_{mice[1]}.pk', 'wb') as f:
        pickle.dump(spikes, f)


