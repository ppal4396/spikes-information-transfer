'''Transfer entropy (TE) calculation on spike train data using the 
continuous-time TE estimator'''

# Reference: https://github.com/jlizier/jidt

from jpype import *
import random
import math
import os
import numpy as np
import json
import sys
import random
import glob
# ================================ config
NUM_REPS = 2
NUM_SPIKES = int(3e3)
NUM_OBSERVATIONS = 2
NUM_SURROGATES = 10
jar_location = "/Users/preethompal/Documents/USYD/honours/jidt/infodynamics.jar"

# ============================== library
def read_spike_times(path):
    '''read spike times from file given in <path>, return numpy array'''
    spike_times = []
    with open(path, 'r') as f:
        spike_times = np.asarray([float(x.strip()) for x in f.readlines()])
    return spike_times

def paths_to_data(probename='mouse2probe8'):
    '''returns dictionary, keys are layer names, values are lists of file paths 
    to data from each cell in that layer'''
    data_paths = {
        'Layer 2,3' : [],
        'Layer 4' : [], 
        'Layer 5' : [], 
        'Layer 6' : []
        } 
    for cell in glob.glob(f"data/{probename}/*.txt"):
        if 'layer_23' in cell:
            data_paths['Layer 2,3'].append(cell)
        elif 'layer_4' in cell:
            data_paths['Layer 4'].append(cell)
        elif 'layer_5' in cell:
            data_paths['Layer 5'].append(cell)
        elif 'layer_6' in cell:
            data_paths['Layer 6'].append(cell)        
    return data_paths

def nice_cell_name(path):
    ''' returns pretty string from file path to data for a cell'''
    return f"Cell {path.split('_')[-1].split('.')[0]}"

# ============================ main
def main():
    # Start the JVM (add the "-Xmx" option with say 1024M if you get crashes due
     #to not enough memory space)
    startJVM('/usr/local/opt/openjdk/bin/java', 
             "-ea", 
             "-Djava.class.path=" + jar_location)
    package = "infodynamics.measures.spiking.integration"
    TECalculator = JPackage(package).TransferEntropyCalculatorSpikingIntegration
    te_calculator = TECalculator()

    # Number of nearest neighbours to search for in the full joint space
        #when estimating probability densities
    te_calculator.setProperty("knns", "4")

    # Property name for what type of norm to use between data points for each 
    #marginal variable
        # used to quickly search for nearest neighbours across multidimensional 
        # variables
    te_calculator.setProperty("NORM_TYPE", "MAX_NORM") 

    # The past destination interspike intervals to consider (and associated 
    # property name and convenience length variable) The code assumes that the 
    # first interval is numbered 1, the next is numbered 2, etc
        # property 'destPastIntervals' is initialised to {1, 2}
        # I think:
        # 1 and 2 respectively label interval from observation point to 
        # target spike and an earlier observation point to target spike.
        # i.e. we're using two intervals at a time when generating pdf of target
        # spike occurring given target history?
    te_calculator.setProperty("DEST_PAST_INTERVALS", "1,2")

    # as previous, but for source
    te_calculator.setProperty("SOURCE_PAST_INTERVALS", "1,2")

    # You can use te_calculator.appendConditionalIntervals() for the above but
    #for conditioned processes. 
        # so I don't do this for pairwise TE, right?

    # Use jittered sampling approach
        # i.e. jitter the intervals when estimating probability densities
        # useful for burstiness for some reason.
    te_calculator.setProperty("DO_JITTERED_SAMPLING", "true")

    # noise level can be used to scale a random value that will shift 
    #each interval
        # setting to zero looks like adds no jittering
    te_calculator.setProperty("JITTERED_SAMPLING_NOISE_LEVEL", "0")

# ============================== WITHIN LAYERS

    data_paths= paths_to_data(probename='mouse2probe8')
    for layer in data_paths.keys():
        print(f"Pairwise TEs between cells in {layer}:")
        
        for cell_a in data_paths[layer]:
            for cell_b in data_paths[layer]:
                if cell_a == cell_b: continue
                
                source_spikes = read_spike_times(cell_a)
                dest_spikes = read_spike_times(cell_b)
                
                recording_length = min(len(source_spikes), len(dest_spikes))
                    # recording lengths aren't always equal
                        # verify this isn't because of a mistake I've made 
                if recording_length < NUM_SPIKES:
                    continue
                    # recordings sometimes only hundreds of spikes long
                        # verify this isn't because of a mistake I've made
                
                #choose a random 3000 consec spikes from source and destination
                
                rand_idx = random.randint(0, recording_length - NUM_SPIKES)
                
                source_obsv = source_spikes[rand_idx:rand_idx + NUM_SPIKES] 
                dest_obsv = source_spikes[rand_idx:rand_idx + NUM_SPIKES]
                
                print(f"\t{nice_cell_name(cell_a)} to {nice_cell_name(cell_b)}")
                
                te_calculator.startAddObservations()
                te_calculator.addObservations(
                    JArray(JDouble, 1)(source_obsv),
                    JArray(JDouble, 1)(dest_obsv)
                )
                te_calculator.finaliseAddObservations()
                
                result = te_calculator.computeAverageLocalOfObservations()
                print(f"\t\tTE result {result:.4f} nats")
                significance = te_calculator.computeSignificance(
                    NUM_SURROGATES, result)
                print(f"\t\t{significance.pValue}")
    
    sys.exit()            
    
# ============================== David's sample

    print("Independent Poisson Processes")
    results_poisson = np.zeros(NUM_REPS)
    for i in range(NUM_REPS):
        te_calculator.startAddObservations()
        for j in range(NUM_OBSERVATIONS):
            sourceArray = NUM_SPIKES*np.random.random(NUM_SPIKES) 
            #random.random() returns between 0 and 1
            sourceArray.sort()
            destArray = NUM_SPIKES*np.random.random(NUM_SPIKES)
            destArray.sort()
            condArray = NUM_SPIKES*np.random.random((2, NUM_SPIKES))
            condArray.sort(axis = 1)
            te_calculator.addObservations(
                JArray(JDouble, 1)(sourceArray), 
                JArray(JDouble, 1)(destArray), 
                JArray(JDouble, 2)(condArray))
        te_calculator.finaliseAddObservations();
        result = te_calculator.computeAverageLocalOfObservations()
        print("TE result %.4f nats" % (result,))
        sig = te_calculator.computeSignificance(NUM_SURROGATES, result)
        print(sig.pValue)
        results_poisson[i] = result
    print(
        "Summary: mean ", np.mean(results_poisson), 
        " std dev ", np.std(results_poisson)
        )

if __name__ == '__main__':
    main()
