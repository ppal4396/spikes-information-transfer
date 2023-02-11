'''Transfer entropy (TE) calculation on spike train data using the 
continuous-time TE estimator'''

#TODO: remove limit on target NUM_SPIKES , or allow a much larger range.
#TODO: when doing average TE per spike: can just multiply result by length of 
#      time in observation window then divide by number of spikes

#Thought: is getting average local TE limited? what if there are random bursts?

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
from scipy import stats
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
        'Layer 23' : [],
        'Layer 4' : [], 
        'Layer 5' : [], 
        'Layer 6' : []
        } 
    for cell in glob.glob(f"data/{probename}/*.txt"):
        if 'layer_23' in cell:
            data_paths['Layer 23'].append(cell)
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
        # ignoring in pairwise TE.

    # Use jittered sampling approach
        # i.e. jitter the intervals when estimating probability densities
        # useful for burstiness for some reason.
    te_calculator.setProperty("DO_JITTERED_SAMPLING", "false")

    # noise level can be used to scale a random value that will shift 
    #each interval 
    #do not set DO_JITTERED_SAMPLING to true with 0 here!
    te_calculator.setProperty("JITTERED_SAMPLING_NOISE_LEVEL", "0")

# ============================== WITHIN LAYERS

    data_paths= paths_to_data(probename='mouse2probe8')
    for layer_a in data_paths.keys():
        for layer_b in data_paths.keys():
            # print(f"Pairwise TEs between cells from {layer_a} to {layer_b}:")
            
            lay_a_to_lay_b_te_results = []
            
            for cell_a in data_paths[layer_a]:
                for cell_b in data_paths[layer_b]:
                    if cell_a == cell_b: continue
                    
                    source_spikes = read_spike_times(cell_a)
                    dest_spikes = read_spike_times(cell_b)
                    
                    dest_length = len(dest_spikes)

                    if dest_length < NUM_SPIKES: continue
                        # recordings sometimes only hundreds of spikes long
                    
                    #choose a random NUM_SPIKES consec spikes from destination
                    rand_idx = random.randint(0, dest_length - NUM_SPIKES)
                    dest_obsv = dest_spikes[rand_idx:rand_idx + NUM_SPIKES]
                    
                    #choose source spikes within destination's observation window
                    start_time = dest_obsv[0]
                    end_time = dest_obsv[-1]
                    start_idx = None
                    stop_idx = None
                    for idx, time_stamp in enumerate(source_spikes):
                        if not start_idx and time_stamp > start_time:
                            start_idx = idx
                        if start_idx and time_stamp >= end_time:
                            stop_idx = idx
                            break                                    
                    
                    if not (start_idx and stop_idx): continue            
                    if stop_idx - start_idx < 100: continue
                    
                    source_obsv = source_spikes[ start_idx : stop_idx ]
                    # print(f"start_time: {start_time}, end_time: {end_time}")
                    # print(f"\tdifference: {end_time - start_time}")
                    # print("source stamps:\n", source_obsv)
                    # print("destination stamps:\n", dest_obsv)
                    # print("\tnum source spikes:", len(source_obsv), "num dest spikes:", len(dest_obsv))
                    
                    # print(f"\t{nice_cell_name(cell_a)} to {nice_cell_name(cell_b)}")
                    
                    te_calculator.initialise()
                    te_calculator.startAddObservations()
                    te_calculator.addObservations(
                        JArray(JDouble, 1)(source_obsv),
                        JArray(JDouble, 1)(dest_obsv)
                    )
                    te_calculator.finaliseAddObservations()
                    
                    result = te_calculator.computeAverageLocalOfObservations()
                    # print(f"\t\tTE result {result:.4f} nats")
                    significance = te_calculator.computeSignificance(
                        NUM_SURROGATES, result)
                    # print(f"\t\t{significance.pValue}")
                    
                    # print("mean of distribution:", significance.getMeanOfDistribution())
                    # print("std of distribution:", significance.getStdOfDistribution())
                    
                    lay_a_to_lay_b_te_results.append(
                        (result, significance.pValue))
                    
                    with open(f'results/{layer_a}_to_{layer_b}.csv', 'a+') as f:
                        line = f"{nice_cell_name(cell_a)},{nice_cell_name(cell_b)},{result:.4f},{significance.pValue}\n"
                        f.write(line)
            
            #zero the negative transfer entropy results and compute average.
            te_results = np.asarray(list(
                map(
                    lambda x: 0 if x[0] < 0 else x[0], 
                    lay_a_to_lay_b_te_results)
                ))
            te_avg = np.mean(te_results)
            te_sd = np.std(te_results)
            
            # count number of significant transfer entropies
            num_sig_links = list(
                map(lambda x: x[1], lay_a_to_lay_b_te_results)
            ).count(0.0)
            
            with open(f'results/pairwise_summary.csv', 'a') as f:
                line = f"{layer_a},{layer_b},{te_avg},{te_sd},{num_sig_links}\n"
                f.write(line)
         
if __name__ == '__main__':
    main()
    
def sample_independent_poissons():
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

