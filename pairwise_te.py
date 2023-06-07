'''Transfer entropy (TE) calculation on spike train data using the 
continuous-time TE estimator'''
# Reference: https://github.com/jlizier/jidt

from jpype import JPackage, startJVM, JArray, JDouble, getDefaultJVMPath

import random
import math
import os
import numpy as np
import json
import sys
import glob
# ================================ config
if __name__ == '__main__':
    if len(sys.argv) < 5:
        print("usage: python pairwise_te.py <mouse> <exact|low_bound_num_spikes> <cluster|local> <NUM_SPIKES>")
        sys.exit(-1)
    cluster = sys.argv[3] == "cluster"
    NUM_SPIKES = int(sys.argv[4])
    NUM_OBSERVATIONS = 2
    NUM_SURROGATES = 100
    jar_location = "/Users/preethompal/Documents/USYD/honours/jidt/infodynamics.jar"
    java_location = "/usr/local/opt/openjdk/bin/java"
    if cluster:
        jar_location = "/home/ppal4396/jidt/infodynamics.jar"
        java_location = getDefaultJVMPath()
    mouse = sys.argv[1]
    do_exact_num_spikes = sys.argv[2] == "exact_num_spikes"
    do_low_bound_num_spikes = sys.argv[2] == "low_bound_num_spikes"
    NUM_SPIKES_COPY = NUM_SPIKES #save NUM_SPIKES since I'm hardcoding thalamus NUM_SPIKES
    P_VALUE = 0.05

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
    if probename == 'mouse2probe7':
        data_paths['Thalamus co'] = []
        data_paths['Thalamus sh'] = [] 
    for cell in glob.glob(f"data/{probename}/*.txt"):
        if 'layer_23' in cell:
            data_paths['Layer 23'].append(cell)
        elif 'layer_4' in cell:
            data_paths['Layer 4'].append(cell)
        elif 'layer_5' in cell:
            data_paths['Layer 5'].append(cell)
        elif 'layer_6' in cell:
            data_paths['Layer 6'].append(cell)
    if probename == 'mouse2probe7':
        for cell in glob.glob(f'data/{probename}/thalamus/*co_cell*.txt'):
            data_paths['Thalamus co'].append(cell)
        for cell in glob.glob(f'data/{probename}/thalamus/*sh_cell*.txt'):
            data_paths['Thalamus sh'].append(cell)
    return data_paths

def nice_cell_name(path):
    ''' returns pretty string from file path to data for a cell'''
    return f"Cell {path.split('_')[-1].split('.')[0]}"

def calculate_average_te_per_spike(result, obs_len, n_spikes):
    '''Average TE per spike: multiply result by length of 
    time in observation window then divide by number of spikes'''
    avg_te_per_spike = (result * obs_len) / n_spikes
    return avg_te_per_spike

# ============================ main
def main():
    # Start the JVM (add the "-Xmx" option with say 1024M if you get crashes due
     #to not enough memory space)
    startJVM(java_location, 
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
        # 1 and 2 respectively label interval from observation point to 
        # target spike and an earlier observation point to target spike.
        # i.e target history being conditioned on, is two intervals long.
    te_calculator.setProperty("DEST_PAST_INTERVALS", "1,2")

    # source history (which is included in TE formulation as a rate) is two
    #intervals long.
    te_calculator.setProperty("SOURCE_PAST_INTERVALS", "1,2")

    # You can use te_calculator.appendConditionalIntervals() for the above but
    #for conditioned processes. 
        # ignoring in pairwise TE.

    # Use jittered sampling approach
    # when choosing a random number of sample histories to estimate TE, 
    # instead of laying these out uniformly; place them at existing target 
    # spikes, then add uniform noise on the interval [-80 ms, 80 ms].
    # Useful for high density bursts.
    te_calculator.setProperty("DO_JITTERED_SAMPLING", "false")

    # noise level can be used to scale a random value that will shift 
    #each interval 
    #do not set DO_JITTERED_SAMPLING to true with 0 here!
    te_calculator.setProperty("JITTERED_SAMPLING_NOISE_LEVEL", "0")

# ============================== BETWEEN LAYERS
    data_paths= paths_to_data(probename=mouse)

    job_no = ''
    source_layers_to_do = list(data_paths.keys())
    if len(sys.argv) > 5:
        if sys.argv[5] == 'lower':
            source_layers_to_do = source_layers_to_do[:2]
        elif sys.argv[4] == 'upper':
            source_layers_to_do = source_layers_to_do[2:]
        job_no = sys.argv[6]

    dest_layers_to_do = list(data_paths.keys())

    for layer_a in source_layers_to_do:
        for layer_b in dest_layers_to_do:
            # print(f"Pairwise TEs between cells from {layer_a} to {layer_b}:")

            if 'Thalamus' in layer_b:
                NUM_SPIKES = int(3e3)
            else:
                NUM_SPIKES = NUM_SPIKES_COPY
            
            lay_a_to_lay_b_te_results = []
            
            n_links = 0
            for cell_a in data_paths[layer_a]:
                for cell_b in data_paths[layer_b]:
                    if cell_a == cell_b: continue
                    
                    source_spikes = read_spike_times(cell_a)
                    dest_spikes = read_spike_times(cell_b)
                    
                    dest_length = len(dest_spikes)

                    if dest_length < NUM_SPIKES: continue
                    
                    if do_exact_num_spikes:
                        #choose a random NUM_SPIKES consec spikes from dest
                        rand_idx = random.randint(0, dest_length - NUM_SPIKES)
                        dest_obsv = dest_spikes[rand_idx:rand_idx + NUM_SPIKES]
                    
                    elif do_low_bound_num_spikes:
                        # as long as dest is over NUM_SPIKES long, observe all 
                        # spikes
                        dest_obsv = dest_spikes
                    
                    #choose source spikes within dest's obsv window
                    start_time = dest_obsv[0]
                    end_time = dest_obsv[-1]
                    start_idx = None
                    stop_idx = None
                    for idx, time_stamp in enumerate(source_spikes):
                        if not start_idx and time_stamp > start_time:
                            start_idx = idx
                        if ((start_idx and time_stamp >= end_time) or
                           (start_idx and idx == len(source_spikes) - 1)):
                            stop_idx = idx
                            break                                    
                    
                    if not (start_idx and stop_idx): continue            
                    if stop_idx - start_idx < 100: continue
                        #i.e. atleast 100 spikes in source.
                    
                    source_obsv = source_spikes[ start_idx : stop_idx ]
                    # print(f"start_time: {start_time}, end_time: {end_time}")
                    # print(f"\tdifference: {end_time - start_time}")
                    # print("source stamps:\n", source_obsv)
                    # print("destination stamps:\n", dest_obsv)
                    # print("\tnum source spikes:", len(source_obsv),
                    # "num dest spikes:", len(dest_obsv))
                    
                    # print(
                    # f"\t{nice_cell_name(cell_a)} to {nice_cell_name(cell_b)}")

                    n_links += 1

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

                    n_source_spikes = len(source_obsv)
                    avg_te_per_source_spike = calculate_average_te_per_spike(
                        result,
                        end_time - start_time,
                        n_source_spikes
                        )
                    
                    n_dest_spikes = len(dest_obsv)
                    avg_te_per_dest_spike = calculate_average_te_per_spike(
                        result,
                        end_time - start_time,
                        n_dest_spikes
                        )
                    
                    # bias correction by shifting result by mean of surrogates
                    surrogate_mean = significance.getMeanOfDistribution()
                    corrected_result = result - surrogate_mean
                    
                    lay_a_to_lay_b_te_results.append(
                        (result, significance.pValue))
                    
                    res_path = f'results/{mouse}/{layer_a}_to_{layer_b}.csv'
                    with open(res_path, 'a+') as f:
                        line = f"{nice_cell_name(cell_a)},"
                        line += f"{nice_cell_name(cell_b)},"
                        line += f"{result:.4f},"
                        line += f"{corrected_result:.4f},"
                        line += f"{significance.pValue},"
                        line += f"{surrogate_mean:.4f},"
                        line += f"{n_source_spikes},"
                        line += f"{avg_te_per_source_spike},"
                        line += f"{n_dest_spikes},"
                        line += f"{avg_te_per_dest_spike},"
                        line += f"{end_time - start_time}\n"
                        f.write(line)
            
            #zero the negative transfer entropy results and compute avg & sd.
            te_zeroed_negs = np.asarray(list(
                map(
                    lambda x: 0 if x[0] < 0 else x[0], 
                    lay_a_to_lay_b_te_results)
                ))
            lay_a_to_lay_b_avg = np.mean(te_zeroed_negs)
            lay_a_to_lay_b_sd = np.std(te_zeroed_negs)
            
            # count number of significant transfer entropies
            n_sig_links = (np.asarray(list(
                map(lambda x: x[1], lay_a_to_lay_b_te_results)
            )) < P_VALUE).sum()
            
            with open(f'results/{mouse}/pairwise_summary{job_no}.csv','a+') as f:
                line = f"{layer_a},"
                line += f"{layer_b},"
                line += f"{lay_a_to_lay_b_avg},"
                line += f"{lay_a_to_lay_b_sd},"
                line += f"{n_sig_links},"
                line += f"{n_links}\n"
                f.write(line)
         
if __name__ == '__main__':
    main()
    