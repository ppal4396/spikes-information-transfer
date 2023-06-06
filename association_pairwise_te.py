import glob
from jpype import JPackage, startJVM, JArray, JDouble, getDefaultJVMPath
from pairwise_te import read_spike_times, paths_to_data, nice_cell_name, calculate_average_te_per_spike
import random
import numpy as np
import sys

# ==============================================================================
#                          Completed Associations:
#  
# MOUSE 2(Waksman)
# - probe 8 = right primary visual cortex (VISp)
#         WITH
#               probe 6 = right anterior visual area (VISa)

# - probe 3 = left primary visual cortex (VISp)
#         WITH
#             probe 7 = left rostrolateral visual area (VISrl)
#             and LGd-sh (dorsal part of the lateral geniculate complex, shell)

# MOUSE 1 (Krebs)
# - probes 3 = right primary visual cortex (VISp)
#         WITH
#               probe 7 = right primary visual cortex (VISp)
#             note asymmetry, could reflect columns to be homogenous processing

# - probes 4 = left primary visual cortex (VISp)
#         WITH
#               probe 8 = left primary visual cortex (VISp)
#               only some connectivity.

# ==============================================================================

# =========================== config

NUM_SPIKES = int(sys.argv[4])
NUM_SURROGATES = 100
jar_location = "/Users/preethompal/Documents/USYD/honours/jidt/infodynamics.jar"
java_location = "/usr/local/opt/openjdk/bin/java"
cluster = sys.argv[3] == "cluster"
if cluster:
    jar_location = "/home/ppal4396/jidt/infodynamics.jar"
    java_location = getDefaultJVMPath()
mouse1 = sys.argv[1]
mouse2 = sys.argv[2]
do_exact_num_spikes = True
do_low_bound_num_spikes = False
NUM_SPIKES_COPY = NUM_SPIKES
P_VALUE = 0.05

# ============================ main
def main():
    startJVM(java_location, 
             "-ea", 
             "-Djava.class.path=" + jar_location)
    
    package = "infodynamics.measures.spiking.integration"
    TECalculator = JPackage(package).TransferEntropyCalculatorSpikingIntegration
    te_calculator = TECalculator()

    te_calculator.setProperty("knns", "4")
    te_calculator.setProperty("NORM_TYPE", "MAX_NORM") 
    te_calculator.setProperty("DEST_PAST_INTERVALS", "1,2")
    te_calculator.setProperty("SOURCE_PAST_INTERVALS", "1,2")
    te_calculator.setProperty("DO_JITTERED_SAMPLING", "false")
    te_calculator.setProperty("JITTERED_SAMPLING_NOISE_LEVEL", "0")

# ============================== BETWEEN LAYERS
    data_paths1 = paths_to_data(probename=mouse1)
    data_paths2 = paths_to_data(probename=mouse2)


    job_no = ''
    source_layers_to_do = list(data_paths1.keys())
    dest_layers_to_do = list(data_paths2.keys())
    if len(sys.argv) > 5:
        if sys.argv[5] == 'lower':
            source_layers_to_do = source_layers_to_do[:2]
        elif sys.argv[4] == 'upper':
            source_layers_to_do = source_layers_to_do[2:]
        job_no = sys.argv[6]

    for layer_a in source_layers_to_do:
        for layer_b in dest_layers_to_do:

            if 'Thalamus' in layer_b:
                NUM_SPIKES = int(3e3)
            else:
                NUM_SPIKES = NUM_SPIKES_COPY
            
            lay_a_to_lay_b_te_results = []
            
            n_links = 0
            for cell_a in data_paths1[layer_a]:
                for cell_b in data_paths2[layer_b]:
                    
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

                    n_links += 1

                    te_calculator.initialise()
                    te_calculator.startAddObservations()
                    te_calculator.addObservations(
                        JArray(JDouble, 1)(source_obsv),
                        JArray(JDouble, 1)(dest_obsv)
                    )
                    te_calculator.finaliseAddObservations()
                    
                    result = te_calculator.computeAverageLocalOfObservations()
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
                    
                    res_path = f'results/{mouse1}_to_{mouse2}/{layer_a}_to_{layer_b}.csv'
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
            
            with open(f'results/{mouse1}_to_{mouse2}/pairwise_summary{job_no}.csv','a+') as f:
                line = f"{layer_a},"
                line += f"{layer_b},"
                line += f"{lay_a_to_lay_b_avg},"
                line += f"{lay_a_to_lay_b_sd},"
                line += f"{n_sig_links},"
                line += f"{n_links}\n"
                f.write(line)
         
if __name__ == '__main__':
    main()