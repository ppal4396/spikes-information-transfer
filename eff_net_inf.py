'''Effective network inference'''
# usage: python3 eff_net_inf.py network_type_name num_spikes repeat_number target_index [local,cluster]
# Reference: https://github.com/jlizier/jidt

'''
NET_TYPE mouse2probe8 REPEAT_NUM 0:
        MAX_NUM_SECOND_INTERVALS=3 for target 8, 37, 51. =60 for all other targets.
        MAX_NUM_SPIKES = 1000
        note: lost all logs for this repeat. accidentally overwrote. 
        but I did check over a few of them and they looked fine.
        and I have the job errors (which are empty)
        (60 targets)

NET_TYPE mouse2probe8 REPEAT_NUM 1:
        MAX_NUM_SECOND_INTERVALS set to pos inf (i.e. ignored)
        MAX_NUM_SPIKES = 3000
        computation time limit: stops adding source intervals after (n_sources / 2) intervals have been added already.
        (60 targets)

NET_TYPE mouse2probe3 REPEAT_NUM 0
        MAX_NUM_SECOND_INTERVALS set to pos inf (i.e. ignored)
        MAX_NUM_SPIKES = 1500
        (103 targets)
'''

'''
TODO
    - within v1 first: mouse2probe8, mouse2probe3, mouse1probe3. Other mouse1 probes are pretty undersampled.
    - between v1 probes next; mouse1probe3 & mouse1probe7
    - v1 and thalamus last. mouse2probe3 & mouse2probe7
'''
from jpype import *
import random
import math
import os
import numpy as np
import pickle
import copy
import sys

net_type_name = sys.argv[1] #mouse probe
num_spikes_string = sys.argv[2]
repeat_num_string = sys.argv[3]
target_index_string = sys.argv[4]
cluster = sys.argv[5] == 'cluster'

NUM_SURROGATES_PER_TE_VAL = 100
P_LEVEL = 0.05
# The number of nearest neighbours to consider in the TE estimation.
KNNS = 10
# The number of random sample points laid down will be NUM_SAMPLES_MULTIPLIER * length_of_target_train
NUM_SAMPLES_MULTIPLIER = 5.0
SURROGATE_NUM_SAMPLES_MULTIPLIER = 5.0
# The number of nearest neighbours to consider when using the local permutation method to create surrogates
K_PERM = 20
# The level of the noise to add to the random sample points used in creating surrogates 
JITTERING_LEVEL = 2000

# When MAX_NUM_SECOND_INTERVALS sources have 2 or more history intervals added into the conditioning set, the inference stops
MAX_NUM_SECOND_INTERVALS = float('inf') 
#ignoring the above limit. Instead, stop adding source intervals after (n_sources / 2) intervals have been added to the conditioning set already.

# Exclude target spikes beyond this number
MAX_NUM_TARGET_SPIKES = int(num_spikes_string)
# The spikes file with the below name is expected to contain a single pickled Python list. This list contains numpy arrays. Each
# numpy array contains the spike times of each candidate target.
SPIKES_FILE_NAME = "data/spikes_LIF_" + net_type_name + ".pk"
OUTPUT_PATH = f'results/{net_type_name}/repeat_{repeat_num_string}/'
OUTPUT_FILE_PREFIX = 'inferred_sources_' + net_type_name + "_" + num_spikes_string + "_" + repeat_num_string + target_index_string
LOG_FILE_NAME = "logs/" + net_type_name + "_" + num_spikes_string + "_" + repeat_num_string +  "_" + target_index_string + ".log"

log = open(LOG_FILE_NAME, "w")
sys.stdout = log

def prepare_conditional_trains(calc_object, cond_set, spikes):
        cond_trains = []
        calc_object.clearConditionalIntervals()
        if len(cond_set) > 0:
                for key in cond_set.keys():
                        cond_trains.append(spikes[key])
                        calc_object.appendConditionalIntervals(JArray(JInt, 1)(cond_set[key]))
        return cond_trains

def set_target_embeddings(calc_object, embedding_list):
        if len(embedding_list) > 0:
                embedding_string = str(embedding_list[0])
                for i in range(2, len(embedding_list)):
                        embedding_string += "," + str(embedding_list[i])
                calc_object.setProperty("DEST_PAST_INTERVALS", embedding_string)
        else:
                calc_object.setProperty("DEST_PAST_INTERVALS", "")

target_index = int(target_index_string)                
print("\n****** Network inference for target neuron", target_index, "******\n\n")

def main():  
    if cluster:
        jar_location = "/home/ppal4396/jidt/infodynamics.jar"
        startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jar_location)
    else:
        jar_location = "/Users/preethompal/Documents/USYD/honours/jidt/infodynamics.jar"
        startJVM('/usr/local/opt/openjdk/bin/java', "-ea", "-Djava.class.path=" + jar_location)
            
    teCalcClass = JPackage("infodynamics.measures.spiking.integration").TransferEntropyCalculatorSpikingIntegration
    teCalc = teCalcClass()
    teCalc.setProperty("knns", str(KNNS))
    teCalc.setProperty("NUM_SAMPLES_MULTIPLIER", str(NUM_SAMPLES_MULTIPLIER))
    teCalc.setProperty("SURROGATE_NUM_SAMPLES_MULTIPLIER", str(SURROGATE_NUM_SAMPLES_MULTIPLIER))
    teCalc.setProperty("K_PERM", str(K_PERM))
    teCalc.setProperty("DO_JITTERED_SAMPLING", "false")
    teCalc.setProperty("JITTERED_SAMPLING_NOISE_LEVEL", str(JITTERING_LEVEL))

    spikes = pickle.load(open(SPIKES_FILE_NAME, 'rb'))
    #take the first MAX_NUM_TARGET_SPIKES target spikes.
    if MAX_NUM_TARGET_SPIKES < len(spikes[target_index]):
        spikes[target_index] = spikes[target_index][:MAX_NUM_TARGET_SPIKES]
    #take all other spikes within time window of the above.
    n_skipped_sources = 0
    for source_index in range(0, len(spikes)):
           if source_index == target_index:
                  continue
           spikes[source_index] = spikes[source_index][
                  (spikes[source_index] > spikes[target_index][0]) & (spikes[source_index] < spikes[target_index][-1])
                  ]
           if len(spikes[source_index]) < 10:
                  n_skipped_sources += 1
    print("Number of target spikes: ", len(spikes[target_index])) #can be [0, MAX_NUM_TARGET_SPIKES]
    print(f"First target spike: {spikes[target_index][0]}")
    print(f"Last target spike: {spikes[target_index][-1]}")
    print(f"Will be skipping {n_skipped_sources} sources.")

    # First determine the correct target embedding
    target_embedding_set = [1]
    next_target_interval = 2
    still_significant = True
    print("**** Determining target embedding set ****\n")
    '''Finds TE from one interval at a time in target past to next target state, in context of already added target past intervals.
        --> averaged across local target past/next target state pairs.'''
    while still_significant:
            set_target_embeddings(teCalc, target_embedding_set)
            teCalc.setProperty("SOURCE_PAST_INTERVALS", str(next_target_interval))
            teCalc.startAddObservations()
            teCalc.addObservations(JArray(JDouble, 1)(spikes[target_index]), JArray(JDouble, 1)(spikes[target_index]))
            teCalc.finaliseAddObservations();
            TE = teCalc.computeAverageLocalOfObservations()
            sig = teCalc.computeSignificance(NUM_SURROGATES_PER_TE_VAL, TE)
            print("candidate interval:", next_target_interval, " TE:", TE, " p val:", sig.pValue)
            if sig.pValue > P_LEVEL:
                    print("Lost significance, end of target embedding determination")
                    still_significant = False
            else:
                    target_embedding_set.append(next_target_interval)
                    next_target_interval += 1
    print("target embedding set:", target_embedding_set, "\n\n")


    # Now add the sources
    # cond_set is a dictionary where keys are added sources and values are lists of included intervals for the
    # source key.
    cond_set = dict()
    # next_interval_for_each_candidate will be a matrix with two columns
    # first column has the source indices, second has the next interval that will be considered
    next_interval_for_each_candidate = np.arange(0, len(spikes), dtype = np.intc)
    next_interval_for_each_candidate = next_interval_for_each_candidate[next_interval_for_each_candidate != target_index]
    next_interval_for_each_candidate = np.column_stack((next_interval_for_each_candidate, np.ones(len(next_interval_for_each_candidate),  dtype = np.intc)))
    still_significant = True
    TE_vals_at_each_round = []
    surrogate_vals_at_each_round = []
    print("**** Adding Sources ****\n")
    num_twos = 0
    num_sig_intervals = 0
    while still_significant:
            print("Current conditioning set:")
            for key in cond_set.keys():
                    print("source", key, "intervals", cond_set[key])
            print("\nEstimating TE on candidate sources")
            cond_trains = prepare_conditional_trains(teCalc, cond_set, spikes)
            TE_vals = np.zeros(next_interval_for_each_candidate.shape[0]) #for each source right now.
            debiased_TE_vals = -1 * np.ones(next_interval_for_each_candidate.shape[0]) #for each source right now.
            surrogate_vals = -1 * np.ones((next_interval_for_each_candidate.shape[0], NUM_SURROGATES_PER_TE_VAL)) #for each source right now
            debiased_surrogate_vals = 1 - np.ones((next_interval_for_each_candidate.shape[0], NUM_SURROGATES_PER_TE_VAL)) #for each source right now
            #iterate through every source
            for i in range(next_interval_for_each_candidate.shape[0]):
                    #if this source has less than 10 spikes, skip.
                    if len(spikes[next_interval_for_each_candidate[i, 0]]) < 10:
                            print(f"Skipping source {next_interval_for_each_candidate[i, 0]} since it has less than 10 spikes.")
                            continue
                    teCalc.startAddObservations()
                    #check TE from one source interval to target state, in context of (embedded target past + conditioning set intervals)
                        #averaged across local source interval/target state/conditioned interval trios
                    teCalc.setProperty("SOURCE_PAST_INTERVALS", str(next_interval_for_each_candidate[i, 1]))
                    if len(cond_set) > 0:
                            teCalc.addObservations(JArray(JDouble, 1)(spikes[next_interval_for_each_candidate[i, 0]]),
                                                JArray(JDouble, 1)(spikes[target_index]), JArray(JDouble, 2)(cond_trains))
                    else:
                            teCalc.addObservations(JArray(JDouble, 1)(spikes[next_interval_for_each_candidate[i, 0]]),
                                                JArray(JDouble, 1)(spikes[target_index]))
                    teCalc.finaliseAddObservations();
                    TE_vals[i] = teCalc.computeAverageLocalOfObservations() #for this source right now.
                    sig = teCalc.computeSignificance(NUM_SURROGATES_PER_TE_VAL, TE_vals[i])
                    surrogate_vals[i] = sig.distribution
                    debiased_TE_vals[i] = TE_vals[i] - np.mean(surrogate_vals[i])
                    debiased_surrogate_vals[i] = sig.distribution - np.mean(surrogate_vals[i])
                    print("Source", next_interval_for_each_candidate[i, 0], "Interval", next_interval_for_each_candidate[i, 1],
                        " TE:",  str(debiased_TE_vals[i]))
                    log.flush()

            TE_vals_at_each_round.append(TE_vals)
            surrogate_vals_at_each_round.append(surrogate_vals)
            sorted_TE_indices = np.argsort(debiased_TE_vals)
            print("\nSorted order of sources:\n", next_interval_for_each_candidate[:, 0][sorted_TE_indices[:]])

            index_of_max_candidate = sorted_TE_indices[-1]
            samples_from_max_dist = np.max(debiased_surrogate_vals, axis = 0) #distribution where each value is a max from surrogate values.
            np.sort(samples_from_max_dist)
            #find the first source interval that had lower estimate than distribution of max.
            index_of_first_greater_than_estimate = np.searchsorted(samples_from_max_dist > debiased_TE_vals[index_of_max_candidate], 1)
            p_val = (NUM_SURROGATES_PER_TE_VAL - index_of_first_greater_than_estimate)/float(NUM_SURROGATES_PER_TE_VAL)
            print("\nMaximum candidate is source", next_interval_for_each_candidate[index_of_max_candidate, 0],
                "interval", next_interval_for_each_candidate[index_of_max_candidate, 1])
            print("p: ", p_val)
            if p_val <= P_LEVEL:
                    #if source already in cond_set, add interval to list.
                    if (next_interval_for_each_candidate[index_of_max_candidate, 0]) in cond_set:
                            cond_set[next_interval_for_each_candidate[index_of_max_candidate, 0]].append(next_interval_for_each_candidate[index_of_max_candidate, 1])
                    #otherwise add source to cond_set (create new list)
                    else:
                            cond_set[next_interval_for_each_candidate[index_of_max_candidate, 0]] = [next_interval_for_each_candidate[index_of_max_candidate, 1]]

                    if next_interval_for_each_candidate[index_of_max_candidate, 1] == 2:
                            num_twos += 1
                    if num_twos >= MAX_NUM_SECOND_INTERVALS:
                            print("\nMaximum number of second intervals reached\n\n")
                            still_significant = False
                    #----- Instead of MAX_NUM_SECOND_INTERVALS, use total num intervals added to limit computation time -----
                    if num_sig_intervals >= len(spikes) / 2:
                           print("\nMaximum number of intervals reached.\n\n")
                           still_significant = False
                    #-----------------------------------------------------------

                    next_interval_for_each_candidate[index_of_max_candidate, 1] += 1
                    num_sig_intervals += 1
                    
                    print("\nCandidate added\n\n")
            else:
                    still_significant = False
                    print("\nLost Significance\n\n")

    print("**** Pruning Sources ****\n")
    # Repeatedly removes the connection that has the lowest TE out of all insignificant connections.
    # Only considers the furthest intervals as candidates in each round.
    everything_significant = False
    while not everything_significant:
            print("Current conditioning set:")
            for key in cond_set.keys():
                    print("source", key, "intervals", cond_set[key])
            print("\nEstimating TE on candidate sources")
            everything_significant = True
            insignificant_sources = []
            insignificant_sources_TE = []
            for candidate_source in cond_set:
                    cond_set_minus_candidate = copy.deepcopy(cond_set)
                    # If more than one interval, remove the last
                    if len(cond_set_minus_candidate[candidate_source]) > 1:
                            cond_set_minus_candidate[candidate_source] = cond_set_minus_candidate[candidate_source][:-1]
                    # Otherwise, remove source from dict
                    else:
                            cond_set_minus_candidate.pop(candidate_source)
                    teCalc.setProperty("SOURCE_PAST_INTERVALS", str(cond_set[candidate_source][-1]))
                    cond_trains = prepare_conditional_trains(teCalc, cond_set_minus_candidate, spikes)
                    teCalc.startAddObservations()
                    if len(cond_set_minus_candidate) > 0:
                            teCalc.addObservations(JArray(JDouble, 1)(spikes[candidate_source]), JArray(JDouble, 1)(spikes[target_index]), JArray(JDouble, 2)(cond_trains))
                    else:
                            teCalc.addObservations(JArray(JDouble, 1)(spikes[candidate_source]), JArray(JDouble, 1)(spikes[target_index]))
                    teCalc.finaliseAddObservations();
                    TE = teCalc.computeAverageLocalOfObservations()
                    sig = teCalc.computeSignificance(NUM_SURROGATES_PER_TE_VAL, TE)
                    print("Source", candidate_source, "Interval", cond_set[candidate_source][-1],
                        " TE:",  str(round(TE, 2)), " p val:", sig.pValue)
                    if sig.pValue > P_LEVEL:
                            everything_significant = False
                            insignificant_sources.append(candidate_source)
                            insignificant_sources_TE.append(TE)
            if not everything_significant:
                    min_TE_source = insignificant_sources[np.argmin(insignificant_sources_TE)]
                    print("removing source", min_TE_source, "interval", cond_set[min_TE_source][-1])
                    if len(cond_set[min_TE_source]) > 1:
                            cond_set[min_TE_source] = cond_set[min_TE_source][:-1]
                    else:
                            cond_set.pop(min_TE_source)

    print("\n\n****** Final Inferred Source Set ******\n")
    for key in cond_set.keys():
            print("source", key, "intervals", cond_set[key])
    print("\nTrue Sources: unavailable.")

    output_file = open(OUTPUT_PATH + OUTPUT_FILE_PREFIX + ".pk", 'wb')
    pickle.dump(cond_set, output_file)
    pickle.dump(surrogate_vals_at_each_round, output_file)
    pickle.dump(TE_vals_at_each_round, output_file)
    output_file.close()
    log.close()

if __name__ == '__main__':
    main()
 