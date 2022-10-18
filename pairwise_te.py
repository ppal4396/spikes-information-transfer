'''Transfer entropy (TE) calculation on generated spike train data using the 
continuous-time TE estimator'''

from jpype import *
import random
import math
import os
import numpy as np
import json
import sys
import random
# ================================ config
NUM_REPS = 2
NUM_SPIKES = int(3e3)
NUM_OBSERVATIONS = 2
NUM_SURROGATES = 10
jar_location = "/Users/preethompal/Documents/USYD/honours/jidt/infodynamics.jar"

# ============================== library
def read_spike_times(path):
    '''read spike times from file given in <path>, return list'''
    spike_times = []
    with open('mouse2probe8/mouse_2_probe_8layer_4_cell_2.txt', 'r') as f:
        spike_times = [float(x.strip()) for x in f.readlines()]
    return spike_times
    
def intervals_from_spike_times(spike_times):
    '''get NUM_SPIKES intervals from some random point in <spike_times>, return
       list
    '''
    intervals = []
    random_spike = random.randint(NUM_SPIKES, len(spike_times))
    for i in range(random_spike - 1, random_spike - (NUM_SPIKES + 1), -1):
        intervals.append(spike_times[random_spike] - spike_times[i])
    return intervals
        

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
        # i.e. the target history, encoded in intervals? 
        # initialises destPastIntervals to {1, 2} ???
    te_calculator.setProperty("DEST_PAST_INTERVALS", "1,2")

    # as previous, but for source
    te_calculator.setProperty("SOURCE_PAST_INTERVALS", "1,2")

    #as previous, but for conditioned intervals.
        #first one is target, second one is source??, or both are target
    te_calculator.appendConditionalIntervals(JArray(JInt, 1)([1, 2]))
    te_calculator.appendConditionalIntervals(JArray(JInt, 1)([1, 2]))

    # i don't understand what any of the three above "intervals" are.

    # Use jittered sampling approach
        # i.e. jitter the intervals when estimating probability densities
        # useful for burstiness for some reason.
    te_calculator.setProperty("DO_JITTERED_SAMPLING", "true")

    # noise level can be used to scale a random value that will shift 
    #each interval
        # setting to zero looks like adds no jittering
    te_calculator.setProperty("JITTERED_SAMPLING_NOISE_LEVEL", "0")

    print("mouse_2_layer_4_cell_1 to mouse_2_layer_4_cell_2")

    cell_one_path = 'mouse2probe8/mouse_2_probe_8layer_4_cell_1.txt'
    cell_two_path = 'mouse2probe8/mouse_2_probe_8layer_4_cell_2.txt'
    
    cell_one_spike_times = read_spike_times(cell_one_path)
    cell_two_spike_times = read_spike_times(cell_two_path)
    
    random_idx = random.randint(0, len(cell_one_spike_times) - NUM_SPIKES)
    #source:
    cell_one_obvs = cell_one_spike_times[random_idx:random_idx + NUM_SPIKES]
    #target:
    cell_two_obvs = cell_two_spike_times[random_idx:random_idx + NUM_SPIKES]
    #not sure where cond arrays are meant to be. 
    # target history and other source?
    # will just use random for now.
    condArray = NUM_SPIKES*np.random.random((2, NUM_SPIKES))
    te_calculator.startAddObservations()
    

    sys.exit()

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
