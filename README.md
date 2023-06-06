# Measuring information transfer between neural spike trains

This repository contains the code used in my honours project at USYD, to 
quantify information transfer between neural spike trains.

The dataset used was published with this paper: 
https://www.science.org/doi/10.1126/science.aav7893
We analysed the Neuropixels data of spontaneuous activity in visual areas of the
two mice Waksman and Krebs.

Information transfer is quantified using the information theoretic measure
called 'transfer entropy'. The tool used for estimating transfer entropy
between events (like neural spikes) in continuous-time was developed in this 
study:
https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008054

We also applied the continuous-time estimator to infer effective networks of 
multivariate information transfers. You can read about these networks
here: https://pubmed.ncbi.nlm.nih.gov/31410382/

`pairwise_te.py` contains transfer entropy estimations between pairs of cells in
our dataset and is adapted from https://github.com/jlizier/jidt/blob/master/demos/python/SpikingTE/SpikeTrainTETesting.py 

`eff_net_inf.py` contains transfer estimation inside an algorithm for effective
network inference and is adapted from https://github.com/jlizier/jidt/blob/master/demos/python/EffectiveNetworkInference/spiking/net_inf.py 

Scripts used to plot results are also included.

Scripts used to extract spike times from the Neuropixels dataset and preprocess 
them into python pickles for use in effective network inference are also included.