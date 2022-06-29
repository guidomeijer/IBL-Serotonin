# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 11:36:24 2022

@author: Guido
"""

import numpy as np
from os.path import join, isdir
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from os import mkdir
import seaborn as sns
from brainbox.io.one import SpikeSortingLoader
from brainbox.metrics.single_units import spike_sorting_metrics
from brainbox.task.closed_loop import (responsive_units, roc_single_event,
                                       roc_between_two_events, generate_pseudo_blocks)
from brainbox.plot import peri_event_time_histogram
from brainbox.population.decode import get_spike_counts_in_bins
from serotonin_functions import (paths, remap, query_ephys_sessions, load_trials, figure_style,
                                 get_artifact_neurons)
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()
lda = LinearDiscriminantAnalysis()

# Settings
NEURON_QC = False
PLOT = True
MIN_FR = 0.5  # minimum firing rate over the whole recording
T_BEFORE = 0 
T_AFTER = 0.3
MIN_NEURONS = 10
_, save_path = paths()

# Query sessions
rec = query_ephys_sessions(one=one)

# Load in artifact neurons
artifact_neurons = get_artifact_neurons()

lda_opto_df = pd.DataFrame()
for i in rec.index.values:

    # Get session details
    pid, eid, probe = rec.loc[i, 'pid'], rec.loc[i, 'eid'], rec.loc[i, 'probe']
    subject, date = rec.loc[i, 'subject'], rec.loc[i, 'date']
    print(f'Processing {subject} {date}')

    # Load trials dataframe
    try:
        trials = load_trials(eid, laser_stimulation=True, patch_old_opto=False, one=one)
    except:
        print('Could not load trials')
        continue
    if trials.shape[0] < 200:
        continue

    # Load in spikes
    sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
    spikes, clusters, channels = sl.load_spike_sorting()
    clusters = sl.merge_clusters(spikes, clusters, channels)

    # Filter neurons that pass QC and artifact neurons
    if NEURON_QC:
        print('Calculating neuron QC metrics..')
        qc_metrics, _ = spike_sorting_metrics(spikes.times, spikes.clusters, spikes.amps, spikes.depths,
                                              cluster_ids=np.arange(clusters.channels.size))
        clusters_pass = np.where(qc_metrics['label'] > 0.5)[0]
    else:
        clusters_pass = np.unique(spikes.clusters)
    clusters_pass = clusters_pass[~np.isin(clusters_pass, artifact_neurons.loc[
        artifact_neurons['pid'] == pid, 'neuron_id'].values)]
    clusters['region'] = remap(clusters['acronym'], combine=True, abbreviate=True)

    # Loop over regions
    for r, region in enumerate(np.unique(clusters['region'])):
        if region == 'root':
            continue

        # Exclude neurons with low firing rates
        clusters_in_region = np.where(clusters['region'] == region)[0]
        fr = np.empty(clusters_in_region.shape[0])
        for nn, neuron_id in enumerate(clusters_in_region):
            fr[nn] = np.sum(spikes.clusters == neuron_id) / spikes.clusters[-1]
        clusters_in_region = clusters_in_region[fr >= MIN_FR]
        
        # Get spikes and clusters
        spks_region = spikes.times[np.isin(spikes.clusters, clusters_in_region) & np.isin(spikes.clusters, clusters_pass)]
        clus_region = spikes.clusters[np.isin(spikes.clusters, clusters_in_region) & np.isin(spikes.clusters, clusters_pass)]
        if np.unique(clus_region).shape[0] < MIN_NEURONS:
            continue      
         
        # Trial start
        times = np.column_stack(((trials['goCue_times'] - T_BEFORE), (trials['goCue_times'] + T_AFTER)))
        spike_counts, cluster_ids = get_spike_counts_in_bins(spks_region, clus_region, times)
        spike_counts = spike_counts.T
        lda_projection = lda.fit_transform(spike_counts, trials['laser_stimulation'])
        lda_trial_start = (np.abs(np.nanmean(lda_projection[trials['laser_stimulation'] == 0]))
                           + np.abs(np.nanmean(lda_projection[trials['laser_stimulation'] == 1])))
        
        # Reward
        trials_slice = trials[trials['feedbackType'] == 1]
        times = np.column_stack(((trials_slice['feedback_times'] - T_BEFORE), (trials_slice['feedback_times'] + T_AFTER)))
        spike_counts, cluster_ids = get_spike_counts_in_bins(spks_region, clus_region, times)
        spike_counts = spike_counts.T
        lda_projection = lda.fit_transform(spike_counts, trials_slice['laser_stimulation'])
        lda_reward = (np.abs(np.nanmean(lda_projection[trials_slice['laser_stimulation'] == 0]))
                      + np.abs(np.nanmean(lda_projection[trials_slice['laser_stimulation'] == 1])))
        
        # Ommission
        trials_slice = trials[trials['feedbackType'] == -1]
        times = np.column_stack(((trials_slice['feedback_times'] - T_BEFORE), (trials_slice['feedback_times'] + T_AFTER)))
        spike_counts, cluster_ids = get_spike_counts_in_bins(spks_region, clus_region, times)
        spike_counts = spike_counts.T
        lda_projection = lda.fit_transform(spike_counts, trials_slice['laser_stimulation'])
        lda_ommission = (np.abs(np.nanmean(lda_projection[trials_slice['laser_stimulation'] == 0]))
                         + np.abs(np.nanmean(lda_projection[trials_slice['laser_stimulation'] == 1])))
        
        # ITI
        times = np.column_stack((((trials['goCue_times'] - T_BEFORE) - 0.5),
                                 ((trials['goCue_times'] + T_AFTER) - 0.5)))
        spike_counts, cluster_ids = get_spike_counts_in_bins(spks_region, clus_region, times)
        spike_counts = spike_counts.T
        lda_projection = lda.fit_transform(spike_counts, trials['laser_stimulation'])
        lda_iti = (np.abs(np.nanmean(lda_projection[trials['laser_stimulation'] == 0]))
                   + np.abs(np.nanmean(lda_projection[trials['laser_stimulation'] == 1])))
        
        # Add results to df
        lda_opto_df = pd.concat((lda_opto_df, pd.DataFrame(index=[lda_opto_df.shape[0]+1], data={
            'subject': subject, 'date': date, 'eid': eid, 'probe': probe,
            'region': region, 'lda_trial_start': lda_trial_start, 'lda_iti': lda_iti,
            'lda_reward': lda_reward, 'lda_ommission': lda_ommission})))

    lda_opto_df.to_csv(join(save_path, 'lda_task_events_opto.csv'))
