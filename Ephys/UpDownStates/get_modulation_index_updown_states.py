#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 11:03:49 2022
By: Guido Meijer
"""

import numpy as np
from os.path import join
import ssm
from brainbox.task.closed_loop import roc_single_event
import pandas as pd
from brainbox.singlecell import calculate_peths
from brainbox.io.one import SpikeSortingLoader
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from serotonin_functions import (load_passive_opto_times, get_neuron_qc, remap, paths,
                                 query_ephys_sessions, load_subjects, get_artifact_neurons,
                                 high_level_regions)
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

K = 2    # number of discrete states
D = 10   # PCs
BIN_SIZE = 0.2
SMOOTHING = 0.2
T_BEFORE = 0.5
MIN_TRIALS = 10
PRE_TIME = [0.5, 0]  # for modulation index
EARLY_POST_TIME = [0, 0.5]
LATE_POST_TIME = [0.5, 1]
OVERWRITE = True
MIN_NEURONS = 10
subjects = load_subjects()

# Get path
_, save_path = paths()

# Query sessions
rec = query_ephys_sessions(one=one)

if OVERWRITE:
    updown_mod_df = pd.DataFrame()
else:
    updown_mod_df = pd.read_csv(join(save_path, 'updown_states_modulation.csv'))

for i in rec.index.values:

    # Get session details
    pid, eid, probe = rec.loc[i, 'pid'], rec.loc[i, 'eid'], rec.loc[i, 'probe']
    subject, date = rec.loc[i, 'subject'], rec.loc[i, 'date']
    sert_cre = subjects.loc[subjects['subject'] == subject, 'sert-cre'].values[0]
    print(f'\nStarting {subject}, {date} ({i+1} of {len(rec)})')

    if not OVERWRITE:
        if pid in updown_mod_df['pid'].values:
            continue

    # Load in artifact neurons
    artifact_neurons = get_artifact_neurons()

    # Load opto times
    opto_times, _ = load_passive_opto_times(eid, one=one)
    if len(opto_times) == 0:
        continue

    # Load in neural data
    sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
    spikes, clusters, channels = sl.load_spike_sorting()
    clusters = sl.merge_clusters(spikes, clusters, channels)

    # Filter neurons that pass QC and are not artifact neurons
    qc_metrics = get_neuron_qc(pid, one=one, ba=ba)
    clusters_pass = np.where(qc_metrics['label'] == 1)[0]
    clusters_pass = clusters_pass[~np.isin(clusters_pass, artifact_neurons.loc[
        artifact_neurons['pid'] == pid, 'neuron_id'].values)]
    spikes.times = spikes.times[np.isin(spikes.clusters, clusters_pass)]
    spikes.clusters = spikes.clusters[np.isin(spikes.clusters, clusters_pass)]

    # Remap to high level regions
    clusters.regions = high_level_regions(clusters.acronym, merge_cortex=True)

    for j, region in enumerate(np.unique(clusters.regions)):

        # Get spikes in region
        region_spikes = spikes.times[np.isin(spikes.clusters, clusters.cluster_id[clusters.regions == region])]
        region_clusters = spikes.clusters[np.isin(spikes.clusters, clusters.cluster_id[clusters.regions == region])]
        if (np.unique(region_clusters).shape[0] < MIN_NEURONS) | (region == 'root'):
            continue

        # Get smoothed firing rates
        peth, _ = calculate_peths(region_spikes, region_clusters, np.unique(region_clusters),
                                  [opto_times[0]-1], pre_time=0, post_time=(opto_times[-1] - opto_times[0])+1,
                                  bin_size=BIN_SIZE, smoothing=SMOOTHING)
        tscale = peth['tscale'] + (opto_times[0]-1)
        pop_act = peth['means'].T

        # Do PCA
        pca = PCA(n_components=D)
        ss = StandardScaler(with_mean=True, with_std=True)
        pop_vector_norm = ss.fit_transform(pop_act)
        pca_proj = pca.fit_transform(pop_vector_norm)

        # Make an hmm and sample from it
        arhmm = ssm.HMM(K, pca_proj.shape[1], observations="gaussian")
        arhmm.fit(pca_proj)
        zhat = arhmm.most_likely_states(pca_proj)

        # Make sure state 0 is inactive and state 1 active
        if np.mean(np.mean(pop_act[zhat == 0, :], 1)) > np.mean(np.mean(pop_act[zhat == 1, :], 1)):
            zhat = np.where((zhat==0)|(zhat==1), zhat^1, zhat)

        # Get state per stimulation onset
        pre_state = np.empty(opto_times.shape)
        for j, opto_time in enumerate(opto_times):
            pre_zhat = zhat[(tscale > opto_time - T_BEFORE) & (tscale <= opto_time)]
            if np.sum(pre_zhat == 0) > np.sum(pre_zhat == 1):
                pre_state[j] = 0
            else:
                pre_state[j] = 1

        if (np.sum(pre_state == 0) < MIN_TRIALS) or (np.sum(pre_state == 1) < MIN_TRIALS):
            print('Too few trials of one or another state')
            continue

        # Get modulation index for active and inactive state and early and late timewindow
        roc_auc, cluster_ids = roc_single_event(region_spikes, region_clusters,
                                                opto_times[pre_state == 0], pre_time=PRE_TIME,
                                                post_time=EARLY_POST_TIME)
        mod_down_early = 2 * (roc_auc - 0.5)
        roc_auc, cluster_ids = roc_single_event(region_spikes, region_clusters,
                                                opto_times[pre_state == 1], pre_time=PRE_TIME,
                                                post_time=EARLY_POST_TIME)
        mod_up_early = 2 * (roc_auc - 0.5)
        roc_auc, cluster_ids = roc_single_event(region_spikes, region_clusters,
                                                opto_times[pre_state == 0], pre_time=PRE_TIME,
                                                post_time=LATE_POST_TIME)
        mod_down_late = 2 * (roc_auc - 0.5)
        roc_auc, cluster_ids = roc_single_event(region_spikes, region_clusters,
                                                opto_times[pre_state == 1], pre_time=PRE_TIME,
                                                post_time=LATE_POST_TIME)
        mod_up_late = 2 * (roc_auc - 0.5)
        cluster_regions = remap(clusters.acronym[cluster_ids])

        # Add to dataframe
        updown_mod_df = pd.concat((updown_mod_df, pd.DataFrame(data={
            'subject': subject, 'date': date, 'eid': eid, 'pid': pid, 'sert_cre': sert_cre,
            'acronym': cluster_regions, 'high_level_region': region, 'neuron_id': cluster_ids,
            'mod_down_early': mod_down_early, 'mod_up_early': mod_up_early,
            'mod_down_late': mod_down_late, 'mod_up_late': mod_up_late})))

    # Save to disk
    updown_mod_df.to_csv(join(save_path, 'updown_states_modulation.csv'))




