#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 11:03:49 2022
By: Guido Meijer
"""

import numpy as np
import seaborn as sns
from os.path import join
import ssm
from ssm.plots import gradient_cmap
import matplotlib.pyplot as plt
from brainbox.task.closed_loop import roc_single_event
import pandas as pd
from brainbox.singlecell import calculate_peths
from brainbox.io.one import SpikeSortingLoader
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from serotonin_functions import (figure_style, load_passive_opto_times, get_neuron_qc, remap, paths,
                                 query_ephys_sessions)
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

K = 2    # number of discrete states
D = 10   # PCs
BIN_SIZE = 0.2
SMOOTHING = 0.1
T_BEFORE = 0.5
MIN_TRIALS = 10
PRE_TIME = [0.5, 0]  # for modulation index
POST_TIME = [0, 0.5]
OVERWRITE = False

# Get path
_, save_path = paths()

# Query sessions
rec = query_ephys_sessions(one=one)

if OVERWRITE:
    state_mod_df = pd.DataFrame()
else:
    state_mod_df = pd.read_csv(join(save_path, 'neural_state_modulation.csv'))

for i in rec.index.values:

    # Get session details
    pid, eid, probe = rec.loc[i, 'pid'], rec.loc[i, 'eid'], rec.loc[i, 'probe']
    subject, date = rec.loc[i, 'subject'], rec.loc[i, 'date']
    print(f'\nStarting {subject}, {date} ({i+1} of {len(rec)})')

    if not OVERWRITE:
        if pid in state_mod_df['pid'].values:
            continue

    # Load opto times
    try:
        opto_times, _ = load_passive_opto_times(eid, one=one)
    except:
        continue

    # Load in neural data
    sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
    spikes, clusters, channels = sl.load_spike_sorting()
    clusters = sl.merge_clusters(spikes, clusters, channels)

    # Filter neurons that pass QC
    qc_metrics = get_neuron_qc(pid, one=one, ba=ba)
    clusters_pass = np.where(qc_metrics['label'] == 1)[0]
    spikes.times = spikes.times[np.isin(spikes.clusters, clusters_pass)]
    spikes.clusters = spikes.clusters[np.isin(spikes.clusters, clusters_pass)]

    # Get smoothed firing rates
    peth, _ = calculate_peths(spikes.times, spikes.clusters, np.unique(spikes.clusters),
                              [opto_times[0]-1], pre_time=0, post_time=(opto_times[-1] - opto_times[0])+1,
                              bin_size=BIN_SIZE, smoothing=SMOOTHING)
    tscale = peth['tscale'] + (opto_times[0]-1)

    # Do PCA
    pca = PCA(n_components=D)
    ss = StandardScaler(with_mean=True, with_std=True)
    pop_vector_norm = ss.fit_transform(peth['means'].T)
    pca_proj = pca.fit_transform(pop_vector_norm)

    # Make an hmm and sample from it
    arhmm = ssm.HMM(K, D, observations="ar")
    arhmm.fit(pca_proj)
    zhat = arhmm.most_likely_states(pca_proj)

    # Make sure state 0 is inactive and state 1 active
    if np.mean(pca_proj[zhat == 0, 0]) > np.mean(pca_proj[zhat == 1, 0]):
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

    # Get modulation index for active and inactive state
    roc_auc, cluster_ids = roc_single_event(spikes.times, spikes.clusters,
                                            opto_times[pre_state == 0], pre_time=PRE_TIME,
                                            post_time=POST_TIME)
    mod_idx_inactive = 2 * (roc_auc - 0.5)
    roc_auc, cluster_ids = roc_single_event(spikes.times, spikes.clusters,
                                            opto_times[pre_state == 1], pre_time=PRE_TIME,
                                            post_time=POST_TIME)
    mod_idx_active = 2 * (roc_auc - 0.5)
    cluster_regions = remap(clusters.acronym[cluster_ids])

    # Add to dataframe
    state_mod_df = pd.concat((state_mod_df, pd.DataFrame(data={
        'subject': subject, 'date': date, 'eid': eid, 'pid': pid,
        'region': cluster_regions, 'neuron_id': cluster_ids,
        'mod_index_low': mod_idx_inactive, 'mod_index_high': mod_idx_active})))

    # Save to disk
    state_mod_df.to_csv(join(save_path, 'neural_state_modulation.csv'))




