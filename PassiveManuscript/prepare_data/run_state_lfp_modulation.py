#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 11:15:53 2022
By: Guido Meijer
"""

import numpy as np
import pandas as pd
from os.path import join
from brainbox.task.closed_loop import roc_single_event
from brainbox.io.one import SpikeSortingLoader
from scipy.signal import welch
from dlc_functions import get_dlc_XYs, get_raw_and_smooth_pupil_dia
from serotonin_functions import (paths, load_passive_opto_times, get_neuron_qc, query_ephys_sessions,
                                 load_lfp)
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

# Settings
PRE_TIME = [0.5, 0]
POST_TIME = [0.5, 1]
WINDOW_SIZE = 1024
FREQS = dict()
FREQS['alpha'] = [8, 12]
FREQS['beta'] = [15, 30]
FREQS['gamma'] = [30, 100]
_, save_path = paths()

# Query sessions
rec = query_ephys_sessions(one=one)
lfp_neurons = pd.DataFrame()
for i in rec.index.values:

    # Get session details
    pid, eid, probe = rec.loc[i, 'pid'], rec.loc[i, 'eid'], rec.loc[i, 'probe']
    subject, date = rec.loc[i, 'subject'], rec.loc[i, 'date']
    print(f'\nStarting {subject}, {date}')

    # Load in LFP and laser pulses
    try:
        opto_train_times, _ = load_passive_opto_times(eid, one=one)
        lfp, time = load_lfp(eid, probe, time_start=opto_train_times[0]-10,
                             time_end=opto_train_times[-1]+10,
                             relative_to='begin', one=one)
    except:
        continue

    # Load in spikes
    try:
        sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
        spikes, clusters, channels = sl.load_spike_sorting()
        clusters = sl.merge_clusters(spikes, clusters, channels)
    except Exception as err:
        print(err)
        continue

    # Filter neurons that pass QC
    qc_metrics = get_neuron_qc(pid, one=one, ba=ba)
    clusters_pass = np.where(qc_metrics['label'] == 1)[0]
    spikes.times = spikes.times[np.isin(spikes.clusters, clusters_pass)]
    spikes.clusters = spikes.clusters[np.isin(spikes.clusters, clusters_pass)]
    if len(spikes.clusters) == 0:
        continue

    # Get LFP power before stimulus onset
    for ff, freq_band in enumerate(FREQS.keys()):
        pre_lfp, pre_lfp_power = np.empty(opto_train_times.shape), np.empty(opto_train_times.shape)
        for p, pulse_start in enumerate(opto_train_times):

            # Get baseline LFP power
            freq, pwr = welch(lfp[:, (time >= pulse_start - PRE_TIME[0]) & (time <= pulse_start - PRE_TIME[1])],
                              fs=2500, window='hann', nperseg=WINDOW_SIZE)

            # Select frequencies of interest
            pre_lfp_power[p] = np.median(pwr[:, (freq >= FREQS[freq_band][0]) & (freq <= FREQS[freq_band][1])])
        pre_lfp[pre_lfp_power < np.nanmedian(pre_lfp_power)] = -1
        pre_lfp[pre_lfp_power > np.nanmedian(pre_lfp_power)] = 1

        # Calculate modulation index for small and large pupil size
        roc_auc, cluster_ids = roc_single_event(spikes.times, spikes.clusters,
                                                opto_train_times[pre_lfp == -1], pre_time=PRE_TIME,
                                                post_time=POST_TIME)
        mod_idx_low = 2 * (roc_auc - 0.5)
        roc_auc, cluster_ids = roc_single_event(spikes.times, spikes.clusters,
                                                opto_train_times[pre_lfp == 1], pre_time=PRE_TIME,
                                                post_time=POST_TIME)
        mod_idx_high = 2 * (roc_auc - 0.5)

        # Add to dataframe
        lfp_neurons = pd.concat((lfp_neurons, pd.DataFrame(data={
            'subject': subject, 'date': date, 'eid': eid, 'probe': probe, 'pid': pid,
            'neuron_id': cluster_ids, 'mod_index_low': mod_idx_low,
            'mod_index_high': mod_idx_high, 'freq_band': freq_band})))

    lfp_neurons.to_csv(join(save_path, 'state_lfp_neurons.csv'), index=False)
    print('Saved output to disk')
lfp_neurons.to_csv(join(save_path, 'state_lfp_neurons.csv'), index=False)

