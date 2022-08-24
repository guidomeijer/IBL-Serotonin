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
from dlc_functions import get_dlc_XYs, get_raw_and_smooth_pupil_dia
from serotonin_functions import paths, load_passive_opto_times, get_neuron_qc, query_ephys_sessions
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

# Settings
OVERWRITE = True
PRE_TIME = [0.5, 0]
POST_TIME = [0.5, 1]
_, save_path = paths()

# Query sessions
rec = query_ephys_sessions(one=one)

if OVERWRITE:
    pupil_neurons = pd.DataFrame()
else:
    pupil_neurons = pd.read_csv(join(save_path, 'state_pupil_neurons.csv'))
    rec = rec[~rec['eid'].isin(pupil_neurons['eid'])]

for i in rec.index.values:

    # Get session details
    pid, eid, probe = rec.loc[i, 'pid'], rec.loc[i, 'eid'], rec.loc[i, 'probe']
    subject, date = rec.loc[i, 'subject'], rec.loc[i, 'date']
    print(f'\nStarting {subject}, {date}')

    # Load in laser pulse times
    try:
        opto_train_times, _ = load_passive_opto_times(eid, one=one)
    except:
        continue

    # Load in camera timestamps and DLC output
    try:
        video_times, XYs = get_dlc_XYs(one, eid)
    except:
        print('Could not load video and/or DLC data')
        continue

    # Get pupil diameter
    print('Calculating smoothed pupil trace')
    raw_diameter, diameter = get_raw_and_smooth_pupil_dia(eid, 'left', one)
    diameter_perc = ((diameter - np.percentile(diameter[~np.isnan(diameter)], 2))
                     / np.percentile(diameter[~np.isnan(diameter)], 2)) * 100

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

    # Get pupil size before stimulus onset
    pre_pupil, pre_pupil_size = np.empty(opto_train_times.shape), np.empty(opto_train_times.shape)
    for j, this_time in enumerate(opto_train_times):
        pre_pupil_size[j] = np.nanmean(diameter_perc[(video_times > this_time - PRE_TIME[0])
                                                     & (video_times < this_time - PRE_TIME[1])])
    if np.sum(np.isnan(pre_pupil_size)) == pre_pupil_size.shape[0]:
        continue

    # Split in three
    quantiles = np.percentile(pre_pupil_size[~np.isnan(pre_pupil_size)], [0, 33.3, 66.6, 100])
    pupil_quantiles = np.searchsorted(quantiles, pre_pupil_size)
    pupil_quantiles[pupil_quantiles == 0] = 1
    pre_pupil[pupil_quantiles == 1] = -1
    pre_pupil[pupil_quantiles == 3] = 1
    """
    # Median split
    pre_pupil[pre_pupil_size < np.nanmedian(pre_pupil_size)] = -1
    pre_pupil[pre_pupil_size > np.nanmedian(pre_pupil_size)] = 1
    """
    # Calculate modulation index for optimal and small/large size
    roc_auc, cluster_ids = roc_single_event(spikes.times, spikes.clusters,
                                            opto_train_times[pre_pupil == -1], pre_time=PRE_TIME,
                                            post_time=POST_TIME)
    mod_idx_small = 2 * (roc_auc - 0.5)
    roc_auc, cluster_ids = roc_single_event(spikes.times, spikes.clusters,
                                            opto_train_times[pre_pupil == 1], pre_time=PRE_TIME,
                                            post_time=POST_TIME)
    mod_idx_large = 2 * (roc_auc - 0.5)

    # Add to dataframe
    pupil_neurons = pd.concat((pupil_neurons, pd.DataFrame(data={
        'subject': subject, 'date': date, 'eid': eid, 'probe': probe, 'pid': pid,
        'neuron_id': cluster_ids, 'mod_index_small': mod_idx_small,
        'mod_index_large': mod_idx_large})))
    pupil_neurons.to_csv(join(save_path, 'state_pupil_neurons.csv'), index=False)
    print('Saved output to disk')
pupil_neurons.to_csv(join(save_path, 'state_pupil_neurons.csv'), index=False)

