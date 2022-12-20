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
from glob import glob
import pandas as pd
from brainbox.io.one import SpikeSortingLoader
from serotonin_functions import (figure_style, load_passive_opto_times, get_neuron_qc, remap, paths,
                                 query_ephys_sessions, make_bins)
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

K = 2    # number of discrete states
D = 3
T_BEFORE = 0  # for state classification
T_AFTER = 0.5
PRE_TIME = [0.5, 0]  # for modulation index
POST_TIME = [0, 0.5]
OVERWRITE = True

# Get path
_, save_path = paths()

# Query sessions
rec = query_ephys_sessions(one=one)

if OVERWRITE:
    state_mod_df = pd.DataFrame()
else:
    state_mod_df = pd.read_csv(join(save_path, 'mov_state_mod.csv'))

for i in rec.index.values:

    # Get session details
    pid, eid, probe = rec.loc[i, 'pid'], rec.loc[i, 'eid'], rec.loc[i, 'probe']
    subject, date = rec.loc[i, 'subject'], rec.loc[i, 'date']
    if not OVERWRITE:
        if eid in state_mod_df['eid'].values:
            continue
    print(f'\nStarting {subject}, {date} ({i+1} of {len(rec)})')

    # Load in video data
    try:
        left_times = one.load_dataset(eid, '_ibl_leftCamera.times.npy')
        right_times = one.load_dataset(eid, '_ibl_rightCamera.times.npy')
        body_times = one.load_dataset(eid, '_ibl_bodyCamera.times.npy')
        left_motion = one.load_dataset(eid, 'leftCamera.ROIMotionEnergy.npy')
        right_motion = one.load_dataset(eid, 'rightCamera.ROIMotionEnergy.npy')
        body_motion = one.load_dataset(eid, 'bodyCamera.ROIMotionEnergy.npy')
    except Exception as err:
        print(err)
        continue

    # Load opto times
    opto_times, _ = load_passive_opto_times(eid, one=one)
    if len(opto_times) == 0:
        continue

    # Select part of recording starting just before opto onset
    left_motion = left_motion[left_times > opto_times[0] - 10]
    left_times = left_times[left_times > opto_times[0] - 10]
    right_motion = right_motion[right_times > opto_times[0] - 10]
    right_times = right_times[right_times > opto_times[0] - 10]
    body_motion = body_motion[body_times > opto_times[0] - 10]
    body_times = body_times[body_times > opto_times[0] - 10]

    # TO DO: cameras have different sampling rates, do some binning
    make_bins(left_motion, left_times, left_times[0], left_times[-1], 0.1)

    # Make an hmm and sample from it
    arhmm = ssm.HMM(K, D, observations="ar")
    arhmm.fit(motSVD[:, :D])
    zhat = arhmm.most_likely_states(motSVD[:, :D])

    # Make sure state 0 is inactive and state 1 active
    if np.mean(motSVD[zhat == 0, 0]) > np.mean(motSVD[zhat == 1, 0]):
        zhat = np.where((zhat==0)|(zhat==1), zhat^1, zhat)

    # Get state per stimulation onset
    pre_state = np.empty(opto_times.shape)
    for j, opto_time in enumerate(opto_times):
        pre_zhat = zhat[(fm_times > opto_time - T_BEFORE) & (fm_times <= opto_time + T_AFTER)]
        if np.sum(pre_zhat == 0) > np.sum(pre_zhat == 1):
            pre_state[j] = 0
        else:
            pre_state[j] = 1

    # Loop over probes
    insertions = one.alyx.rest('insertions', 'list', session=eid)
    pids = [i['id'] for i in insertions]
    for pid in pids:
        if pid not in rec['pid'].values:
            continue

        # Load in spikes
        sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
        spikes, clusters, channels = sl.load_spike_sorting()
        clusters = sl.merge_clusters(spikes, clusters, channels)

        # Filter neurons that pass QC
        qc_metrics = get_neuron_qc(pid, one=one, ba=ba)
        clusters_pass = np.where(qc_metrics['label'] == 1)[0]
        spikes.times = spikes.times[np.isin(spikes.clusters, clusters_pass)]
        spikes.clusters = spikes.clusters[np.isin(spikes.clusters, clusters_pass)]

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
            'mod_index_inactive': mod_idx_inactive, 'mod_index_active': mod_idx_active})))

    # Save to disk
    state_mod_df.to_csv(join(save_path, 'mov_state_mod.csv'))




