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
                                 query_ephys_sessions)
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

K = 2    # number of discrete states
D = 25   # dimension of the observations
T_BEFORE = 0  # for state classification
T_AFTER = 0.5
PRE_TIME = [0.5, 0]  # for modulation index
POST_TIME = [0, 0.5]
FM_DIR = '/media/guido/Data2/Facemap/'  # dir with facemap data
OVERWRITE = True

# Get path
_, save_path = paths()

# Get all processed facemap files
fm_files = glob(join(FM_DIR, '*_proc.npy'))

# Query sessions
rec = query_ephys_sessions(one=one)

if OVERWRITE:
    state_mod_df = pd.DataFrame()
else:
    state_mod_df = pd.read_csv(join(save_path, 'state_modulation.csv'))

for i, path in enumerate(fm_files):

    # Get session data
    subject = path[-40:-31]
    date = path[-30:-20]
    try:
        eid = one.search(subject=subject, date_range=date)[0]
    except:
        continue

    if not OVERWRITE:
        if eid in state_mod_df['eid'].values:
            continue

    print(f'Starting {subject}, {date}')

    # Load in timestamp data
    try:
        times = one.load_dataset(eid, '_ibl_leftCamera.times.npy')
    except:
        continue

    # Load in facemap data
    fm_dict = np.load(path, allow_pickle=True).item()

    # Facemap data is the last part of the video
    fm_times = times[times.shape[0] - fm_dict['motSVD'][1].shape[0]:]

    # Load opto times
    try:
        opto_times, _ = load_passive_opto_times(eid, one=one)
    except:
        continue

    # Select part of recording starting just before opto onset
    motSVD = fm_dict['motSVD'][1][fm_times > opto_times[0] - 10, :D]
    fm_times = fm_times[fm_times > opto_times[0] - 10]

    if fm_times.shape[0] == 0:
        continue

    if np.sum(fm_times[-1] > opto_times) != opto_times.shape[0]:
        print('Mismatch!')
        continue

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
    state_mod_df.to_csv(join(save_path, 'state_modulation.csv'))




