#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 14:52:05 2022
By: Guido Meijer
"""

import numpy as np
from os.path import join
import ssm
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from brainbox.singlecell import calculate_peths
from brainbox.plot import peri_event_time_histogram
from brainbox.io.one import SpikeSortingLoader
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from serotonin_functions import (load_passive_opto_times, get_neuron_qc, paths, query_ephys_sessions,
                                 figure_style, load_subjects, high_level_regions, remap)
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

# Settings
K = 2    # number of discrete states
BIN_SIZE = 0.2
SMOOTHING = 0.2
do_PCA = True
D = 5   # dimensions of PCA
OVERWRITE = True
T_BEFORE = 1  # for PSTH
T_AFTER = 4
PLOT = True
MIN_NEURONS = 5

# Get path
fig_path, save_path = paths()

# Query sessions
rec_both = query_ephys_sessions(anesthesia='both', one=one)
rec_both['anesthesia'] = 'both'
rec_anes = query_ephys_sessions(anesthesia='yes', one=one)
rec_anes['anesthesia'] = 'yes'
rec = pd.concat((rec_both, rec_anes)).reset_index(drop=True)
subjects = load_subjects()

if OVERWRITE:
    state_trans_df = pd.DataFrame()
    up_down_state_df = pd.DataFrame()
else:
    up_down_state_df = pd.read_csv(join(save_path, 'updown_state_anesthesia.csv'))
    state_trans_df = pd.read_csv(join(save_path, 'updown_state_trans_anesthesia.csv'))

for i in rec.index.values:

    # Get session details
    pid, eid, probe = rec.loc[i, 'pid'], rec.loc[i, 'eid'], rec.loc[i, 'probe']
    subject, date = rec.loc[i, 'subject'], rec.loc[i, 'date']
    sert_cre = subjects.loc[subjects['subject'] == subject, 'sert-cre'].values[0]
    print(f'\nStarting {subject}, {date}, {probe} ({i+1} of {len(rec)})')

    if not OVERWRITE:
        if pid in up_down_state_df['pid'].values:
            continue

    # Load opto times
    if rec.loc[i, 'anesthesia'] == 'both':
        opto_times, _ = load_passive_opto_times(eid, anesthesia=True, one=one)
    elif rec.loc[i, 'anesthesia'] == 'yes':
        opto_times, _ = load_passive_opto_times(eid, one=one)

    # Load in neural data
    sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
    spikes, clusters, channels = sl.load_spike_sorting()
    clusters = sl.merge_clusters(spikes, clusters, channels)

    # Filter neurons that pass QC
    qc_metrics = get_neuron_qc(pid, one=one, ba=ba)
    clusters_pass = np.where(qc_metrics['label'] == 1)[0]
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
                                  [opto_times[0]-300], pre_time=0, post_time=(opto_times[-1] - opto_times[0])+301,
                                  bin_size=BIN_SIZE, smoothing=SMOOTHING)
        tscale = peth['tscale'] + (opto_times[0]-300)
        pop_act = peth['means'].T

        # Do PCA
        if do_PCA:
            pca = PCA(n_components=D)
            ss = StandardScaler(with_mean=True, with_std=True)
            pop_vector_norm = ss.fit_transform(pop_act)
            pca_proj = pca.fit_transform(pop_vector_norm)

            # Make an hmm and sample from it
            arhmm = ssm.HMM(K, pca_proj.shape[1], observations="gaussian")
            arhmm.fit(pca_proj)
            zhat = arhmm.most_likely_states(pca_proj)

        else:
            # Make an hmm and sample from it
            arhmm = ssm.HMM(K, pop_act.shape[1], observations="gaussian")
            arhmm.fit(pop_act)
            zhat = arhmm.most_likely_states(pop_act)

            # Make an hmm and sample from it
            arhmm = ssm.HMM(K, pop_act.shape[1], observations="gaussian")
            arhmm.fit(pop_act)
            zhat = arhmm.most_likely_states(pop_act)

        # Make sure state 0 is inactive and state 1 active
        if np.mean(np.mean(pop_act[zhat == 0, :], 1)) > np.mean(np.mean(pop_act[zhat == 1, :], 1)):
            zhat = np.where((zhat==0)|(zhat==1), zhat^1, zhat)

        # Get state change times
        to_down = tscale[np.concatenate((np.zeros(1), np.diff(zhat))) == -1]
        to_up = tscale[np.concatenate((np.zeros(1), np.diff(zhat))) == 1]
        state_change_times = tscale[np.concatenate((np.zeros(1), np.diff(zhat))) != 0]
        state_change_id = np.diff(zhat)[np.diff(zhat) != 0]

        # Get state change PETH
        to_down_peths, _ = calculate_peths(to_down, np.ones(to_down.shape[0]), [1],
                                           opto_times, T_BEFORE, T_AFTER, BIN_SIZE, SMOOTHING)
        to_up_peths, _ = calculate_peths(to_up, np.ones(to_up.shape[0]), [1],
                                         opto_times, T_BEFORE, T_AFTER, BIN_SIZE, SMOOTHING)

        # Add to df
        state_trans_df = pd.concat((state_trans_df, pd.DataFrame(data={
            'subject': subject, 'date': date, 'eid': eid, 'pid': pid, 'sert-cre': sert_cre, 'region': region,
            'to_down_peths': to_down_peths['means'][0], 'to_up_peths': to_up_peths['means'][0],
            'time': to_down_peths['tscale']})))

        up_down_state_df = pd.concat((up_down_state_df, pd.DataFrame(data={
            'subject': subject, 'date': date, 'eid': eid, 'pid': pid, 'sert-cre': sert_cre, 'region': region,
            'state': zhat, 'time': tscale, 'opto': tscale >= opto_times[0]})))

        if PLOT:
            colors, dpi = figure_style()
            f, ax = plt.subplots(1, 1, figsize=(2, 2), dpi=dpi)
            ax.add_patch(Rectangle((0, 0), 1, 100, color='royalblue', alpha=0.25, lw=0))
            ax.add_patch(Rectangle((0, 0), 1, -100, color='royalblue', alpha=0.25, lw=0))
            peri_event_time_histogram(to_down, np.ones(to_down.shape[0]), opto_times, 1,
                                      t_before=T_BEFORE, t_after=T_AFTER, bin_size=BIN_SIZE,
                                      smoothing=SMOOTHING, include_raster=True,
                                      error_bars='sem', pethline_kwargs={'color': colors['suppressed'], 'lw': 1},
                                      errbar_kwargs={'color': colors['suppressed'], 'alpha': 0.3},
                                      raster_kwargs={'color': colors['suppressed'], 'lw': 0.5},
                                      eventline_kwargs={'lw': 0}, ax=ax)
            peri_event_time_histogram(to_up, np.ones(to_up.shape[0]), opto_times, 1,
                                      t_before=T_BEFORE, t_after=T_AFTER, bin_size=BIN_SIZE,
                                      smoothing=SMOOTHING, include_raster=True,
                                      error_bars='sem', pethline_kwargs={'color': colors['enhanced'], 'lw': 1},
                                      errbar_kwargs={'color': colors['enhanced'], 'alpha': 0.3},
                                      raster_kwargs={'color': colors['enhanced'], 'lw': 0.5},
                                      eventline_kwargs={'lw': 0}, ax=ax)
            ax.set(ylabel='State change rate (changes/s)', xlabel='Time (s)',
                   yticks=[0, 0.4, 0.8, 1.2], xticks=[-1, 0, 1, 2, 3, 4])
            # ax.plot([0, 1], [0, 0], lw=2.5, color='royalblue')
            plt.tight_layout()
            plt.savefig(join(fig_path, 'Ephys', 'UpDownStates', 'Anesthesia',
                             f'{region}_{subject}_{date}_{probe}.jpg'), dpi=600)
            plt.close(f)

    # Save data
    state_trans_df.to_csv(join(save_path, 'updown_state_trans_anesthesia.csv'))
    up_down_state_df.to_csv(join(save_path, 'updown_state_anesthesia.csv'))
    print('Saved results to disk')