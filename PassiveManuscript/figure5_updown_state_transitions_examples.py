#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 15:36:47 2022
By: Guido Meijer
"""


import numpy as np
from os.path import join
import pandas as pd
import ssm
import matplotlib.pyplot as plt
import seaborn as sns
from brainbox.plot import peri_event_time_histogram
from matplotlib.patches import Rectangle
from brainbox.singlecell import calculate_peths
from brainbox.io.one import SpikeSortingLoader
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from serotonin_functions import (paths, figure_style, load_passive_opto_times, high_level_regions,
                                 get_neuron_qc)
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

# Settings
CORTEX_PID = '36f8f321-c21b-4f2d-876b-c289ce3f8bed'
STR_PID = '04954136-75a8-4a20-9054-37b0bffd3b8b'
MB_PID = 'aa89d1e8-51da-40f6-a215-c9703a6ceb30'
K = 2    # number of discrete states
BIN_SIZE = 0.2
SMOOTHING = 0.2
D = 10   # dimensions of PCA
T_BEFORE = 1  # for PSTH
T_AFTER = 4

# Get paths
fig_path, save_path = paths(dropbox=True)
fig_path = join(fig_path, 'PaperPassive', 'figure5')

# %% Cortex example

# Load opto times
opto_times, _ = load_passive_opto_times(one.pid2eid(CORTEX_PID)[0], anesthesia=True, one=one)

# Load in neural data
sl = SpikeSortingLoader(pid=CORTEX_PID, one=one, atlas=ba)
spikes, clusters, channels = sl.load_spike_sorting()
clusters = sl.merge_clusters(spikes, clusters, channels)

# Filter neurons that pass QC
qc_metrics = get_neuron_qc(CORTEX_PID, one=one, ba=ba)
clusters_pass = np.where(qc_metrics['label'] == 1)[0]
spikes.times = spikes.times[np.isin(spikes.clusters, clusters_pass)]
spikes.clusters = spikes.clusters[np.isin(spikes.clusters, clusters_pass)]

# Remap to high level regions
clusters.regions = high_level_regions(clusters.acronym, merge_cortex=True)

# Get spikes in region
region_spikes = spikes.times[np.isin(spikes.clusters, clusters.cluster_id[clusters.regions == 'Cortex'])]
region_clusters = spikes.clusters[np.isin(spikes.clusters, clusters.cluster_id[clusters.regions == 'Cortex'])]

# Get smoothed firing rates
peth, _ = calculate_peths(region_spikes, region_clusters, np.unique(region_clusters),
                          [opto_times[0]-300], pre_time=0, post_time=(opto_times[-1] - opto_times[0])+301,
                          bin_size=BIN_SIZE, smoothing=SMOOTHING)
tscale = peth['tscale'] + (opto_times[0]-300)
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

# Get state change times
to_down = tscale[np.concatenate((np.zeros(1), np.diff(zhat))) == -1]
to_up = tscale[np.concatenate((np.zeros(1), np.diff(zhat))) == 1]
state_change_times = tscale[np.concatenate((np.zeros(1), np.diff(zhat))) != 0]
state_change_id = np.diff(zhat)[np.diff(zhat) != 0]

# Plot
colors, dpi = figure_style()
f, ax = plt.subplots(1, 1, figsize=(1.5, 1.75), dpi=dpi)
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
       yticks=[0, 1], xticks=[-1, 0, 1, 2, 3, 4], title='Cortex')
# ax.plot([0, 1], [0, 0], lw=2.5, color='royalblue')
plt.tight_layout()
plt.savefig(join(fig_path, 'cortex_example_updown_transitions.pdf'))
plt.savefig(join(fig_path, 'cortex_example_updown_transitions.jpg'), dpi=600)

# %% Striatum example

# Load opto times
opto_times, _ = load_passive_opto_times(one.pid2eid(STR_PID)[0], anesthesia=True, one=one)

# Load in neural data
sl = SpikeSortingLoader(pid=STR_PID, one=one, atlas=ba)
spikes, clusters, channels = sl.load_spike_sorting()
clusters = sl.merge_clusters(spikes, clusters, channels)

# Filter neurons that pass QC
qc_metrics = get_neuron_qc(STR_PID, one=one, ba=ba)
clusters_pass = np.where(qc_metrics['label'] == 1)[0]
spikes.times = spikes.times[np.isin(spikes.clusters, clusters_pass)]
spikes.clusters = spikes.clusters[np.isin(spikes.clusters, clusters_pass)]

# Remap to high level regions
clusters.regions = high_level_regions(clusters.acronym, merge_cortex=True)

# Get spikes in region
region_spikes = spikes.times[np.isin(spikes.clusters, clusters.cluster_id[clusters.regions == 'Striatum'])]
region_clusters = spikes.clusters[np.isin(spikes.clusters, clusters.cluster_id[clusters.regions == 'Striatum'])]

# Get smoothed firing rates
peth, _ = calculate_peths(region_spikes, region_clusters, np.unique(region_clusters),
                          [opto_times[0]-300], pre_time=0, post_time=(opto_times[-1] - opto_times[0])+301,
                          bin_size=BIN_SIZE, smoothing=SMOOTHING)
tscale = peth['tscale'] + (opto_times[0]-300)
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

# Get state change times
to_down = tscale[np.concatenate((np.zeros(1), np.diff(zhat))) == -1]
to_up = tscale[np.concatenate((np.zeros(1), np.diff(zhat))) == 1]
state_change_times = tscale[np.concatenate((np.zeros(1), np.diff(zhat))) != 0]
state_change_id = np.diff(zhat)[np.diff(zhat) != 0]

# Plot
colors, dpi = figure_style()
f, ax = plt.subplots(1, 1, figsize=(1.5, 1.75), dpi=dpi)
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
       yticks=[0, 1], xticks=[-1, 0, 1, 2, 3, 4], title='Striatum')
# ax.plot([0, 1], [0, 0], lw=2.5, color='royalblue')
plt.tight_layout()
plt.savefig(join(fig_path, 'striatum_example_updown_transitions.pdf'))
plt.savefig(join(fig_path, 'striatum_example_updown_transitions.jpg'), dpi=600)

# %% Hippocampus example

# Load opto times
opto_times, _ = load_passive_opto_times(one.pid2eid(MB_PID)[0], anesthesia=True, one=one)

# Load in neural data
sl = SpikeSortingLoader(pid=MB_PID, one=one, atlas=ba)
spikes, clusters, channels = sl.load_spike_sorting()
clusters = sl.merge_clusters(spikes, clusters, channels)

# Filter neurons that pass QC
qc_metrics = get_neuron_qc(MB_PID, one=one, ba=ba)
clusters_pass = np.where(qc_metrics['label'] == 1)[0]
spikes.times = spikes.times[np.isin(spikes.clusters, clusters_pass)]
spikes.clusters = spikes.clusters[np.isin(spikes.clusters, clusters_pass)]

# Remap to high level regions
clusters.regions = high_level_regions(clusters.acronym, merge_cortex=True)

# Get spikes in region
region_spikes = spikes.times[np.isin(spikes.clusters, clusters.cluster_id[clusters.regions == 'Midbrain'])]
region_clusters = spikes.clusters[np.isin(spikes.clusters, clusters.cluster_id[clusters.regions == 'Midbrain'])]

# Get smoothed firing rates
peth, _ = calculate_peths(region_spikes, region_clusters, np.unique(region_clusters),
                          [opto_times[0]-300], pre_time=0, post_time=(opto_times[-1] - opto_times[0])+301,
                          bin_size=BIN_SIZE, smoothing=SMOOTHING)
tscale = peth['tscale'] + (opto_times[0]-300)
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

# Get state change times
to_down = tscale[np.concatenate((np.zeros(1), np.diff(zhat))) == -1]
to_up = tscale[np.concatenate((np.zeros(1), np.diff(zhat))) == 1]
state_change_times = tscale[np.concatenate((np.zeros(1), np.diff(zhat))) != 0]
state_change_id = np.diff(zhat)[np.diff(zhat) != 0]

# Plot
colors, dpi = figure_style()
f, ax = plt.subplots(1, 1, figsize=(1.5, 1.75), dpi=dpi)
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
       yticks=[0, 1], xticks=[-1, 0, 1, 2, 3, 4], title='Midbrain')
# ax.plot([0, 1], [0, 0], lw=2.5, color='royalblue')
plt.tight_layout()
plt.savefig(join(fig_path, 'midbrain_example_updown_transitions.pdf'))
plt.savefig(join(fig_path, 'midbrain_example_updown_transitions.jpg'), dpi=600)