#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 14:46:00 2022
By: Guido Meijer
"""

import numpy as np
from one.api import ONE
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from brainbox.singlecell import calculate_peths
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
import ssm
from os.path import join
from serotonin_functions import (load_passive_opto_times, paths, get_artifact_neurons,
                                 query_ephys_sessions, figure_style, get_neuron_qc)
from brainbox.processing import bincount2D
from ibllib.atlas import AllenAtlas
one = ONE()
ba = AllenAtlas()

# Settings
K = 2    # number of discrete states
BIN_SIZE = 0.2
SMOOTHING = 0
D = 5   # dimensions of PCA

# Paths
fig_path, save_path = paths(dropbox=True)
fig_path = join(fig_path, 'PaperPassive', 'figure5')

# Recording
SUBJECT = 'ZFM-04820'
DATE = '2022-09-16'
PROBE = 'probe00'

# Query sessions
rec = query_ephys_sessions(anesthesia='both', one=one)
pid = rec.loc[(rec['subject'] == SUBJECT) & (rec['date'] == DATE) & (rec['probe'] == PROBE), 'pid'].values[0]
eid = rec.loc[(rec['subject'] == SUBJECT) & (rec['date'] == DATE) & (rec['probe'] == PROBE), 'eid'].values[0]

# Load in artifact neurons
artifact_neurons = get_artifact_neurons()

# Load in opto times
opto_times_awake, _ = load_passive_opto_times(eid, one=one)
opto_times_anes, _ = load_passive_opto_times(eid, anesthesia=True, one=one)

# Load in spikes
spikes_times = one.load_dataset(eid, 'spikes.times.npy', collection=f'alf/{PROBE}/pykilosort')
spikes_clusters = one.load_dataset(eid, 'spikes.clusters.npy', collection=f'alf/{PROBE}/pykilosort')
spikes_depths = one.load_dataset(eid, 'spikes.depths.npy', collection=f'alf/{PROBE}/pykilosort')

qc_metrics = get_neuron_qc(pid, one=one, ba=ba)
clusters_pass = np.where(qc_metrics['label'] == 1)[0]
spikes_times = spikes_times[np.isin(spikes_clusters, clusters_pass)]
spikes_depths = spikes_depths[np.isin(spikes_clusters, clusters_pass)]
spikes_clusters = spikes_clusters[np.isin(spikes_clusters, clusters_pass)]

# Convert to mm
spikes_depths = spikes_depths / 1000

# Get spike raster
R, times, depths = bincount2D(spikes_times, spikes_depths, xbin=0.01, ybin=0.02, weights=None)

# Get smoothed firing rates
peth, _ = calculate_peths(spikes_times, spikes_clusters, np.unique(spikes_clusters),
                          [opto_times_anes[0]], pre_time=0, post_time=150,
                          bin_size=BIN_SIZE, smoothing=SMOOTHING)
tscale = peth['tscale'] - peth['tscale'][0]
time_ax = peth['tscale'] + opto_times_anes[0]
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


# %% Plot figure
colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(2, 1.75), sharey=True, dpi=dpi)

for i in range(opto_times_anes.shape[0]):
    ax1.add_patch(Rectangle((opto_times_awake[i], 0), 1, 4, color='royalblue', alpha=0.25, lw=0))
ax1.imshow(R, aspect='auto', cmap='binary', vmin=0, vmax=np.std(R),
           extent=np.r_[times[[0, -1]], depths[[0, -1]]], origin='upper')
ax1.set(xlim=[opto_times_awake[0] - 5, opto_times_awake[0] + 30], ylim=[0, 4], ylabel='Depth (mm)')
#ax1.set_title('Awake', color=colors['awake'], fontweight='bold')
ax1.set(xticks=[ax1.get_xlim()[0] + 1, ax1.get_xlim()[0] + 6])
ax1.text(ax1.get_xlim()[0] + 3.5, 4.4, '5s', ha='center', va='center')
ax1.axes.get_xaxis().set_visible(False)
ax1.invert_yaxis()

plt.tight_layout()
sns.despine(trim=True, offset=2)
plt.savefig(join(fig_path, 'raster_awake.pdf'))


# %% Raster anesthesia
f, ax1 = plt.subplots(1, 1, figsize=(3, 1.75), sharey=True, dpi=dpi)
for i in range(opto_times_anes.shape[0]):
    ax1.add_patch(Rectangle((opto_times_anes[i], 0), 1, 4, color='royalblue', alpha=0.25, lw=0))
ax1.imshow(R, aspect='auto', cmap='binary', vmin=0, vmax=np.std(R),
           extent=np.r_[times[[0, -1]], depths[[0, -1]]], origin='upper')
ax1.set(xlim=[opto_times_anes[6] - 5, opto_times_anes[6] + 40], ylim=[0, 4])
#ax1.set_title('Anesthesia', color=colors['anesthesia'], fontweight='bold')
ax1.set(xticks=[ax1.get_xlim()[0] + 1, ax1.get_xlim()[0] + 6], ylabel='Depth (mm)')
ax1.text(ax1.get_xlim()[0] + 3.5, 4.4, '5s', ha='center', va='center')
ax1.axes.get_xaxis().set_visible(False)
ax1.invert_yaxis()

plt.tight_layout()
sns.despine(trim=True, offset=2)
plt.savefig(join(fig_path, 'raster_anesthesia.pdf'))
plt.savefig(join(fig_path, 'raster_anesthesia.jpg'), dpi=600)

# %% Raster anesthesia with states
f, ax1 = plt.subplots(1, 1, figsize=(3, 1.75), sharey=True, dpi=dpi)
cmap = ListedColormap([colors['suppressed'], colors['enhanced']])
ax1.imshow(R, aspect='auto', cmap='binary', vmin=0, vmax=np.std(R),
           extent=np.r_[times[[0, -1]], depths[[0, -1]]], origin='upper')
ax1.imshow(zhat[None,:], aspect="auto", extent=[time_ax[0], time_ax[-1], 4, 0],
           cmap=cmap, vmin=0, vmax=K-1, alpha=0.5)
ax1.set(xlim=[opto_times_anes[6] - 5, opto_times_anes[6] + 40], ylim=[0, 4])
#ax1.set_title('Anesthesia', color=colors['anesthesia'], fontweight='bold')
ax1.set(xticks=[ax1.get_xlim()[0] + 1, ax1.get_xlim()[0] + 6], ylabel='Depth (mm)')
ax1.text(ax1.get_xlim()[0] + 3.5, 4.4, '5s', ha='center', va='center')
ax1.axes.get_xaxis().set_visible(False)
ax1.invert_yaxis()

plt.tight_layout()
sns.despine(trim=True, offset=2)
plt.savefig(join(fig_path, 'raster_anesthesia.pdf'))
plt.savefig(join(fig_path, 'raster_anesthesia.jpg'), dpi=600)

# %% Raster anesthesia no light
f, ax1 = plt.subplots(1, 1, figsize=(3, 1.75), sharey=True, dpi=dpi)
ax1.imshow(R, aspect='auto', cmap='binary', vmin=0, vmax=np.std(R),
           extent=np.r_[times[[0, -1]], depths[[0, -1]]], origin='upper')
ax1.set(xlim=[opto_times_anes[6] - 5, opto_times_anes[6] + 40], ylim=[0, 4])
#ax1.set_title('Anesthesia', color=colors['anesthesia'], fontweight='bold')
ax1.set(xticks=[ax1.get_xlim()[0] + 1, ax1.get_xlim()[0] + 6], ylabel='Depth (mm)')
ax1.text(ax1.get_xlim()[0] + 3.5, 4.4, '5s', ha='center', va='center')
ax1.axes.get_xaxis().set_visible(False)
ax1.invert_yaxis()

plt.tight_layout()
sns.despine(trim=True, offset=2)
plt.savefig(join(fig_path, 'raster_anesthesia_no_light.pdf'))
plt.savefig(join(fig_path, 'raster_anesthesia_no_light.jpg'), dpi=600)



# %%

f, ax1 = plt.subplots(1, 1, figsize=(3.5, 1.75), dpi=dpi)

ax1.imshow(R, aspect='auto', cmap='binary', vmin=0, vmax=np.std(R),
           extent=np.r_[times[[0, -1]], depths[[0, -1]]], origin='upper')
ax1.set(xlim=[1300, 2600], ylim=[0, 4], ylabel='Depth (mm)')
ax1.set(xticks=[ax1.get_xlim()[0] + 10, ax1.get_xlim()[0] + 70])
ax1.text(ax1.get_xlim()[0] + 40, 4.3, '1m', ha='center', va='center')
ax1.axes.get_xaxis().set_visible(False)
ax1.invert_yaxis()

plt.tight_layout()
sns.despine(trim=True, offset=2)
plt.savefig(join(fig_path, 'raster_transition_awake_anesthesia.pdf'))