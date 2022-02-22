#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:17:26 2021
By: Guido Meijer
"""

from os.path import join
from serotonin_functions import paths, figure_style
from sklearn.cluster import KMeans
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
import pandas as pd

# Settings
fig_dir, data_dir = paths()
CLUSTERING = 'gaussian'  # gaussian or k-means
FEATURES = ['spike_width', 'rp_slope', 'pt_ratio', 'rc_slope', 'spike_amp']
#FEATURES = ['spike_width', 'spike_amp']
#FEATURES = ['spike_width', 'rp_slope', 'rc_slope', 'pt_ratio']
#FEATURES = ['spike_width', 'rp_slope', 'rc_slope', 'pt_ratio', 'peak_to_trough', 'firing_rate']
#FEATURES = ['spike_width', 'firing_rate']

# Load in waveforms
waveforms_df = pd.read_pickle(join(data_dir, 'waveform_metrics.p'))

# Exclude positive spikes
excl_df = waveforms_df[waveforms_df['pt_subtract'] > -0.05]
waveforms_df = waveforms_df[waveforms_df['pt_subtract'] <= -0.05]
waveforms_df = waveforms_df.reset_index(drop=True)

if CLUSTERING == 'k-means':
    # K-means clustering
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=100).fit(waveforms_df[FEATURES].to_numpy())
    waveforms_df['type'] = kmeans.labels_
elif CLUSTERING == 'gaussian':
    # Mixture of Gaussians clustering
    gauss_mix = GaussianMixture(n_components=2).fit(waveforms_df[FEATURES].to_numpy())
    waveforms_df['type'] = gauss_mix.predict(waveforms_df[FEATURES].to_numpy())

# Get the RS and FS labels right
if (waveforms_df.loc[waveforms_df['type'] == 0, 'firing_rate'].mean()
        > waveforms_df.loc[waveforms_df['type'] == 1, 'firing_rate'].mean()):
    # type 0 is fast spiking
    waveforms_df.loc[waveforms_df['type'] == 0, 'type'] = 'FS'
    waveforms_df.loc[waveforms_df['type'] == 1, 'type'] = 'RS'
else:
    # type 1 is fast spiking
    waveforms_df.loc[waveforms_df['type'] == 0, 'type'] = 'RS'
    waveforms_df.loc[waveforms_df['type'] == 1, 'type'] = 'FS'


# Save result
neuron_type = waveforms_df.copy()
excl_df['type'] = 'Und.'
neuron_type = neuron_type.append(excl_df).sort_values('eid')
neuron_type = neuron_type.drop(['waveform', 'spike_width', 'firing_rate', 'rp_slope', 'spike_amp', 'pt_ratio',
                                'rc_slope', 'pt_subtract', 'peak_to_trough', 'n_waveforms'], axis=1)
neuron_type.to_csv(join(data_dir, 'neuron_type.csv'), index=False)

# %% Plot
colors, dpi = figure_style()
time_ax = np.linspace(0, (waveforms_df.loc[1, 'waveform'].shape[0]/30000)*1000,
                      waveforms_df.loc[1, 'waveform'].shape[0])

f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(6, 3.5), dpi=dpi)
ax1.plot(time_ax, waveforms_df.loc[waveforms_df['type'] == 'RS', 'waveform'].to_numpy().mean(),
         color=colors['RS'], label='RS')
ax1.plot(time_ax, waveforms_df.loc[waveforms_df['type'] == 'FS', 'waveform'].to_numpy().mean(),
         color=colors['FS'], label='FS')
ax1.legend(frameon=False)
ax1.set(ylabel='mV', xlabel='Time (ms)', xlim=[0, 3])

ax2.scatter(waveforms_df.loc[waveforms_df['type'] == 'RS', 'spike_width'],
            waveforms_df.loc[waveforms_df['type'] == 'RS', 'rp_slope'], label='RS', color=colors['RS'], s=1)
ax2.scatter(waveforms_df.loc[waveforms_df['type'] == 'FS', 'spike_width'],
            waveforms_df.loc[waveforms_df['type'] == 'FS', 'rp_slope'], label='FS', color=colors['FS'], s=1)
ax2.set(xlabel='Spike width (ms)', ylabel='Repolarization slope', xlim=[0, 1.5])

ax3.scatter(waveforms_df.loc[waveforms_df['type'] == 'RS', 'spike_width'],
            waveforms_df.loc[waveforms_df['type'] == 'RS', 'firing_rate'], label='RS', color=colors['RS'], s=1)
ax3.scatter(waveforms_df.loc[waveforms_df['type'] == 'FS', 'spike_width'],
            waveforms_df.loc[waveforms_df['type'] == 'FS', 'firing_rate'], label='FS', color=colors['FS'], s=1)
ax3.set(xlabel='Spike width (ms)', ylabel='Firing rate (spks/s)', xlim=[0, 1.5], ylim=[0, 100])

ax4.scatter(waveforms_df.loc[waveforms_df['type'] == 'RS', 'spike_width'],
            waveforms_df.loc[waveforms_df['type'] == 'RS', 'pt_ratio'], label='RS', color=colors['RS'], s=1)
ax4.scatter(waveforms_df.loc[waveforms_df['type'] == 'FS', 'spike_width'],
            waveforms_df.loc[waveforms_df['type'] == 'FS', 'pt_ratio'], label='FS', color=colors['FS'], s=1)
ax4.set(xlabel='Spike width (ms)', ylabel='Peak-to-trough ratio', xlim=[0, 1.5], ylim=[0, 1])

ax5.scatter(waveforms_df.loc[waveforms_df['type'] == 'RS', 'spike_width'],
            waveforms_df.loc[waveforms_df['type'] == 'RS', 'rc_slope'], label='RS', color=colors['RS'], s=1)
ax5.scatter(waveforms_df.loc[waveforms_df['type'] == 'FS', 'spike_width'],
            waveforms_df.loc[waveforms_df['type'] == 'FS', 'rc_slope'], label='FS', color=colors['FS'], s=1)
ax5.set(xlabel='Spike width (ms)', ylabel='Recovery slope', xlim=[0, 1.5])

ax6.scatter(waveforms_df.loc[waveforms_df['type'] == 'RS', 'spike_width'],
            waveforms_df.loc[waveforms_df['type'] == 'RS', 'spike_amp'], label='RS', color=colors['RS'], s=1)
ax6.scatter(waveforms_df.loc[waveforms_df['type'] == 'FS', 'spike_width'],
            waveforms_df.loc[waveforms_df['type'] == 'FS', 'spike_amp'], label='FS', color=colors['FS'], s=1)
ax6.set(xlabel='Spike width (ms)', ylabel='Spike amplitude (uV)', xlim=[0, 1.5])

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_dir, 'Ephys', 'NeuronType', 'neuron_type_classification'), dpi=300)
plt.savefig(join(fig_dir, 'Ephys', 'NeuronType', 'neuron_type_classification.pdf'))

# %%
"""
f, axs = plt.subplots(int(np.floor(np.sqrt(len(waveforms_df['eid'].unique())))),
                      int(np.ceil(np.sqrt(len(waveforms_df['eid'].unique())))), figsize=(8, 4), dpi=dpi)
axs = axs.flatten()
for i, eid in enumerate(waveforms_df['eid'].unique()):
    these_waveforms = waveforms_df[waveforms_df['eid'] == eid]
    for k in these_waveforms.index.values:
        this_waveform = waveforms_df.loc[k, 'waveform']
        this_waveform = -(this_waveform / np.min(this_waveform))
        axs[i].plot(time_ax, this_waveform, color=colors[these_waveforms.loc[k, 'type']], lw=0.2)
        axs[i].axis('off')
plt.savefig(join(fig_dir, 'Ephys', 'NeuronType', 'neuron_type_classification_all_waveforms'), dpi=300)

f, ax1 = plt.subplots(1, 1, figsize=(3, 3), dpi=dpi)
random_wfs = np.random.choice(waveforms_df[waveforms_df['type'] == 'RS'].index.values, 100)
plot_wfs = np.concatenate([waveforms_df[waveforms_df['type'] == 'FS'].index.values, random_wfs])
for k in plot_wfs:
    this_waveform = waveforms_df.loc[k, 'waveform']
    this_waveform = -(this_waveform / np.min(this_waveform))
    ax1.plot(time_ax, this_waveform, color=colors[waveforms_df.loc[k, 'type']], lw=0.2)
    ax1.axis('off')
plt.savefig(join(fig_dir, 'Ephys', 'NeuronType', 'neuron_type_classification_example_waveforms'), dpi=300)
plt.savefig(join(fig_dir, 'Ephys', 'NeuronType', 'neuron_type_classification_example_waveforms.pdf'))
"""
