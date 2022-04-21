#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:17:26 2021
By: Guido Meijer
"""

from os.path import join
from serotonin_functions import paths, figure_style
from sklearn.cluster import KMeans
from scipy.stats import kstest
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
import pandas as pd

# Settings
fig_dir, data_dir = paths(dropbox=True)
fig_dir = join(fig_dir, 'PaperPassive')
CLUSTERING = 'gaussian'  # gaussian or k-means
#FEATURES = ['spike_width', 'rp_slope', 'pt_ratio', 'rc_slope', 'spike_amp']
FEATURES = ['spike_width', 'pt_ratio']
#FEATURES = ['spike_width', 'rp_slope', 'rc_slope', 'pt_ratio']
#FEATURES = ['spike_width', 'rp_slope', 'rc_slope', 'pt_ratio', 'peak_to_trough', 'firing_rate']
#FEATURES = ['spike_width', 'firing_rate']


def fix_hist_step_vertical_line_at_end(ax):
    axpolygons = [poly for poly in ax.get_children() if isinstance(poly, mpl.patches.Polygon)]
    for poly in axpolygons:
        poly.set_xy(poly.get_xy()[:-1])


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

print(f'{(np.sum(waveforms_df["type"] == "FS") / waveforms_df.shape[0]) * 100:.2f}% fast spiking')
print(f'{(np.sum(waveforms_df["type"] == "RS") / waveforms_df.shape[0]) * 100:.2f}% regural spiking')

# Save result
neuron_type = waveforms_df.copy()
excl_df['type'] = 'Und.'
neuron_type = neuron_type.append(excl_df).sort_values('eid')
neuron_type = neuron_type.drop(['waveform', 'spike_width', 'firing_rate', 'rp_slope', 'spike_amp', 'pt_ratio',
                                'rc_slope', 'pt_subtract', 'peak_to_trough', 'n_waveforms'], axis=1)
neuron_type.to_csv(join(data_dir, 'neuron_type.csv'), index=False)

_, p_value = kstest(waveforms_df.loc[waveforms_df['type'] == 'RS', 'firing_rate'],
                    waveforms_df.loc[waveforms_df['type'] == 'FS', 'firing_rate'])
print(f'KS-test p-value: {p_value}')

# %% Plot mean waveforms
colors, dpi = figure_style()
time_ax = np.linspace(0, (waveforms_df.loc[1, 'waveform'].shape[0]/30000)*1000,
                      waveforms_df.loc[1, 'waveform'].shape[0])

f, ax = plt.subplots(1, 1, figsize=(1.5, 1.75), dpi=dpi)
ax.plot(time_ax, waveforms_df.loc[waveforms_df['type'] == 'RS', 'waveform'].to_numpy().mean(),
         color=colors['RS'], label='RS')
ax.plot(time_ax, waveforms_df.loc[waveforms_df['type'] == 'FS', 'waveform'].to_numpy().mean(),
         color=colors['FS'], label='NS')
ax.plot([0.1, 0.1], [-0.18, -0.08], color='k', lw=0.5)
ax.plot([0.1, 1.1], [-0.18, -0.18], color='k', lw=0.5)
ax.text(-0.25, -0.16, '0.1 mV', rotation=90)
ax.text(0.25, -0.21, '1 ms')
ax.set(xlim=[0, 3], ylim=[-0.2, 0.101])
ax.axis('off')

#plt.tight_layout()
#sns.despine(trim=True)
plt.savefig(join(fig_dir, 'figure3_mean_waveforms.pdf'), bbox_inches='tight')

# %% Plot waveform clustering
f, ax = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
ax.scatter(waveforms_df.loc[waveforms_df['type'] == 'RS', 'spike_width'],
            waveforms_df.loc[waveforms_df['type'] == 'RS', 'pt_ratio'], label='Regular spiking',
            color=colors['RS'], s=1)
ax.scatter(waveforms_df.loc[waveforms_df['type'] == 'FS', 'spike_width'],
            waveforms_df.loc[waveforms_df['type'] == 'FS', 'pt_ratio'], label='Narrow spiking',
            color=colors['FS'], s=1)
ax.set(xlabel='Spike width (ms)', ylabel='Peak-to-trough ratio', xlim=[0, 1.5], ylim=[0, 1])
ax.legend(frameon=False, markerscale=2, bbox_to_anchor=(0.2, 0.8), handletextpad=0.1)

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_dir, 'figure3_waveform_clustering.pdf'))

# %% Plot firing rate distribution of clusters

f, ax = plt.subplots(1, 1, figsize=(1.5, 1.75), dpi=dpi)
ax.hist(waveforms_df.loc[waveforms_df['type'] == 'RS', 'firing_rate'], histtype='step',
         color=colors['RS'], density=True, bins=100, cumulative=True)
ax.hist(waveforms_df.loc[waveforms_df['type'] == 'FS', 'firing_rate'], histtype='step',
         color=colors['FS'], density=True, bins=100, cumulative=True)
ax.set(xlabel='Firing rate (spks/s)', ylabel='Density')
fix_hist_step_vertical_line_at_end(ax)

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_dir, 'figure3_firing_rate_dist.pdf'))


