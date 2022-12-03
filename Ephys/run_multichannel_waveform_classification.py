#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:17:26 2021
By: Guido Meijer
"""

from os.path import join
from serotonin_functions import paths, figure_style, remap
from scipy.stats import kstest
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
from matplotlib.lines import Line2D
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import pandas as pd
from one.api import ONE
one = ONE()


def gaus(x, a, x0, sigma):
    return a * np.exp(-(x-x0)**2 / (2 * sigma**2))


# Settings
REGIONS = ['VISa', 'VISam', 'VISp', 'MOs']
#REGIONS = ['MOs']
CLUSTERING = 'gaussian'
MIN_SPIKE_AMP = 0
PHI = 180
THETA = 15
FEATURES = ['spike_width', 'pt_ratio', 'spread', 'v_above', 'v_below']

# Paths
fig_dir, data_dir = paths()
FIG_PATH = join(fig_dir, 'Ephys', 'NeuronType')

# Load in waveforms
waveforms_df = pd.read_pickle(join(data_dir, 'waveform_metrics.p'))
waveforms_df = waveforms_df.rename(columns={'cluster_id': 'neuron_id'})

# Select neurons from dorsal cortex
waveforms_df = waveforms_df[np.in1d(remap(waveforms_df['regions']), REGIONS)]

# Add insertion angles
for i, pid in enumerate(np.unique(waveforms_df['pid'])):
    traj = one.alyx.rest('trajectories', 'list', provenance='Micro-manipulator', probe_insertion=pid)[0]
    waveforms_df.loc[waveforms_df['pid'] == pid, 'theta'] = traj['theta']
    waveforms_df.loc[waveforms_df['pid'] == pid, 'phi'] = traj['phi']

# Select only insertions with the same angle and side
waveforms_df = waveforms_df[(waveforms_df['theta'] == THETA) & (waveforms_df['phi'] == PHI)]

# Exclude positive spikes
excl_df = waveforms_df[waveforms_df['pt_subtract'] > -0.05]
waveforms_df = waveforms_df[waveforms_df['pt_subtract'] <= -0.05]
waveforms_df = waveforms_df.reset_index(drop=True)

# Calculate multichannel features
multich_waveforms = []
multich_dist_soma = []
for i in waveforms_df.index:

    # Get multichannel waveform for this neuron
    wf_ch_sort = waveforms_df.loc[i, 'waveform_2D']
    dist_soma = waveforms_df.loc[i, 'dist_soma']

    # Exclude waveforms from channels where the amplitude is too low
    spike_amps = np.empty(wf_ch_sort.shape[0])
    for j in range(wf_ch_sort.shape[0]):
        spike_amps[j] = np.abs(np.min(wf_ch_sort[j, :]) - np.max(wf_ch_sort[j, :]))
    wf_ch_sort = wf_ch_sort[spike_amps > MIN_SPIKE_AMP, :]
    dist_soma = dist_soma[spike_amps > MIN_SPIKE_AMP]

    # Get normalized amplitude per channel and time of waveform trough
    norm_amp = np.empty(wf_ch_sort.shape[0])
    time_trough = np.empty(wf_ch_sort.shape[0])
    for k in range(wf_ch_sort.shape[0]):
        norm_amp[k] = np.abs(np.min(wf_ch_sort[k, :]) - np.max(wf_ch_sort[k, :]))
        time_trough[k] = (np.argmin(wf_ch_sort[k, :]) / 30000) * 1000  # ms
    norm_amp = (norm_amp - np.min(norm_amp)) / (np.max(norm_amp) - np.min(norm_amp))

    # Get spread and velocity
    try:
        popt, pcov = curve_fit(gaus, dist_soma, norm_amp, p0=[1, 0, 0.1])
        fit = gaus(dist_soma, *popt)
        spread = (np.sum(fit / np.max(fit) > 0.12) * 20) / 1000
        v_below, _ = np.polyfit(time_trough[dist_soma <= 0], dist_soma[dist_soma <= 0], 1)
        v_above, _ = np.polyfit(time_trough[dist_soma >= 0], dist_soma[dist_soma >= 0], 1)
    except:
        waveforms_df.loc[i, 'spread'] = np.nan
        waveforms_df.loc[i, 'v_below'] = np.nan
        waveforms_df.loc[i, 'v_above'] = np.nan
        continue

    # Add new waveforms to list
    multich_waveforms.append(wf_ch_sort)
    multich_dist_soma.append(dist_soma)

    # Add to df
    waveforms_df.loc[i, 'spread'] = spread
    waveforms_df.loc[i, 'v_above'] = v_above
    waveforms_df.loc[i, 'v_below'] = v_below

# Exclude neurons for which multichannel features could not be calculated
waveforms_df = waveforms_df[~np.isnan(waveforms_df['spread'])]
waveforms_df = waveforms_df.reset_index(drop=True)

if CLUSTERING == 'k-means':
    # K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=100).fit(waveforms_df[FEATURES].to_numpy())
    waveforms_df['group_label'] = kmeans.labels_
elif CLUSTERING == 'gaussian':
    # Mixture of Gaussians clustering
    gauss_mix = GaussianMixture(n_components=3, random_state=42).fit(waveforms_df[FEATURES].to_numpy())
    waveforms_df['group_label'] = gauss_mix.predict(waveforms_df[FEATURES].to_numpy())

# Get the RS and FS labels right
fs_label = waveforms_df.groupby('group_label').median(numeric_only=True)['spike_width'].idxmin()
waveforms_df.loc[waveforms_df['group_label'] == fs_label, 'type'] = 'NS'

rs1_label = waveforms_df.groupby('group_label').mean(numeric_only=True)['v_below'].idxmax()
types = np.array([0, 1, 2])
rs2_label = types[~np.isin(types, np.array([fs_label, rs1_label]))][0]
if rs2_label == fs_label:
    rs2_label = types[~np.isin(types, np.array([fs_label, rs1_label]))][0]
waveforms_df.loc[waveforms_df['group_label'] == rs1_label, 'type'] = 'RS1'
waveforms_df.loc[waveforms_df['group_label'] == rs2_label, 'type'] = 'RS2'

# Print result
FS_perc = (np.sum(waveforms_df['type'] == 'NS') / waveforms_df.shape[0]) * 100
RS1_perc = (np.sum(waveforms_df['type'] == 'RS1') / waveforms_df.shape[0]) * 100
RS2_perc = (np.sum(waveforms_df['type'] == 'RS2') / waveforms_df.shape[0]) * 100
print(f'FS: {FS_perc:.2f}%\nRS1: {RS1_perc:.2f}%\nRS2: {RS2_perc:.2f}%')

# Save result
neuron_type = waveforms_df.copy()
excl_df['type'] = 'Und.'
neuron_type = neuron_type.append(excl_df).sort_values('eid')
neuron_type = neuron_type.drop(['waveform', 'spike_width', 'firing_rate', 'rp_slope', 'spike_amp', 'pt_ratio',
                                'rc_slope', 'pt_subtract', 'peak_to_trough', 'n_waveforms',
                                'waveform_2D'], axis=1)
neuron_type.to_csv(join(data_dir, 'neuron_type_multichannel.csv'), index=False)

# %%
colors, dpi = figure_style()
time_ax = np.linspace(0, (waveforms_df['waveform_2D'][1].shape[1]/30000)*1000,
                          waveforms_df['waveform_2D'][1].shape[1])

# Plot clustering
f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(6, 3.5), dpi=dpi)
ax1.plot(time_ax, waveforms_df.loc[waveforms_df['type'] == 'RS1', 'waveform'].to_numpy().mean(),
         color=colors['RS1'], label='RS1')
ax1.plot(time_ax, waveforms_df.loc[waveforms_df['type'] == 'RS2', 'waveform'].to_numpy().mean(),
         color=colors['RS2'], label='RS2')
ax1.plot(time_ax, waveforms_df.loc[waveforms_df['type'] == 'NS', 'waveform'].to_numpy().mean(),
         color=colors['NS'], label='NS')
ax1.legend(frameon=False)
ax1.set(ylabel='mV', xlabel='Time (ms)')

ax2.scatter(waveforms_df.loc[waveforms_df['type'] == 'RS1', 'spike_width'],
            waveforms_df.loc[waveforms_df['type'] == 'RS1', 'rp_slope'], label='RS1', color=colors['RS1'], s=1)
ax2.scatter(waveforms_df.loc[waveforms_df['type'] == 'RS2', 'spike_width'],
            waveforms_df.loc[waveforms_df['type'] == 'RS2', 'rp_slope'], label='RS2', color=colors['RS2'], s=1)
ax2.scatter(waveforms_df.loc[waveforms_df['type'] == 'NS', 'spike_width'],
            waveforms_df.loc[waveforms_df['type'] == 'NS', 'rp_slope'], label='NS', color=colors['NS'], s=1)
ax2.set(xlabel='Spike width (ms)', ylabel='Repolarization slope')

ax3.scatter(waveforms_df.loc[waveforms_df['type'] == 'RS1', 'spike_width'],
            waveforms_df.loc[waveforms_df['type'] == 'RS1', 'firing_rate'], label='RS1', color=colors['RS1'], s=1)
ax3.scatter(waveforms_df.loc[waveforms_df['type'] == 'RS2', 'spike_width'],
            waveforms_df.loc[waveforms_df['type'] == 'RS2', 'firing_rate'], label='RS2', color=colors['RS2'], s=1)
ax3.scatter(waveforms_df.loc[waveforms_df['type'] == 'NS', 'spike_width'],
            waveforms_df.loc[waveforms_df['type'] == 'NS', 'firing_rate'], label='NS', color=colors['NS'], s=1)
ax3.set(xlabel='Spike width (ms)', ylabel='Firing rate (spks/s)')

ax4.scatter(waveforms_df.loc[waveforms_df['type'] == 'RS1', 'spike_width'],
            waveforms_df.loc[waveforms_df['type'] == 'RS1', 'pt_ratio'], label='RS1', color=colors['RS1'], s=1)
ax4.scatter(waveforms_df.loc[waveforms_df['type'] == 'RS2', 'spike_width'],
            waveforms_df.loc[waveforms_df['type'] == 'RS2', 'pt_ratio'], label='RS2', color=colors['RS2'], s=1)
ax4.scatter(waveforms_df.loc[waveforms_df['type'] == 'NS', 'spike_width'],
            waveforms_df.loc[waveforms_df['type'] == 'NS', 'pt_ratio'], label='NS', color=colors['NS'], s=1)
ax4.set(xlabel='Spike width (ms)', ylabel='Peak-to-trough ratio')

ax5.scatter(waveforms_df.loc[waveforms_df['type'] == 'RS1', 'spike_width'],
            waveforms_df.loc[waveforms_df['type'] == 'RS1', 'rc_slope'], label='RS1', color=colors['RS1'], s=1)
ax5.scatter(waveforms_df.loc[waveforms_df['type'] == 'RS2', 'spike_width'],
            waveforms_df.loc[waveforms_df['type'] == 'RS2', 'rc_slope'], label='RS2', color=colors['RS2'], s=1)
ax5.scatter(waveforms_df.loc[waveforms_df['type'] == 'NS', 'spike_width'],
            waveforms_df.loc[waveforms_df['type'] == 'NS', 'rc_slope'], label='NS', color=colors['NS'], s=1)
ax5.set(xlabel='Spike width (ms)', ylabel='Recovery slope')

ax6.scatter(waveforms_df.loc[waveforms_df['type'] == 'RS1', 'spike_width'],
            waveforms_df.loc[waveforms_df['type'] == 'RS1', 'spike_amp'], label='RS1', color=colors['RS1'], s=1)
ax6.scatter(waveforms_df.loc[waveforms_df['type'] == 'RS2', 'spike_width'],
            waveforms_df.loc[waveforms_df['type'] == 'RS2', 'spike_amp'], label='RS2', color=colors['RS2'], s=1)
ax6.scatter(waveforms_df.loc[waveforms_df['type'] == 'NS', 'spike_width'],
            waveforms_df.loc[waveforms_df['type'] == 'NS', 'spike_amp'], label='NS', color=colors['NS'], s=1)
ax6.set(xlabel='Spike width (ms)', ylabel='Spike amplitude (uV)')

plt.tight_layout()
sns.despine(trim=False)
plt.savefig(join(FIG_PATH, 'multichannel_clustering.jpg'), dpi=600)

# %%
d_len = waveforms_df['dist_soma'].apply(lambda x: len(x)).max()
t_len = waveforms_df['waveform_2D'].apply(lambda x: x.shape[1]).max()
t_x = np.linspace(0, (t_len / 30000) * 1000, t_len)
dist_soma = np.float32(np.round(np.linspace(waveforms_df['dist_soma'].apply(lambda x: np.min(x)).min(),
                                            waveforms_df['dist_soma'].apply(lambda x: np.max(x)).max(),
                                            d_len), 2))

waveforms_1, size_1 = np.zeros((d_len, t_len)), np.zeros((d_len, t_len))
prop_1 = np.full(d_len, np.nan)
for i in waveforms_df.loc[waveforms_df['type'] == 'NS'].index:
    waveforms_1[np.in1d(dist_soma, multich_dist_soma[i])] = (
        waveforms_1[np.in1d(dist_soma, multich_dist_soma[i])]
        + multich_waveforms[i])
    size_1[np.in1d(dist_soma, multich_dist_soma[i])] = size_1[
        np.in1d(dist_soma, multich_dist_soma[i])] + 1
    this_prop = np.full(d_len, np.nan)
    this_prop[np.in1d(dist_soma, multich_dist_soma[i])] = t_x[
        multich_waveforms[i].argmin(axis=1)]
    prop_1 = np.vstack((prop_1, this_prop))
waveforms_1 = waveforms_1 / size_1

waveforms_2, size_2 = np.zeros((d_len, t_len)), np.zeros((d_len, t_len))
prop_2 = np.full(d_len, np.nan)
for i in waveforms_df.loc[waveforms_df['type'] == 'RS1'].index:
    waveforms_2[np.in1d(dist_soma, multich_dist_soma[i])] = (
        waveforms_2[np.in1d(dist_soma, multich_dist_soma[i])]
        + multich_waveforms[i])
    size_2[np.in1d(dist_soma, multich_dist_soma[i])] = size_2[
        np.in1d(dist_soma, multich_dist_soma[i])] + 1
    this_prop = np.full(d_len, np.nan)
    this_prop[np.in1d(dist_soma, multich_dist_soma[i])] = t_x[
        multich_waveforms[i].argmin(axis=1)]
    prop_2 = np.vstack((prop_2, this_prop))
waveforms_2 = waveforms_2 / size_2

waveforms_3, size_3 = np.zeros((d_len, t_len)), np.zeros((d_len, t_len))
prop_3 = np.full(d_len, np.nan)
for i in waveforms_df.loc[waveforms_df['type'] == 'RS2'].index:
    waveforms_3[np.in1d(dist_soma, multich_dist_soma[i])] = (
        waveforms_3[np.in1d(dist_soma, multich_dist_soma[i])]
        + multich_waveforms[i])
    size_3[np.in1d(dist_soma, multich_dist_soma[i])] = size_3[
        np.in1d(dist_soma, multich_dist_soma[i])] + 1
    this_prop = np.full(d_len, np.nan)
    this_prop[np.in1d(dist_soma, multich_dist_soma[i])] = t_x[
        multich_waveforms[i].argmin(axis=1)]
    prop_3 = np.vstack((prop_3, this_prop))
waveforms_3 = waveforms_3 / size_3

figure_style()
f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(10, 5), dpi=dpi)
ax3.imshow(np.flipud(waveforms_3), cmap='Greys_r', aspect='auto',
           vmin=-np.max(waveforms_1), vmax=np.max(waveforms_1))
ax3.get_xaxis().set_visible(False)
ax3.get_yaxis().set_visible(False)
ax3.set(title='RS2', xlim=[np.argmin(np.abs(t_x - 1)), np.argmin(np.abs(t_x - 2))])

ax2.imshow(np.flipud(waveforms_2), cmap='Greys_r', aspect='auto',
           vmin=-np.max(waveforms_2), vmax=np.max(waveforms_2))
ax2.get_xaxis().set_visible(False)
ax2.get_yaxis().set_visible(False)
ax2.set(title='RS1', xlim=[np.argmin(np.abs(t_x - 1)), np.argmin(np.abs(t_x - 2))])

ax1.imshow(np.flipud(waveforms_1), cmap='Greys_r', aspect='auto',
           vmin=-np.max(waveforms_3), vmax=np.max(waveforms_3))
ax1.get_xaxis().set_visible(False)
ax1.set(title='FS', xlim=[np.argmin(np.abs(t_x - 1)), np.argmin(np.abs(t_x - 2))],
        yticks=np.linspace(0, 10, 5), yticklabels=np.round(np.linspace(-.1, .1, 5), 2),
        ylabel='Distance to soma (um)')

for i in waveforms_df.loc[waveforms_df['type'] == 'RS2'].index:
    ax6.plot(t_x[multich_waveforms[i].argmin(axis=1)],
             multich_dist_soma[i], color=[.7, .7, .7], alpha=0.2)
ax6.errorbar(np.nanmedian(prop_3, axis=0), dist_soma,
             xerr=np.nanstd(prop_3, axis=0)/np.sqrt(np.sum(~np.isnan(prop_3), axis=0)), lw=3)
ax6.set(xlim=[1, 2], xlabel='Time (ms)', yticks=np.round(np.linspace(-.1, .1, 5), 2))

if waveforms_df.loc[waveforms_df['type'] == 'RS1'].shape[0] > 0:
    for i in waveforms_df.loc[waveforms_df['type'] == 'RS1'].index:
        ax5.plot(t_x[multich_waveforms[i].argmin(axis=1)],
                 multich_dist_soma[i], color=[.7, .7, .7], alpha=0.2)
    ax5.errorbar(np.nanmedian(prop_2, axis=0), dist_soma,
                 xerr=np.nanstd(prop_2, axis=0)/np.sqrt(np.sum(~np.isnan(prop_2), axis=0)), lw=3)
    ax5.set(xlim=[1, 2], xlabel='Time (ms)', yticks=np.round(np.linspace(-.1, .1, 5), 2))

for i in waveforms_df.loc[waveforms_df['type'] == 'NS'].index:
    ax4.plot(t_x[multich_waveforms[i].argmin(axis=1)],
             multich_dist_soma[i], color=[.7, .7, .7], alpha=0.2)
ax4.errorbar(np.nanmedian(prop_1, axis=0), dist_soma,
             xerr=np.nanstd(prop_1, axis=0)/np.sqrt(np.sum(~np.isnan(prop_1), axis=0)), lw=3)
ax4.set(xlim=[1, 2], xlabel='Time (ms)', ylabel='Distance to soma (um)',
        yticks=np.round(np.linspace(-.1, .1, 5), 2))

plt.savefig(join(FIG_PATH, 'multichannel_waveform_groups.jpg'), dpi=600)