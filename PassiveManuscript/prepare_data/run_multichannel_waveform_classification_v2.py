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
from scipy.stats import linregress
from matplotlib.lines import Line2D
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import pandas as pd
#from one.api import ONE
#one = ONE()


def gaus(x, sigma):
    return np.exp(-x**2 / (2 * sigma**2))


# Settings
REGIONS = ['VISa', 'VISam', 'VISp', 'MOs', 'RSPd']
#REGIONS = ['MOs']
CLUSTERING = 'k-means'
DENOISED = False
SPREAD_CUTOFF = 0.2
SW_CUTOFF = 0.35
#FEATURES = ['spike_width', 'pt_ratio', 'rp_slope', 'rc_slope', 'spread', 'v_above', 'v_below']
#FEATURES = ['spike_width', 'pt_ratio', 'rp_slope', 'rc_slope', 'spread', 'v_above', 'v_below']
FEATURES_1 = ['spike_width', 'pt_ratio']
FEATURES_2 = ['v_below', 'spread']

# Paths
fig_dir, data_dir = paths(dropbox=True)
FIG_PATH = join(fig_dir, 'PaperPassive', 'figure3')

# Load in waveforms
if DENOISED:
    waveforms_df = pd.read_pickle(join(data_dir, 'waveform_metrics_denoised.p'))
else:
    waveforms_df = pd.read_pickle(join(data_dir, 'waveform_metrics.p'))
waveforms_df = waveforms_df.rename(columns={'cluster_id': 'neuron_id'})

# Select neurons from dorsal cortex
waveforms_df = waveforms_df[np.in1d(remap(waveforms_df['regions']), REGIONS)]

"""
# Add insertion angles
for i, pid in enumerate(np.unique(waveforms_df['pid'])):
    traj = one.alyx.rest('trajectories', 'list', provenance='Micro-manipulator', probe_insertion=pid)[0]
    waveforms_df.loc[waveforms_df['pid'] == pid, 'theta'] = traj['theta']
    waveforms_df.loc[waveforms_df['pid'] == pid, 'phi'] = traj['phi']
"""

# Exclude positive spikes
waveforms_df = waveforms_df[waveforms_df['pt_subtract'] < -0.025]
waveforms_df = waveforms_df.reset_index(drop=True)

# Calculate multichannel features
multich_waveforms = []
multich_time_trough = []
multich_dist_soma = []
multich_dist_soma_spread = []
dist_time_df = pd.DataFrame()
for i in waveforms_df.index:

    # Get multichannel waveform for this neuron
    wf_ch_sort = waveforms_df.loc[i, 'waveform_2D']
    dist_soma = waveforms_df.loc[i, 'dist_soma']

    # Exclude waveforms from channels where the amplitude is too low
    spike_amps = np.empty(wf_ch_sort.shape[0])
    for j in range(wf_ch_sort.shape[0]):
        spike_amps[j] = np.abs(np.min(wf_ch_sort[j, :]) - np.max(wf_ch_sort[j, :]))

    # Get normalized amplitude per channel and time of waveform trough
    norm_amp = np.empty(wf_ch_sort.shape[0])
    time_trough = np.empty(wf_ch_sort.shape[0])
    for k in range(wf_ch_sort.shape[0]):
        norm_amp[k] = np.abs(np.min(wf_ch_sort[k, :]) - np.max(wf_ch_sort[k, :]))
        time_trough[k] = (np.argmin(wf_ch_sort[k, :]) / 30000) * 1000  # ms
    time_trough = time_trough - time_trough[dist_soma == 0]
    norm_amp = (norm_amp - np.min(norm_amp)) / (np.max(norm_amp) - np.min(norm_amp))

    # Get spread and velocity
    try:
        
        # Fit Gaussian with two independent arms, one for each side
        x_fit = np.arange(-0.1, 0.11, 0.001)
        param_above, _ = curve_fit(gaus, dist_soma[dist_soma >= 0], norm_amp[dist_soma >= 0], p0=[0.01])
        fit_above = gaus(x_fit[x_fit >= 0], *param_above)
        param_below, _ = curve_fit(gaus, dist_soma[dist_soma <= 0], norm_amp[dist_soma <= 0], p0=[0.01])
        fit_below = gaus(x_fit[x_fit <= 0], *param_below)
        fit = np.empty(x_fit.shape)
        fit[x_fit >= 0] = fit_above
        fit[x_fit <= 0] = fit_below
        
        # Get spread of waveform
        lower_lim = x_fit[fit / np.max(fit) > SPREAD_CUTOFF][0]
        upper_lim = x_fit[fit / np.max(fit) > SPREAD_CUTOFF][-1]
        """
        
        lower_lim = dist_soma[norm_amp > 0.12][0]
        upper_lim = dist_soma[norm_amp > 0.12][-1]
        """
        dist_soma_spread = np.abs(lower_lim) + np.abs(upper_lim)
        
        """
        v_below = (np.max(time_trough[(dist_soma <= 0) & (dist_soma >= lower_lim)])
                   / dist_soma[(dist_soma <= 0) & (dist_soma >= lower_lim)][0])
        if v_below == float('inf'):
            v_below = 0
        v_above = (np.max(time_trough[(dist_soma >= 0) & (dist_soma <= upper_lim)])
                   / dist_soma[(dist_soma >= 0) & (dist_soma <= upper_lim)][-1])
        if v_above == float('inf'):
            v_above = 0
        
        v_below, _ = np.polyfit(time_trough[(dist_soma <= 0) & (dist_soma >= lower_lim)],
                                dist_soma[(dist_soma <= 0) & (dist_soma >= lower_lim)], 1)
        v_above, _ = np.polyfit(time_trough[(dist_soma >= 0) & (dist_soma <= upper_lim)],
                                dist_soma[(dist_soma >= 0) & (dist_soma <= upper_lim)], 1)
        """
        
        if np.sum((dist_soma >= 0) & (dist_soma <= upper_lim)) <= 1:
            v_above = np.nan
        elif np.sum(np.diff(time_trough[(dist_soma >= 0) & (dist_soma <= upper_lim)])) == 0:
            v_above = 0
        else:
            v_above = linregress(time_trough[(dist_soma >= 0) & (dist_soma <= upper_lim)],
                                 dist_soma[(dist_soma >= 0) & (dist_soma <= upper_lim)])[0]
            
        if np.sum((dist_soma <= 0) & (dist_soma >= lower_lim)) <= 1:
            v_below = np.nan
        elif np.sum(np.diff(time_trough[(dist_soma <= 0) & (dist_soma >= lower_lim)])) == 0:
            v_below = 0
        else:
            v_below = linregress(time_trough[(dist_soma <= 0) & (dist_soma >= lower_lim)],
                                 dist_soma[(dist_soma <= 0) & (dist_soma >= lower_lim)])[0]
        
        
        
    except Exception as err:
        print(err)
        waveforms_df.loc[i, 'spread'] = np.nan
        waveforms_df.loc[i, 'v_below'] = np.nan
        waveforms_df.loc[i, 'v_above'] = np.nan
        continue
    
    #if (v_below < 0.61) & (v_below > 0.59) & (dist_soma_spread < 0.05):
    #    alkdj

    # Add new waveforms to list
    multich_waveforms.append(wf_ch_sort)
    multich_time_trough.append(time_trough)
    multich_dist_soma.append(dist_soma)
    multich_dist_soma_spread.append(dist_soma[(dist_soma >= lower_lim) & (dist_soma <= upper_lim)])

    # Add to df
    waveforms_df.loc[i, 'spread'] = dist_soma_spread
    waveforms_df.loc[i, 'v_above'] = v_above
    waveforms_df.loc[i, 'v_below'] = v_below
    waveforms_df.loc[i, 'upper_lim'] = upper_lim
    waveforms_df.loc[i, 'lower_lim'] = lower_lim
    
    # Add distance and time to soma to another dataframe
    dist_time_df = pd.concat((dist_time_df, pd.DataFrame(data={
        'time_soma': time_trough[(dist_soma >= lower_lim) & (dist_soma <= upper_lim)],
        'dist_soma': dist_soma[(dist_soma >= lower_lim) & (dist_soma <= upper_lim)],
        'pid': waveforms_df.loc[i, 'pid'], 'neuron_id': waveforms_df.loc[i, 'neuron_id']})))

# Exclude neurons for which multichannel features could not be calculated
waveforms_df = waveforms_df[~np.isnan(waveforms_df['spread'])]
waveforms_df = waveforms_df.reset_index(drop=True)

"""
# %% Cluster neurons recorded from the left at 15 degrees angle
left_15_df = waveforms_df[(waveforms_df['theta'] == 15) & (waveforms_df['phi'] == 180)]

if CLUSTERING == 'k-means':
    # K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=100).fit(left_15_df[FEATURES].to_numpy())
    left_15_df['group_label'] = kmeans.labels_
elif CLUSTERING == 'gaussian':
    # Mixture of Gaussians clustering
    gauss_mix = GaussianMixture(n_components=3, random_state=42).fit(left_15_df[FEATURES].to_numpy())
    left_15_df['group_label'] = gauss_mix.predict(left_15_df[FEATURES].to_numpy())

# Get the RS and FS labels right
fs_label = left_15_df.groupby('group_label').median(numeric_only=True)['spike_width'].idxmin()
left_15_df.loc[left_15_df['group_label'] == fs_label, 'type'] = 'NS'

rs1_label = left_15_df.groupby('group_label').mean(numeric_only=True)['v_below'].idxmax()
types = np.array([0, 1, 2])
rs2_label = types[~np.isin(types, np.array([fs_label, rs1_label]))][0]
if rs2_label == fs_label:
    rs2_label = types[~np.isin(types, np.array([fs_label, rs1_label]))][0]
left_15_df.loc[left_15_df['group_label'] == rs1_label, 'type'] = 'RS1'
left_15_df.loc[left_15_df['group_label'] == rs2_label, 'type'] = 'RS2'

# Print result
FS_perc = (np.sum(left_15_df['type'] == 'NS') / left_15_df.shape[0]) * 100
RS1_perc = (np.sum(left_15_df['type'] == 'RS1') / left_15_df.shape[0]) * 100
RS2_perc = (np.sum(left_15_df['type'] == 'RS2') / left_15_df.shape[0]) * 100
print(f'\nLeft 15 deg\nFS: {FS_perc:.2f}%\nRS1: {RS1_perc:.2f}%\nRS2: {RS2_perc:.2f}%')

# %% Cluster neurons recorded from the right at 10 degrees angle
right_10_df = waveforms_df[(waveforms_df['theta'] == 10) & (waveforms_df['phi'] == 0)]

if CLUSTERING == 'k-means':
    # K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=100).fit(right_10_df[FEATURES].to_numpy())
    right_10_df['group_label'] = kmeans.labels_
elif CLUSTERING == 'gaussian':
    # Mixture of Gaussians clustering
    gauss_mix = GaussianMixture(n_components=3, random_state=42).fit(right_10_df[FEATURES].to_numpy())
    right_10_df['group_label'] = gauss_mix.predict(right_10_df[FEATURES].to_numpy())

# Get the RS and FS labels right
fs_label = right_10_df.groupby('group_label').median(numeric_only=True)['spike_width'].idxmin()
right_10_df.loc[right_10_df['group_label'] == fs_label, 'type'] = 'NS'

rs1_label = right_10_df.groupby('group_label').mean(numeric_only=True)['v_below'].idxmax()
types = np.array([0, 1, 2])
rs2_label = types[~np.isin(types, np.array([fs_label, rs1_label]))][0]
if rs2_label == fs_label:
    rs2_label = types[~np.isin(types, np.array([fs_label, rs1_label]))][0]
right_10_df.loc[right_10_df['group_label'] == rs1_label, 'type'] = 'RS1'
right_10_df.loc[right_10_df['group_label'] == rs2_label, 'type'] = 'RS2'

# Print result
FS_perc = (np.sum(right_10_df['type'] == 'NS') / right_10_df.shape[0]) * 100
RS1_perc = (np.sum(right_10_df['type'] == 'RS1') / right_10_df.shape[0]) * 100
RS2_perc = (np.sum(right_10_df['type'] == 'RS2') / right_10_df.shape[0]) * 100
print(f'\nRight 10 deg\nFS: {FS_perc:.2f}%\nRS1: {RS1_perc:.2f}%\nRS2: {RS2_perc:.2f}%')


# %% Cluster neurons recorded from the left at 10 degrees angle

left_10_df = waveforms_df[(waveforms_df['theta'] == 10) & (waveforms_df['phi'] == 180)]

if CLUSTERING == 'k-means':
    # K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=100).fit(left_10_df[FEATURES].to_numpy())
    left_10_df['group_label'] = kmeans.labels_
elif CLUSTERING == 'gaussian':
    # Mixture of Gaussians clustering
    gauss_mix = GaussianMixture(n_components=3, random_state=42).fit(left_10_df[FEATURES].to_numpy())
    left_10_df['group_label'] = gauss_mix.predict(left_10_df[FEATURES].to_numpy())

# Get the RS and FS labels right
fs_label = left_10_df.groupby('group_label').median(numeric_only=True)['spike_width'].idxmin()
left_10_df.loc[left_10_df['group_label'] == fs_label, 'type'] = 'NS'

rs1_label = left_10_df.groupby('group_label').mean(numeric_only=True)['v_below'].idxmax()
types = np.array([0, 1, 2])
rs2_label = types[~np.isin(types, np.array([fs_label, rs1_label]))][0]
if rs2_label == fs_label:
    rs2_label = types[~np.isin(types, np.array([fs_label, rs1_label]))][0]
left_10_df.loc[left_10_df['group_label'] == rs1_label, 'type'] = 'RS1'
left_10_df.loc[left_10_df['group_label'] == rs2_label, 'type'] = 'RS2'

# Print result
FS_perc = (np.sum(left_10_df['type'] == 'NS') / left_10_df.shape[0]) * 100
RS1_perc = (np.sum(left_10_df['type'] == 'RS1') / left_10_df.shape[0]) * 100
RS2_perc = (np.sum(left_10_df['type'] == 'RS2') / left_10_df.shape[0]) * 100
print(f'\nLeft 10 deg\nFS: {FS_perc:.2f}%\nRS1: {RS1_perc:.2f}%\nRS2: {RS2_perc:.2f}%')


# Add clustering back to main df
waveforms_df.loc[left_15_df.index, 'type'] = left_15_df['type']
waveforms_df.loc[left_10_df.index, 'type'] = left_10_df['type']
waveforms_df.loc[right_10_df.index, 'type'] = right_10_df['type']
"""


# %% Cluster neurons 

"""
# First cluster narrow spiking from regular spiking
if CLUSTERING == 'k-means':
    # K-means clustering
    kmeans = KMeans(n_clusters=2, random_state=22, n_init=1).fit(waveforms_df[FEATURES_1].to_numpy())
    waveforms_df['group_label'] = kmeans.labels_
elif CLUSTERING == 'gaussian':
    # Mixture of Gaussians clustering
    gauss_mix = GaussianMixture(n_components=2, random_state=42).fit(waveforms_df[FEATURES_1].to_numpy())
    waveforms_df['group_label'] = gauss_mix.predict(waveforms_df[FEATURES_1].to_numpy())
    
# Get the RS and FS labels right
if (waveforms_df.loc[waveforms_df['group_label'] == 0, 'spike_width'].mean()
        < waveforms_df.loc[waveforms_df['group_label'] == 1, 'spike_width'].mean()):
    # type 0 is narrow spiking
    waveforms_df.loc[waveforms_df['group_label'] == 0, 'type'] = 'NS'
    waveforms_df.loc[waveforms_df['group_label'] == 1, 'type'] = 'RS'
else:
    # type 1 is narrow spiking
    waveforms_df.loc[waveforms_df['group_label'] == 0, 'type'] = 'RS'
    waveforms_df.loc[waveforms_df['group_label'] == 1, 'type'] = 'NS'
"""
waveforms_df.loc[waveforms_df['spike_width'] < SW_CUTOFF, 'type'] = 'NS'
waveforms_df.loc[waveforms_df['spike_width'] >= SW_CUTOFF, 'type'] = 'RS'
    

# Then do another clustering on the RS group to split those into RS1 and RS2
waveforms_rs_df = waveforms_df[(waveforms_df['type'] == 'RS') & (waveforms_df['v_below'] != 0)
                               & ~np.isnan(waveforms_df['v_below']) & ~np.isnan(waveforms_df['v_above'])]
if CLUSTERING == 'k-means':
    # K-means clustering
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=100).fit(waveforms_rs_df[FEATURES_2].to_numpy())
    waveforms_rs_df['group_label'] = kmeans.labels_
elif CLUSTERING == 'gaussian':
    # Mixture of Gaussians clustering
    gauss_mix = GaussianMixture(n_components=2, random_state=42).fit(waveforms_rs_df[FEATURES_2].to_numpy())
    waveforms_rs_df['group_label'] = gauss_mix.predict(waveforms_rs_df[FEATURES_2].to_numpy())

# Get the RS1 and RS2 labels right
if (waveforms_rs_df.loc[waveforms_rs_df['group_label'] == 0, 'v_below'].mean()
        > waveforms_rs_df.loc[waveforms_rs_df['group_label'] == 1, 'v_below'].mean()):
    # type 0 is RS1
    waveforms_rs_df.loc[waveforms_rs_df['group_label'] == 0, 'type'] = 'RS1'
    waveforms_rs_df.loc[waveforms_rs_df['group_label'] == 1, 'type'] = 'RS2'
else:
    # type 1 is narrow spiking
    waveforms_rs_df.loc[waveforms_rs_df['group_label'] == 0, 'type'] = 'RS2'
    waveforms_rs_df.loc[waveforms_rs_df['group_label'] == 1, 'type'] = 'RS1'

# Merge into original dataframe
waveforms_df.loc[waveforms_rs_df.index.values, 'type'] = waveforms_rs_df['type']

# Assume neurons with v_below of 0 to be RS1
waveforms_df.loc[(waveforms_df['v_below'] == 0) & (waveforms_df['type'] == 'RS'), 'type'] = 'RS1'

# Set unclustered waveforms as undefined
waveforms_df.loc[waveforms_df['type'] == 'RS', 'type'] = 'Und.'

# %%

# Print result
FS_perc = (np.sum(waveforms_df['type'] == 'NS') / waveforms_df[waveforms_df['type'] != 'Und.'].shape[0]) * 100
RS1_perc = (np.sum(waveforms_df['type'] == 'RS1') / waveforms_df[waveforms_df['type'] != 'Und.'].shape[0]) * 100
RS2_perc = (np.sum(waveforms_df['type'] == 'RS2') / waveforms_df[waveforms_df['type'] != 'Und.'].shape[0]) * 100
print(f'\nOverall result\nFS: {FS_perc:.2f}%\nRS1: {RS1_perc:.2f}%\nRS2: {RS2_perc:.2f}%')

# Save result
neuron_type = waveforms_df.copy()
neuron_type = neuron_type.drop(['waveform', 'spike_width', 'firing_rate', 'rp_slope', 'spike_amp', 'pt_ratio',
                                'rc_slope', 'pt_subtract', 'peak_to_trough', 'n_waveforms',
                                'waveform_2D'], axis=1)
neuron_type.to_csv(join(data_dir, 'neuron_type_multichannel.csv'), index=False)

# Add clustering result to distance time df
dist_time_type_df = pd.merge(dist_time_df, neuron_type[['type', 'pid', 'neuron_id']],
                             on=['pid', 'neuron_id']).sort_values(['pid', 'neuron_id'])

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
            waveforms_df.loc[waveforms_df['type'] == 'RS1', 'pt_ratio'], label='RS1', color=colors['RS'], s=1)
ax2.scatter(waveforms_df.loc[waveforms_df['type'] == 'RS2', 'spike_width'],
            waveforms_df.loc[waveforms_df['type'] == 'RS2', 'pt_ratio'], label='RS2', color=colors['RS'], s=1)
ax2.scatter(waveforms_df.loc[waveforms_df['type'] == 'NS', 'spike_width'],
            waveforms_df.loc[waveforms_df['type'] == 'NS', 'pt_ratio'], label='NS', color=colors['NS'], s=1)
ax2.set(xlabel='Spike width (ms)', ylabel='Peak-to-trough ratio')

ax3.scatter(waveforms_df.loc[waveforms_df['type'] == 'RS1', 'spike_width'],
            waveforms_df.loc[waveforms_df['type'] == 'RS1', 'firing_rate'], label='RS1', color=colors['RS'], s=1)
ax3.scatter(waveforms_df.loc[waveforms_df['type'] == 'RS2', 'spike_width'],
            waveforms_df.loc[waveforms_df['type'] == 'RS2', 'firing_rate'], label='RS2', color=colors['RS'], s=1)
ax3.scatter(waveforms_df.loc[waveforms_df['type'] == 'NS', 'spike_width'],
            waveforms_df.loc[waveforms_df['type'] == 'NS', 'firing_rate'], label='NS', color=colors['NS'], s=1)
ax3.set(xlabel='Spike width (ms)', ylabel='Firing rate (spks/s)')

ax4.scatter(waveforms_df.loc[waveforms_df['type'] == 'RS1', 'spike_width'],
            waveforms_df.loc[waveforms_df['type'] == 'RS1', 'spread'], label='RS1', color=colors['RS1'], s=1)
ax4.scatter(waveforms_df.loc[waveforms_df['type'] == 'RS2', 'spike_width'],
            waveforms_df.loc[waveforms_df['type'] == 'RS2', 'spread'], label='RS2', color=colors['RS2'], s=1)
ax4.scatter(waveforms_df.loc[waveforms_df['type'] == 'NS', 'spike_width'],
            waveforms_df.loc[waveforms_df['type'] == 'NS', 'spread'], label='NS', color=colors['NS'], s=1)
ax4.set(xlabel='Spike width (ms)', ylabel='Waveform spread (mm)')

ax5.scatter(waveforms_df.loc[waveforms_df['type'] == 'RS1', 'v_below'],
            waveforms_df.loc[waveforms_df['type'] == 'RS1', 'spread'], label='RS1', color=colors['RS1'], s=1)
ax5.scatter(waveforms_df.loc[waveforms_df['type'] == 'RS2', 'v_below'],
            waveforms_df.loc[waveforms_df['type'] == 'RS2', 'spread'], label='RS2', color=colors['RS2'], s=1)
#ax5.scatter(waveforms_df.loc[waveforms_df['type'] == 'NS', 'v_below'],
#            waveforms_df.loc[waveforms_df['type'] == 'NS', 'spread'], label='NS', color=colors['NS'], s=1)
ax5.set(xlabel='Velocity below (ms/mm)', ylabel='Waveform spread (mm)')

ax6.scatter(waveforms_df.loc[waveforms_df['type'] == 'RS1', 'v_below'],
            waveforms_df.loc[waveforms_df['type'] == 'RS1', 'upper_lim'], label='RS1', color=colors['RS1'], s=1)
ax6.scatter(waveforms_df.loc[waveforms_df['type'] == 'RS2', 'v_below'],
            waveforms_df.loc[waveforms_df['type'] == 'RS2', 'upper_lim'], label='RS2', color=colors['RS2'], s=1)
#ax6.scatter(waveforms_df.loc[waveforms_df['type'] == 'NS', 'spike_width'],
#            waveforms_df.loc[waveforms_df['type'] == 'NS', 'v_above'], label='NS', color=colors['NS'], s=1)
ax6.set(xlabel='v below', ylabel='upper_lim')

plt.tight_layout()
sns.despine(trim=False)
plt.savefig(join(FIG_PATH, 'multichannel_clustering.jpg'), dpi=600)


# %% Get average waveforms for the three groups
d_len = waveforms_df['dist_soma'].apply(lambda x: len(x)).max()
t_len = waveforms_df['waveform_2D'].apply(lambda x: x.shape[1]).max()
t_x = np.linspace(0, (t_len / 30000) * 1000, t_len)
dist_soma = np.float32(np.round(np.linspace(waveforms_df['dist_soma'].apply(lambda x: np.min(x)).min(),
                                            waveforms_df['dist_soma'].apply(lambda x: np.max(x)).max(),
                                            d_len), 2))

waveforms_1, size_1 = np.zeros((d_len, t_len)), np.zeros((d_len, t_len))
for i in waveforms_df.loc[waveforms_df['type'] == 'NS'].index:
    waveforms_1[np.in1d(dist_soma, waveforms_df.loc[i, 'dist_soma']), :] += waveforms_df.loc[i, 'waveform_2D']
    size_1[np.in1d(dist_soma, waveforms_df.loc[i, 'dist_soma']), :] += 1
waveforms_1 = waveforms_1 / size_1

waveforms_2, size_2 = np.zeros((d_len, t_len)), np.zeros((d_len, t_len))
for i in waveforms_df.loc[waveforms_df['type'] == 'RS1'].index:
    waveforms_2[np.in1d(dist_soma, waveforms_df.loc[i, 'dist_soma']), :] += waveforms_df.loc[i, 'waveform_2D']
    size_2[np.in1d(dist_soma, waveforms_df.loc[i, 'dist_soma']), :] += 1
waveforms_2 = waveforms_2 / size_2

waveforms_3, size_3 = np.zeros((d_len, t_len)), np.zeros((d_len, t_len))
for i in waveforms_df.loc[waveforms_df['type'] == 'NS'].index:
    waveforms_3[np.in1d(dist_soma, waveforms_df.loc[i, 'dist_soma']), :] += waveforms_df.loc[i, 'waveform_2D']
    size_1[np.in1d(dist_soma, waveforms_df.loc[i, 'dist_soma']), :] += 1
waveforms_3 = waveforms_3 / size_3


# %% Plot multichannel waveforms

figure_style()
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(5, 1.75), dpi=dpi)
ax1.imshow(np.flipud(waveforms_1), cmap='coolwarm', aspect='auto',
           vmin=-0.07, vmax=0.07)
ax1.text(50, 10, f'n={np.sum(waveforms_df["type"] == "NS")}', fontsize=7, color='k')
ax1.get_xaxis().set_visible(False)
ax1.set(title='NS', xlim=[np.argmin(np.abs(t_x - 1)), np.argmin(np.abs(t_x - 2))],
        yticks=np.linspace(0, 10, 5), yticklabels=np.round(np.linspace(-.1, .1, 5), 2),
        ylabel='Distance to soma (um)')

ax2.imshow(np.flipud(waveforms_2), cmap='coolwarm', aspect='auto',
           vmin=-0.07, vmax=0.07)
ax2.text(50, 10, f'n={np.sum(waveforms_df["type"] == "RS1")}', fontsize=7, color='k')
ax2.get_xaxis().set_visible(False)
ax2.get_yaxis().set_visible(False)
ax2.set(title='RS1', xlim=[np.argmin(np.abs(t_x - 1)), np.argmin(np.abs(t_x - 2))],
        yticks=np.linspace(0, 10, 5), yticklabels=np.round(np.linspace(-.1, .1, 5), 2))

ax3.imshow(np.flipud(waveforms_3), cmap='coolwarm', aspect='auto',
           vmin=-0.07, vmax=0.07)
ax3.text(50, 10, f'n={np.sum(waveforms_df["type"] == "RS2")}', fontsize=7, color='k')
ax3.get_xaxis().set_visible(False)
ax3.get_yaxis().set_visible(False)
ax3.set(title='RS2', xlim=[np.argmin(np.abs(t_x - 1)), np.argmin(np.abs(t_x - 2))],
        yticks=np.linspace(0, 10, 5), yticklabels=np.round(np.linspace(-.1, .1, 5), 2))

#%% Plot time and distance to soma

f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
sns.lineplot(data=dist_time_type_df[dist_time_type_df['type'] != 'Und.'], x='dist_soma', y='time_soma',
             ax=ax1, errorbar='se', hue='type', hue_order=['NS', 'RS1', 'RS2'], 
             palette=[colors['NS'], colors['RS1'], colors['RS2']])
ax1.legend(title='', frameon=False, prop={'size': 5.5}, bbox_to_anchor=(0.9, 0.35))
ax1.set(xlabel='Distance to soma (mm)', ylabel='Time rel. to soma (ms)')

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(FIG_PATH, 'multichannel_waveform_groups.pdf'))


"""
#%% Plot time and distance to soma
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(4, 1.75), dpi=dpi, sharey=True)
for i in waveforms_df.loc[waveforms_df['type'] == 'NS'].index:
    ax1.plot(t_x[multich_waveforms[i].argmin(axis=1)][np.in1d(multich_dist_soma[i], multich_dist_soma_spread[i])],
             multich_dist_soma_spread[i], color=[.7, .7, .7], alpha=0.2)
ax1.errorbar(np.nanmedian(prop_1, axis=0), dist_soma,
             xerr=np.nanstd(prop_1, axis=0)/np.sqrt(np.sum(~np.isnan(prop_1), axis=0)), lw=2)
ax1.set(xlim=[1.2, 1.7], xlabel='Time (ms)', ylabel='Distance to soma (um)',
        yticks=np.round(np.linspace(-.1, .1, 5), 2))

if waveforms_df.loc[waveforms_df['type'] == 'RS1'].shape[0] > 0:
    for i in waveforms_df.loc[waveforms_df['type'] == 'RS1'].index:
        ax2.plot(t_x[multich_waveforms[i].argmin(axis=1)][np.in1d(multich_dist_soma[i], multich_dist_soma_spread[i])],
                 multich_dist_soma_spread[i], color=[.7, .7, .7], alpha=0.2)
    ax2.errorbar(np.nanmedian(prop_2, axis=0), dist_soma,
                 xerr=np.nanstd(prop_2, axis=0)/np.sqrt(np.sum(~np.isnan(prop_2), axis=0)), lw=2)
    ax2.set(xlim=[1, 2], xlabel='Time (ms)', yticks=np.round(np.linspace(-.1, .1, 5), 2))

for i in waveforms_df.loc[waveforms_df['type'] == 'RS2'].index:
    ax3.plot(t_x[multich_waveforms[i].argmin(axis=1)][np.in1d(multich_dist_soma[i], multich_dist_soma_spread[i])],
             multich_dist_soma_spread[i], color=[.7, .7, .7], alpha=0.2)
ax3.errorbar(np.nanmedian(prop_3, axis=0), dist_soma,
             xerr=np.nanstd(prop_3, axis=0)/np.sqrt(np.sum(~np.isnan(prop_3), axis=0)), lw=2)
ax3.set(xlim=[1, 2], xlabel='Time (ms)', yticks=np.round(np.linspace(-.1, .1, 5), 2))


plt.savefig(join(FIG_PATH, 'multichannel_waveform_groups.jpg'), dpi=600)
"""