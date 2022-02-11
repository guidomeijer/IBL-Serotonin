#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:17:26 2021
By: Guido Meijer
"""

import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from brainbox.singlecell import calculate_peths
from brainbox.metrics.single_units import spike_sorting_metrics
from serotonin_functions import figure_style
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import brainbox.io.one as bbone
from sklearn.preprocessing import StandardScaler
from brainbox.population.decode import get_spike_counts_in_bins
from sklearn.decomposition import PCA
from serotonin_functions import (paths, remap, query_ephys_sessions, load_passive_opto_times,
                                 get_artifact_neurons)
from one.api import ONE
from ibllib.atlas import AllenAtlas
one = ONE()
ba = AllenAtlas()
pca = PCA(n_components=3)

# Settings
MIN_NEURONS = 10  # per region
T_BEFORE = 0.5
T_AFTER = 1.5
BASELINE = [-0.5, 0]
BIN_SIZE = 0.01
SMOOTHING = 0.02
MIN_FR = 0.1
PLOT = True
_, fig_path, save_path = paths()
fig_path = join(fig_path, 'Ephys', 'PCA')

# Get binning time vectors
BIN_CENTERS = np.arange(-T_BEFORE, T_AFTER, BIN_SIZE) + (BIN_SIZE / 2)

# Query sessions
eids, _ = query_ephys_sessions(one=one)

pca_df = pd.DataFrame()
artifact_neurons = get_artifact_neurons()
for i, eid in enumerate(eids):

    # Get session details
    ses_details = one.get_details(eid)
    subject = ses_details['subject']
    date = ses_details['start_time'][:10]
    print(f'Starting {subject}, {date}')

    # Load in laser pulse times
    try:
        opto_train_times, _ = load_passive_opto_times(eid, one=one)
    except:
        print('Session does not have passive laser pulses')
        continue
    if len(opto_train_times) == 0:
        print('Did not find ANY laser pulses!')
        continue
    else:
        print(f'Found {len(opto_train_times)} passive laser pulses')

    # Load in spikes
    spikes, clusters, channels = bbone.load_spike_sorting_with_channel(
        eid, aligned=True, dataset_types=['spikes.amps', 'spikes.depths'], one=one, brain_atlas=ba)

    for p, probe in enumerate(spikes.keys()):

        # Filter neurons that pass QC
        if 'metrics' in clusters[probe].keys():
            clusters_pass = np.where(clusters[probe]['metrics']['label'] == 1)[0]
        else:
            print('Calculating neuron QC metrics..')
            qc_metrics, _ = spike_sorting_metrics(spikes[probe].times, spikes[probe].clusters,
                                                  spikes[probe].amps, spikes[probe].depths,
                                                  cluster_ids=np.arange(clusters[probe].channels.size))
            clusters_pass = np.where(qc_metrics['label'] == 1)[0]

        # Select spikes of passive period
        start_passive = opto_train_times[0] - 360
        spikes[probe].clusters = spikes[probe].clusters[spikes[probe].times > start_passive]
        spikes[probe].times = spikes[probe].times[spikes[probe].times > start_passive]

        # Exclude artifact neurons
        clusters_pass = np.array([i for i in clusters_pass if i not in artifact_neurons.loc[
            (artifact_neurons['eid'] == eid) & (artifact_neurons['probe'] == probe), 'neuron_id'].values])
        if clusters_pass.shape[0] == 0:
                continue

        # Select QC pass neurons
        spikes[probe].times = spikes[probe].times[np.isin(spikes[probe].clusters, clusters_pass)]
        spikes[probe].clusters = spikes[probe].clusters[np.isin(spikes[probe].clusters, clusters_pass)]
        clusters_pass = clusters_pass[np.isin(clusters_pass, np.unique(spikes[probe].clusters))]

        # Get regions from Beryl atlas
        clusters[probe]['region'] = remap(clusters[probe]['atlas_id'], combine=True)
        clusters_regions = clusters[probe]['region'][clusters_pass]

        # Loop over regions
        for r, region in enumerate(np.unique(clusters_regions)):
            if region == 'root':
                continue

            # Select spikes and clusters in this brain region
            clusters_in_region = clusters_pass[clusters_regions == region]
            spks_region = spikes[probe].times[np.isin(spikes[probe].clusters, clusters_in_region)]
            clus_region = spikes[probe].clusters[np.isin(spikes[probe].clusters, clusters_in_region)]
            if len(clusters_in_region) < MIN_NEURONS:
                continue

            # Exclude neurons with too low firing rates
            excl_neurons = []
            for n, neuron_id in enumerate(clusters_in_region):
                if np.sum(clus_region == neuron_id) / 360 < MIN_FR:
                    excl_neurons.append(neuron_id)
            clusters_in_region = np.setdiff1d(clusters_in_region, excl_neurons)

            # Get smoothed firing rates
            peths, _ = calculate_peths(spks_region, clus_region, clusters_in_region,
                                       opto_train_times, T_BEFORE, T_AFTER, BIN_SIZE, SMOOTHING)
            pop_vector = peths['means'].T
            time = peths['tscale']

            # Normalize data
            ss = StandardScaler(with_mean=True, with_std=True)
            pop_vector_norm = ss.fit_transform(pop_vector)

            # Perform PCA
            pca_proj = pca.fit_transform(pop_vector_norm)

            # Plot result
            colors, dpi = figure_style()
            f = plt.figure(figsize=(6, 3), dpi=dpi)
            ax = plt.axes(projection='3d')
            p = ax.scatter3D(pca_proj[:, 0], pca_proj[:, 1], pca_proj[:, 2], c=peths['tscale'],
                             cmap='twilight_r')
            ax.set(xlabel='PCA 1', ylabel='PCA 2', zlabel='PCA 3', title=f'{region}')
            axins = inset_axes(ax, width="5%", height="80%", loc='upper left',
                               bbox_to_anchor=(1.3, 0., 1, 1),
                               bbox_transform=ax.transAxes,
                               borderpad=0,)
            cbar = f.colorbar(p, cax=axins, shrink=0.6)
            cbar.ax.set_ylabel('Time (s)', rotation=270, labelpad=18)
            plt.savefig(join(fig_path, 'SinglePlots', f'{region}_{subject}_{date}.jpg'), dpi=300)
            plt.savefig(join(fig_path, 'SinglePlots', f'{region}_{subject}_{date}.pdf'))
            plt.close(f)

            f, ax1 = plt.subplots(1, 1, figsize=(3, 3), dpi=dpi)
            ax1.plot(time, pca_proj[:, 0])
            ax1.plot([0, 0], ax1.get_ylim(), ls='--', color='grey')
            ax1.set(ylabel='First principal component', xlabel='Time (s)', title=f'{region}')
            sns.despine(trim=True)
            plt.tight_layout()
            plt.savefig(join(fig_path, 'SinglePlots', f'{region}_{subject}_{date}_PCA1.jpg'), dpi=300)
            plt.savefig(join(fig_path, 'SinglePlots', f'{region}_{subject}_{date}_PCA1.pdf'))
            plt.close(f)

            # Add to dataframe
            pca_df = pca_df.append(pd.DataFrame(data={
                'subject': subject, 'date': date, 'eid': eid, 'probe': probe, 'region': region,
                'time': time, 'n_neurons': np.sum(clusters_regions == region),
                'pca1': pca_proj[:, 0], 'pca2': pca_proj[:, 1], 'pca3': pca_proj[:, 2]}))

        pca_df.to_csv(join(save_path, 'pca_regions.csv'), index=False)
pca_df = pca_df.reset_index()
pca_df.to_csv(join(save_path, 'pca_regions.csv'), index=False)

# %% Plot

f, ax1 = plt.subplots(1, 1)
sns.lineplot(x='pca1', y='pca2', data=pca_df[pca_df['region'] == 'ORBl'], estimator=None, units='subject', hue='subject')
#sns.lineplot(x='pca1', y='pca2', data=pca_df, estimator=None, units='region', hue='region')



