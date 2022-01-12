#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:17:26 2021
By: Guido Meijer
"""

import numpy as np
from os.path import join, isdir
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
from scipy.optimize import curve_fit
import seaborn as sns
from scipy.stats import pearsonr
from brainbox.metrics.single_units import spike_sorting_metrics
from brainbox.task.closed_loop import roc_single_event
from serotonin_functions import figure_style
import brainbox.io.one as bbone
from brainbox.population.decode import get_spike_counts_in_bins
from brainbox.plot import peri_event_time_histogram
from sklearn.decomposition import PCA
from serotonin_functions import (paths, combine_regions, query_ephys_sessions, load_passive_opto_times,
                                 get_artifact_neurons)
from one.api import ONE
from ibllib.atlas import AllenAtlas
one = ONE()
ba = AllenAtlas()
pca = PCA(n_components=3)

# Settings
MIN_NEURONS = 10  # per region
BIN_SIZE = 0.3  # sec
BIN_CENTERS = np.arange(-1, 2.1, 0.2)  # sec
PLOT = True
_, fig_path, save_path = paths()
fig_path = join(fig_path, 'Ephys', 'PCA')

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
        clusters[probe]['acronym'] = combine_regions(clusters[probe]['acronym'])
        clusters_regions = clusters[probe]['acronym'][clusters_pass]

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

            # Do PCA
            pop_vector = np.zeros((BIN_CENTERS.shape[0], clusters_in_region.shape[0]))
            for t, bin_center in enumerate(BIN_CENTERS):

                # Get population vector for time bin
                times = np.column_stack(((opto_train_times + (bin_center - (BIN_SIZE / 2))),
                                        (opto_train_times + (bin_center + (BIN_SIZE / 2)))))
                this_pop_vector, _ = get_spike_counts_in_bins(spks_region, clus_region, times)
                pop_vector[t, :] = np.median(this_pop_vector.T, axis=0)
            pca_proj = pca.fit_transform(pop_vector)

            # Add to dataframe
            pca_df = pca_df.append(pd.DataFrame(data={
                'subject': subject, 'date': date, 'eid': eid, 'probe': probe, 'region': region,
                'time': bin_center, 'n_neurons': np.sum(clusters_regions == region),
                'pca1': pca_proj[:, 0], 'pca2': pca_proj[:, 1], 'pca3': pca_proj[:, 2]}))

        pca_df.to_csv(join(save_path, 'pca_regions.csv'), index=False)
pca_df = pca_df.reset_index()
pca_df.to_csv(join(save_path, 'pca_regions.csv'), index=False)

# %% Plot

f, ax1 = plt.subplots(1, 1)
sns.lineplot(x='pca1', y='pca2', data=pca_df[pca_df['region'] == 'ORBl'], estimator=None, units='subject', hue='subject')
#sns.lineplot(x='pca1', y='pca2', data=pca_df, estimator=None, units='region', hue='region')



