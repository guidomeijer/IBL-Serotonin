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
from brainbox.metrics.single_units import spike_sorting_metrics
import brainbox.io.one as bbone
from brainbox.population.decode import get_spike_counts_in_bins
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from serotonin_functions import (paths, remap, query_ephys_sessions, load_passive_opto_times,
                                 get_artifact_neurons, figure_style)
from one.api import ONE
from ibllib.atlas import AllenAtlas
lda = LinearDiscriminantAnalysis()
one = ONE()
ba = AllenAtlas()

# Settings
MIN_NEURONS = 5  # per region
PLOT = True
T_BEFORE = 0.5
T_AFTER = 1.5
BIN_SIZE = 0.2
_, fig_path, save_path = paths()
fig_path = join(fig_path, 'Ephys', 'LDA', 'LDA_passive_regions')
save_path = join(save_path)

# Query sessions
eids, _ = query_ephys_sessions(one=one)

# Get binning time vectors
BIN_CENTERS = np.arange(-T_BEFORE, T_AFTER, BIN_SIZE) + (BIN_SIZE / 2)

lda_dist_df = pd.DataFrame()
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

        # Get regions from Beryl atlas
        clusters[probe]['acronym'] = remap(clusters[probe]['atlas_id'])
        clusters_regions = clusters[probe]['acronym'][clusters_pass]

        # Get a number of random onset times in the spontaneous activity as control
        control_times = np.random.uniform(low=start_passive, high=opto_train_times[0],
                                          size=opto_train_times.shape[0], )
        all_times = np.concatenate((control_times, opto_train_times))
        laser_on = np.concatenate((np.zeros(control_times.shape[0]), np.ones(control_times.shape[0])))

        # Loop over regions
        for r, region in enumerate(np.unique(clusters_regions)):

            # Select spikes and clusters in this brain region
            clusters_in_region = clusters_pass[clusters_regions == region]
            spks_region = spikes[probe].times[np.isin(spikes[probe].clusters, clusters_in_region)]
            clus_region = spikes[probe].clusters[np.isin(spikes[probe].clusters, clusters_in_region)]
            if len(clusters_in_region) < MIN_NEURONS:
                continue

            lda_dist = np.empty(BIN_CENTERS.shape[0])
            for b, bin_center in enumerate(BIN_CENTERS):
                times = np.column_stack((((all_times + bin_center) - (BIN_SIZE / 2)),
                                         ((all_times + bin_center) + (BIN_SIZE / 2))))
                pop_vector, cluster_ids = get_spike_counts_in_bins(spks_region, clus_region, times)
                pop_vector = pop_vector.T
                if np.sum(pop_vector) > 0:
                    lda_projection = lda.fit_transform(pop_vector, laser_on)
                    lda_dist[b] = (np.abs(np.nanmean(lda_projection[laser_on == 0]))
                                   + np.abs(np.nanmean(lda_projection[laser_on == 1])))
                else:
                    lda_dist[b] = np.nan

            # Plot
            if PLOT:
                colors, dpi = figure_style()
                f, ax1 = plt.subplots(1, 1, figsize=(3, 3), dpi=dpi)
                ax1.plot(BIN_CENTERS, lda_dist, zorder=2)
                ax1.plot([0, 0], ax1.get_ylim(), ls='--', color='red', zorder=1)
                ax1.set(xlabel='Time (s)', ylabel='LDA score', title=f'{region}')
                plt.tight_layout()
                sns.despine(trim=True)
                plt.savefig(join(fig_path, f'{region}_{subject}_{date}_{probe}'), dpi=300)
                plt.close(f)

            # Add to dataframe
            lda_dist_df = lda_dist_df.append(pd.DataFrame(data={
                'subject': subject, 'date': date, 'probe': probe, 'eid': eid,
                'lda_dist': lda_dist, 'region': region, 'time': BIN_CENTERS}))

lda_dist_df.to_csv(join(save_path, 'lda_opto_per_region.csv'), index=False)

# %% Plot
lda_dist_df = lda_dist_df.reset_index(drop=True)
lda_dist_df = lda_dist_df[lda_dist_df['region'] != 'root']

colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(5, 5), dpi=dpi)
sns.lineplot(x='time', y='lda_dist', data=lda_dist_df, hue='region', estimator=None, units='eid',
             palette='Set3')


