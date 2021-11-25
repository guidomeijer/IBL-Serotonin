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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from serotonin_functions import (paths, remap, query_ephys_sessions, load_opto_times,
                                 get_artifact_neurons)
from one.api import ONE
from ibllib.atlas import AllenAtlas
lda = LinearDiscriminantAnalysis()
one = ONE()
ba = AllenAtlas()

# Settings
MIN_NEURONS = 5  # per region
BIN_SIZE = 50  # in ms
BIN_START = np.arange(-1000, 1, 50)  # ms relative to pulse
DECAY_TIMES = np.arange(-1, 2.1, 0.2)
PLOT = True
_, fig_path, save_path = paths()
fig_path = join(fig_path, 'Ephys', 'AutoCorrelation')

# Query sessions
eids, _ = query_ephys_sessions(one=one)

# Convert into seconds
BIN_SIZE_S = BIN_SIZE / 1000
BIN_START_S = BIN_START / 1000


def exponential_decay(x, A, tau, B):
    y = (A * np.exp(-(x / tau))) + B
    return y


timeconstant_df = pd.DataFrame()
artifact_neurons = get_artifact_neurons()
for i, eid in enumerate(eids):

    # Get session details
    ses_details = one.get_details(eid)
    subject = ses_details['subject']
    date = ses_details['start_time'][:10]
    print(f'Starting {subject}, {date}')

    # Load in laser pulse times
    try:
        opto_train_times = load_opto_times(eid, one=one)
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
        clusters[probe]['acronym'] = remap(clusters[probe]['atlas_id'])
        brain_region = clusters[probe]['acronym'][clusters_pass]

        for t, decay_time in enumerate(DECAY_TIMES):

            # Get relative times
            these_times = opto_train_times + decay_time

            # Pre-allocate 3D matrix (bin1 x bin2 x neuron)
            corr_matrix = np.empty((BIN_START.shape[0], BIN_START.shape[0], np.unique(spikes[probe].clusters).shape[0]))

            # Correlate every timebin with every other timebin
            for k, bin1 in enumerate(BIN_START_S):
                for m, bin2 in enumerate(BIN_START_S):
                    # Get spike counts of all neurons during bin 1
                    times1 = np.column_stack(((these_times + bin1),
                                             (these_times + (bin1 + BIN_SIZE_S))))
                    pop_vector1, cluster_ids = get_spike_counts_in_bins(
                                                spikes[probe].times, spikes[probe].clusters, times1)

                    # Get spike counts of all neurons during bin 2
                    times2 = np.column_stack(((these_times + bin2),
                                             (these_times + (bin2 + BIN_SIZE_S))))
                    pop_vector2, cluster_ids = get_spike_counts_in_bins(
                                                spikes[probe].times, spikes[probe].clusters, times2)

                    # Correlate the two bins for each neuron
                    for n, cluster in enumerate(cluster_ids):

                        # Correlate time bins
                        corr_matrix[k, m, n], _ = pearsonr(pop_vector1[n], pop_vector2[n])

            # Average matrix over neurons in brain region and fit exponential decay at population level
            for r, region in enumerate(np.unique(brain_region)):
                # Get average matrix
                mat = np.nanmean(corr_matrix[:, :, brain_region == region], axis=2)

                # Get flattened vector from matrix
                corr_bin = []
                for j in range(1, mat.shape[0]):
                    corr_bin.append(np.mean(np.diag(mat, j)))

                # Fit exponential decay starting at the bin with maximum autocorrelation decay, if that
                # doesn't work start at the max autocorrelation, if that doesn't work start at beginning
                fit_start = np.argmin(np.diff(corr_bin))
                if fit_start > BIN_START.shape[0]/3:
                    fit_start = np.argmax(corr_bin)
                if fit_start > BIN_START.shape[0]/3:
                    fit_start = 0
                delta_time = np.arange(BIN_SIZE, BIN_SIZE*corr_matrix.shape[0], BIN_SIZE)
                try:
                    fitted_params, _ = curve_fit(exponential_decay, delta_time[fit_start:],
                                                 corr_bin[fit_start:], [0.5, 200, 0])
                    timeconstant_df = timeconstant_df.append(pd.DataFrame(
                                            index=[0], data={
                                                'subject': subject, 'date': date, 'eid': eid,
                                                'probe': probe, 'region': region, 'time': decay_time,
                                                'n_neurons': np.sum(brain_region == region),
                                                'time_constant': fitted_params[1]}))
                except:
                    continue

        timeconstant_df.to_csv(join(save_path, 'time_constant_regions.csv'), index=False)
timeconstant_df = timeconstant_df.reset_index()
timeconstant_df.to_csv(join(save_path, 'time_constant_regions.csv'), index=False)

# %% Plot

f, ax1 = plt.subplots(1, 1)
sns.lineplot(x='time', y='time_constant', data=timeconstant_df[timeconstant_df['region'] == 'PO'],
             estimator=None, units='subject', hue='subject')


g = sns.FacetGrid(data=timeconstant_df, col='region', col_wrap=5)
g.map(sns.lineplot, 'time', 'time_constant')

