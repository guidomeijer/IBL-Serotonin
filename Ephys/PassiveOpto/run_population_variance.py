#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:17:26 2021
By: Guido Meijer
"""

import numpy as np
from os.path import join, isdir
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from brainbox.population.decode import get_spike_counts_in_bins
import pandas as pd
from os import mkdir
import seaborn as sns
from brainbox.task.closed_loop import roc_single_event
from brainbox.metrics.single_units import spike_sorting_metrics
from serotonin_functions import figure_style
import brainbox.io.one as bbone
from brainbox.plot import peri_event_time_histogram
from serotonin_functions import (paths, remap, query_ephys_sessions, load_passive_opto_times,
                                 get_artifact_neurons, figure_style)
from one.api import ONE
from ibllib.atlas import AllenAtlas, BrainRegions
ba = AllenAtlas()
br = BrainRegions()
one = ONE()

# Settings
MIN_NEURONS = 10
MIN_FR = 0.01  # spks/s
BASELINE = [-1, -0.5]
STIM = [0.5, 1]
PLOT = True
OVERWRITE = True
NEURON_QC = True
_, fig_path, save_path = paths()
fig_path = join(fig_path, 'Ephys', 'Population', 'Variance')

# Query sessions
eids, _, subjects = query_ephys_sessions(return_subjects=True, one=one)

# Get artifact neurons
artifact_neurons = get_artifact_neurons()

pop_var_df = pd.DataFrame()
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
        eid, aligned=True, one=one, dataset_types=['spikes.amps', 'spikes.depths'], brain_atlas=ba)

    for p, probe in enumerate(spikes.keys()):
        if 'acronym' not in clusters[probe].keys():
            print(f'No brain regions found for {eid}')
            continue

        # Filter neurons that pass QC
        if NEURON_QC:
            print('Calculating neuron QC metrics..')
            qc_metrics, _ = spike_sorting_metrics(spikes[probe].times, spikes[probe].clusters,
                                                  spikes[probe].amps, spikes[probe].depths,
                                                  cluster_ids=np.arange(clusters[probe].channels.size))
            clusters_pass = np.where(qc_metrics['label'] == 1)[0]
        else:
            clusters_pass = np.unique(spikes[probe].clusters)
        spikes[probe].times = spikes[probe].times[np.isin(spikes[probe].clusters, clusters_pass)]
        spikes[probe].clusters = spikes[probe].clusters[np.isin(spikes[probe].clusters, clusters_pass)]
        if len(spikes[probe].clusters) == 0:
            continue

        # Exclude artifact neurons
        clusters_pass = np.array([i for i in clusters_pass if i not in artifact_neurons.loc[
            (artifact_neurons['eid'] == eid) & (artifact_neurons['probe'] == probe), 'neuron_id'].values])
        if clusters_pass.shape[0] == 0:
            continue

        # Get regions from Beryl atlas
        clusters[probe]['acronym'] = remap(clusters[probe]['atlas_id'], combine=True,
                                           split_thalamus=False)
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

            # Get population response
            times = np.column_stack(((opto_train_times + BASELINE[0]),
                                     (opto_train_times + BASELINE[1])))
            pop_vector, cluster_ids = get_spike_counts_in_bins(spks_region, clus_region, times)
            firing_rate = np.mean(pop_vector, axis=1)
            baseline_rate = np.log10(firing_rate[firing_rate > MIN_FR])

            times = np.column_stack(((opto_train_times + STIM[0]),
                                     (opto_train_times + STIM[1])))
            pop_vector, cluster_ids = get_spike_counts_in_bins(spks_region, clus_region, times)
            firing_rate = np.mean(pop_vector, axis=1)
            stim_rate = np.log10(firing_rate[firing_rate > MIN_FR])

            # Plot
            colors, dpi = figure_style()
            f, ax1 = plt.subplots(1, 1, figsize=(3, 3), dpi=dpi)
            x = np.linspace(-1, 2, 100)
            ax1.plot(x, stats.norm.pdf(x, np.mean(baseline_rate), np.std(baseline_rate)),
                     color=colors['no-stim'], label='Baseline')
            ax1.plot(x, stats.norm.pdf(x, np.mean(stim_rate), np.std(stim_rate)), color=colors['stim'],
                     label='Stim')
            ax1.legend(frameon=False)
            ax1.set(title=f'{region}', ylabel='Probability', xlabel='Firing rate (spks/s)',
                    xticks=[-1, 0, 1, 2], xticklabels=[0.1, 1, 10, 100])

            plt.tight_layout()
            sns.despine(trim=True)
            plt.savefig(join(fig_path, f'{subject}_{date}_{region}.pdf'))
            plt.savefig(join(fig_path, f'{subject}_{date}_{region}.jpg'), dpi=300)
            plt.close(f)

            # Add to dataframe
            pop_var_df = pd.concat((pop_var_df, pd.DataFrame(index=[pop_var_df.shape[0]+1], data={
                'pop_var_bl': np.std(baseline_rate), 'pop_var_stim': np.std(stim_rate),
                'pop_mean_bl': np.mean(baseline_rate), 'pop_mean_stim': np.mean(stim_rate),
                'subject': subject, 'date': date, 'region': region})))

pop_var_df['delta_var'] = pop_var_df['pop_var_stim'] - pop_var_df['pop_var_bl']
pop_var_df['delta_mean'] = pop_var_df['pop_mean_stim'] - pop_var_df['pop_mean_bl']

# %% Plot
colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(4, 4), dpi=dpi)
sns.scatterplot(x='delta_mean', y='delta_var', data=pop_var_df, ax=ax1, hue='region', palette='Set2')


