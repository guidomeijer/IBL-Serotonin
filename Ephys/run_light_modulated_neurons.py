#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:17:26 2021
By: Guido Meijer
"""

import numpy as np
from os.path import join, isdir
import matplotlib.pyplot as plt
from glob import glob
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
from os import mkdir
from ibllib.io import spikeglx
from brainbox.task.closed_loop import roc_single_event
from my_functions import figure_style
import brainbox.io.one as bbone
from brainbox.plot import peri_event_time_histogram
from serotonin_functions import paths, remap
from oneibl.one import ONE
one = ONE()

# Settings
T_BEFORE = 1  # for plotting
T_AFTER = 2
PRE_TIME = [0.5, 0]  # for significance testing
POST_TIME = [0, 0.5]
BIN_SIZE = 0.05
PERMUTATIONS = 500
_, fig_path, save_path = paths()
fig_path = join(fig_path, '5HT', 'light-modulated-neurons')
save_path = join(save_path, '5HT')

# Query sessions
eids = one.search(task_protocol='_iblrig_tasks_opto_ephysChoiceWorld', dataset_types=['spikes.times'])

light_neurons = pd.DataFrame()
for i, eid in enumerate(eids):

    # Get session details
    ses_details = one.get_details(eid)
    subject = ses_details['subject']
    date = ses_details['start_time'][:10]

    # Load in laser pulses
    one.load(eid, dataset_types=['ephysData.raw.nidq', 'ephysData.raw.meta', 'ephysData.raw.ch',
                                 'ephysData.raw.sync', 'ephysData.raw.timestamps'], download_only=True)
    session_path = one.path_from_eid(eid)
    nidq_file = glob(str(session_path.joinpath('raw_ephys_data/_spikeglx_ephysData_g*_t0.nidq.cbin')))[0]
    sr = spikeglx.Reader(nidq_file)
    offset = int((sr.shape[0] / sr.fs - 720) * sr.fs)
    opto_trace = sr.read_sync_analog(slice(offset, offset + int(720 * sr.fs)))[:, 1]
    opto_times = np.arange(offset, offset + len(opto_trace)) / sr.fs

    # Get start times of pulse trains
    opto_high_times = opto_times[opto_trace > 1]
    if len(opto_high_times) == 0:
        print(f'No pulses found for {eid}')
        continue
    opto_train_times = opto_high_times[np.concatenate(([True], np.diff(opto_high_times) > 1))]

    # Load in spikes
    spikes, clusters, channels = bbone.load_spike_sorting_with_channel(eid, aligned=True, one=one)

    for p, probe in enumerate(spikes.keys()):
        if 'acronym' not in clusters[probe].keys():
            print(f'No brain regions found for {eid}')
            continue

        # Select spikes of passive period
        spikes[probe].clusters = spikes[probe].clusters[spikes[probe].times > opto_times[0]]
        spikes[probe].times = spikes[probe].times[spikes[probe].times > opto_times[0]]

        # Filter neurons that pass QC
        clusters_pass = np.where(clusters[probe]['metrics']['label'] == 1)[0]
        spikes[probe].times = spikes[probe].times[np.isin(spikes[probe].clusters, clusters_pass)]
        spikes[probe].clusters = spikes[probe].clusters[np.isin(spikes[probe].clusters, clusters_pass)]

        # Determine significant neurons
        print('Calculating significant neurons..')
        roc_auc, cluster_ids = roc_single_event(spikes[probe].times, spikes[probe].clusters,
                                                opto_train_times, pre_time=PRE_TIME, post_time=POST_TIME)
        roc_auc_permut = np.zeros([PERMUTATIONS, len(np.unique(spikes[probe].clusters))])
        for k in range(PERMUTATIONS):
            roc_auc_permut[k, :] = roc_single_event(
                spikes[probe].times, spikes[probe].clusters,
                np.random.uniform(low=opto_times[0], high=opto_times[-1], size=opto_train_times.shape[0]),
                pre_time=PRE_TIME, post_time=POST_TIME)[0]
        significant = ((roc_auc > np.percentile(roc_auc_permut, 97.5, axis=0))
                       | (roc_auc < np.percentile(roc_auc_permut, 2.5, axis=0)))

        cluster_regions = remap(clusters[probe].atlas_id[cluster_ids])
        for r, region in enumerate(np.unique(cluster_regions)):

            # Add to dataframe
            light_neurons = light_neurons.append(pd.DataFrame(index=[light_neurons.shape[0] + 1], data={
                'subject': subject, 'date': date, 'eid': eid, 'region': region,
                'n_neurons': np.sum(cluster_regions == region),
                'perc_sig': (np.sum(significant[cluster_regions == region]) / np.sum(cluster_regions == region)) * 100,
                'perc_enh': (np.sum(roc_auc[cluster_regions == region] > np.percentile(roc_auc_permut[:, cluster_regions == region], 97.5, axis=0)) / np.sum(cluster_regions == region)) * 100,
                'perc_supp': (np.sum(roc_auc[cluster_regions == region] < np.percentile(roc_auc_permut[:, cluster_regions == region], 2.5, axis=0)) / np.sum(cluster_regions == region)) * 100}))

        # Plot light modulated units
        for n, cluster in enumerate(cluster_ids[significant]):
            if not isdir(join(fig_path, f'{cluster_regions[cluster_ids == cluster][0]}')):
                mkdir(join(fig_path, f'{cluster_regions[cluster_ids == cluster][0]}'))

            # Plot PSTH
            figure_style()
            p, ax = plt.subplots()
            peri_event_time_histogram(spikes[probe].times, spikes[probe].clusters, opto_train_times,
                                      cluster, t_before=T_BEFORE, t_after=T_AFTER, bin_size=BIN_SIZE,
                                      include_raster=True, error_bars='sem', ax=ax,
                                      pethline_kwargs={'color': 'black', 'lw': 2},
                                      errbar_kwargs={'color': 'black', 'alpha': 0.3},
                                      eventline_kwargs={'lw': 0})
            ax.set(ylim=[ax.get_ylim()[0], ax.get_ylim()[1] + ax.get_ylim()[1] * 0.2])
            ax.plot([0, 1], [ax.get_ylim()[1] - ax.get_ylim()[1] * 0.05,
                             ax.get_ylim()[1] - ax.get_ylim()[1] * 0.05], lw=4, color='royalblue')
            ax.set(ylabel='spikes/s', xlabel='Time (s)',
                   yticks=np.linspace(0, np.round(ax.get_ylim()[1]), 3))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            plt.tight_layout()
            plt.savefig(join(fig_path, cluster_regions[cluster_ids == cluster][0],
                             f'{subject}_{date}_{probe}_neuron{cluster}'))
            plt.close(p)

light_neurons.to_csv(join(save_path, 'light_modulated_neurons.csv'))
