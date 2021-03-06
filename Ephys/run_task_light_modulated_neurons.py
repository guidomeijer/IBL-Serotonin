#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 10:03:53 2021
By: Guido Meijer
"""

import numpy as np
from os.path import join, isdir
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
from os import mkdir
import seaborn as sns
from brainbox.io.one import SpikeSortingLoader
from brainbox.metrics.single_units import spike_sorting_metrics
from brainbox.task.closed_loop import (responsive_units, roc_single_event,
                                       roc_between_two_events, generate_pseudo_blocks)
from brainbox.plot import peri_event_time_histogram
from serotonin_functions import paths, remap, query_ephys_sessions, load_trials, figure_style
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

# Settings
OVERWRITE = True
NEURON_QC = True
PLOT = True
ITERATIONS = 500
T_BEFORE = 1  # for plotting
T_AFTER = 2
PRE_TIME = [0.5, 0]  # for significance testing
POST_TIME = [0, 0.5]
BIN_SIZE = 0.05
fig_path, save_path = paths()
fig_path = join(fig_path, 'Ephys', 'SingleNeurons', 'TaskNeurons')

# Query sessions
rec = query_ephys_sessions(one=one)

if OVERWRITE:
    task_neurons = pd.DataFrame()
else:
    task_neurons = pd.read_csv(join(save_path, 'task_modulated_neurons.csv'))
    rec = rec[~rec['eid'].isin(task_neurons['eid'])]

for i in rec.index.values:

    # Get session details
    pid, eid, probe = rec.loc[i, 'pid'], rec.loc[i, 'eid'], rec.loc[i, 'probe']
    subject, date = rec.loc[i, 'subject'], rec.loc[i, 'date']

    # Get session details
    ses_details = one.get_details(eid)
    subject = ses_details['subject']
    date = ses_details['start_time'][:10]

    # Load trials dataframe
    try:
        trials = load_trials(eid, laser_stimulation=True, patch_old_opto=False, one=one)
    except:
        print('Could not load trials')
        continue
    if trials.shape[0] < 200:
        continue

    # Load in spikes
    sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
    spikes, clusters, channels = sl.load_spike_sorting()
    clusters = sl.merge_clusters(spikes, clusters, channels)

    # Filter neurons that pass QC
    if NEURON_QC:
        print('Calculating neuron QC metrics..')
        qc_metrics, _ = spike_sorting_metrics(spikes.times, spikes.clusters, spikes.amps, spikes.depths,
                                              cluster_ids=np.arange(clusters.channels.size))
        clusters_pass = np.where(qc_metrics['label'] == 1)[0]
    else:
        clusters_pass = np.unique(spikes.clusters)
    spikes.times = spikes.times[np.isin(spikes.clusters, clusters_pass)]
    spikes.clusters = spikes.clusters[np.isin(spikes.clusters, clusters_pass)]
    if len(spikes.clusters) == 0:
        continue

    # Determine stimulus responsive neurons
    _, _, p_values, _ = responsive_units(spikes.times, spikes.clusters, trials['goCue_times'],
                                         pre_time=PRE_TIME, post_time=POST_TIME)
    task_resp = p_values < 0.05
    roc_task, neuron_ids = roc_single_event(spikes.times, spikes.clusters, trials['goCue_times'],
                                            pre_time=PRE_TIME, post_time=POST_TIME)
    roc_task = 2 * (roc_task - 0.5)  # Recalculate modulation index

    # Determine stimulus evoked light modulated neurons
    print('Determining task opto modulated neurons..')
    roc_auc, neuron_ids = roc_between_two_events(spikes.times, spikes.clusters, trials['goCue_times'],
                                                 trials['laser_stimulation'], pre_time=PRE_TIME[1], post_time=POST_TIME[1])
    roc_stim_mod = 2 * (roc_auc - 0.5)  # Recalculate modulation index
    pseudo_roc = np.empty((ITERATIONS, neuron_ids.shape[0]))
    for k in range(ITERATIONS):
        pseudo_blocks = generate_pseudo_blocks(trials.shape[0], first5050=0)
        this_pseudo_roc = roc_between_two_events(spikes.times, spikes.clusters,
                                                 trials['goCue_times'], (pseudo_blocks == 0.2).astype(int),
                                                 pre_time=PRE_TIME[1], post_time=POST_TIME[1])[0]
        pseudo_roc[k, :] = 2 * (this_pseudo_roc - 0.5)  # Recalculate modulation index

    stim_mod = ((roc_stim_mod > np.percentile(pseudo_roc, 95, axis=0))
                | (roc_stim_mod < np.percentile(pseudo_roc, 5, axis=0)))
    print(f'Found {np.sum(stim_mod)} stim. modulated neurons')

    # Add results to df
    cluster_regions = remap(clusters.acronym[neuron_ids])
    task_neurons = pd.concat((task_neurons, pd.DataFrame(data={
        'subject': subject, 'date': date, 'eid': eid, 'probe': probe,
        'neuron_id': neuron_ids, 'region': cluster_regions,
        'task_responsive': task_resp, 'task_roc': roc_task,
        'opto_modulated': stim_mod, 'opto_mod_roc': roc_stim_mod})))

    if PLOT:
        for n, neuron_id in enumerate(neuron_ids[stim_mod]):
            if not isdir(join(fig_path, f'{cluster_regions[neuron_ids == neuron_id][0]}')):
                mkdir(join(fig_path, f'{cluster_regions[neuron_ids == neuron_id][0]}'))

            # Plot PSTH
            colors, dpi = figure_style()
            p, ax = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
            peri_event_time_histogram(spikes.times, spikes.clusters,
                                      trials.loc[trials['laser_stimulation'] == 1, 'goCue_times'],
                                      neuron_id, t_before=T_BEFORE, t_after=T_AFTER, bin_size=BIN_SIZE,
                                      include_raster=False, error_bars='sem', ax=ax,
                                      pethline_kwargs={'color': colors['stim'], 'lw': 1},
                                      errbar_kwargs={'color': colors['stim'], 'alpha': 0.3},
                                      eventline_kwargs={'lw': 0})
            this_y_lim = ax.get_ylim()
            peri_event_time_histogram(spikes.times, spikes.clusters,
                                      trials.loc[trials['laser_stimulation'] == 0, 'goCue_times'],
                                      neuron_id, t_before=T_BEFORE, t_after=T_AFTER, bin_size=BIN_SIZE,
                                      include_raster=False, error_bars='sem', ax=ax,
                                      pethline_kwargs={'color': colors['no-stim'], 'lw': 1},
                                      errbar_kwargs={'color': colors['no-stim'], 'alpha': 0.3},
                                      eventline_kwargs={'lw': 0})
            ax.set(ylim=[np.min([this_y_lim[0], ax.get_ylim()[0]]),
                         np.max([this_y_lim[1], ax.get_ylim()[1]]) + np.max([this_y_lim[1], ax.get_ylim()[1]]) * 0.2])
            ax.set(ylabel='Firing rate (spikes/s)', xlabel='Time from trial start (s)',
                   yticks=np.linspace(0, np.round(ax.get_ylim()[1]), 3))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            sns.despine(trim=True, offset=2)
            plt.tight_layout()
            plt.savefig(join(fig_path, cluster_regions[neuron_ids == neuron_id][0],
                             f'{subject}_{date}_{probe}_neuron{neuron_id}.pdf'))
            plt.close(p)

task_neurons.to_csv(join(save_path, 'task_modulated_neurons.csv'))
