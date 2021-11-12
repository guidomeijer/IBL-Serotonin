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
from brainbox.metrics.single_units import spike_sorting_metrics
from brainbox.task.closed_loop import (responsive_units, differentiate_units, roc_single_event,
                                       roc_between_two_events, generate_pseudo_blocks)
import brainbox.io.one as bbone
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
_, fig_path, save_path = paths()
fig_path = join(fig_path, 'Ephys', 'SingleNeurons', 'StimNeurons')

# Query sessions
eids, _ = query_ephys_sessions(one=one)

if OVERWRITE:
    stim_neurons = pd.DataFrame()
else:
    stim_neurons = pd.read_csv(join(save_path, 'stim_modulated_neurons.csv'))
    eids = eids[~np.isin(eids, stim_neurons['eid'])]
for i, eid in enumerate(eids):

    # Get session details
    ses_details = one.get_details(eid)
    subject = ses_details['subject']
    date = ses_details['start_time'][:10]

    # Load trials dataframe
    try:
        trials = load_trials(eid, laser_stimulation=True, one=one)
    except:
        print('Could not load trials')
        continue
    if trials.shape[0] < 200:
        continue

    # Load in spikes
    spikes, clusters, channels = bbone.load_spike_sorting_with_channel(
        eid, aligned=True, one=one, dataset_types=['spikes.amps', 'spikes.depths'], brain_atlas=ba)

    for p, probe in enumerate(spikes.keys()):

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

        # Determine stimulus responsive neurons
        _, _, p_values, _ = responsive_units(spikes[probe].times, spikes[probe].clusters, trials['goCue_times'],
                                             pre_time=PRE_TIME, post_time=POST_TIME)
        task_resp = p_values < 0.05
        roc_task, neuron_ids = roc_single_event(spikes[probe].times, spikes[probe].clusters, trials['goCue_times'],
                                                pre_time=PRE_TIME, post_time=POST_TIME)
        roc_task = 2 * (roc_task - 0.5)  # Recalculate modulation index

        # Determine stimulus evoked light modulated neurons
        print('Determining stim modulated neurons..')
        roc_auc, neuron_ids = roc_between_two_events(spikes[probe].times, spikes[probe].clusters, trials['goCue_times'],
                                                     trials['laser_stimulation'], pre_time=PRE_TIME[1], post_time=POST_TIME[1])
        roc_stim_mod = 2 * (roc_auc - 0.5)  # Recalculate modulation index
        pseudo_roc = np.empty((ITERATIONS, neuron_ids.shape[0]))
        for k in range(ITERATIONS):
            pseudo_blocks = generate_pseudo_blocks(trials.shape[0], first5050=0)
            this_pseudo_roc = roc_between_two_events(spikes[probe].times, spikes[probe].clusters,
                                                     trials['goCue_times'], (pseudo_blocks == 0.2).astype(int),
                                                     pre_time=PRE_TIME[1], post_time=POST_TIME[1])[0]
            pseudo_roc[k, :] = 2 * (this_pseudo_roc - 0.5)  # Recalculate modulation index

        stim_mod = ((roc_stim_mod > np.percentile(pseudo_roc, 95, axis=0))
                    | (roc_stim_mod < np.percentile(pseudo_roc, 5, axis=0)))
        print(f'Found {stim_mod.shape[0]} stim. modulated neurons')

        # Add results to df
        cluster_regions = remap(clusters[probe].atlas_id[neuron_ids])
        stim_neurons = stim_neurons.append(pd.DataFrame(data={
            'subject': subject, 'date': date, 'eid': eid, 'probe': probe,
            'neuron_id': neuron_ids, 'region': cluster_regions,
            'task_responsive': task_resp, 'task_roc': roc_task,
            'stim_modulated': stim_mod, 'stim_mod_roc': roc_stim_mod}))

        if PLOT:
            for n, neuron_id in enumerate(neuron_ids[stim_mod]):
                if not isdir(join(fig_path, f'{cluster_regions[neuron_ids == neuron_id][0]}')):
                    mkdir(join(fig_path, f'{cluster_regions[neuron_ids == neuron_id][0]}'))

                # Plot PSTH
                colors, dpi = figure_style()
                p, ax = plt.subplots(1, 1, figsize=(2, 2), dpi=dpi)
                peri_event_time_histogram(spikes[probe].times, spikes[probe].clusters,
                                          trials.loc[trials['laser_stimulation'] == 1, 'goCue_times'],
                                          neuron_id, t_before=T_BEFORE, t_after=T_AFTER, bin_size=BIN_SIZE,
                                          include_raster=False, error_bars='sem', ax=ax,
                                          pethline_kwargs={'color': colors['stim'], 'lw': 1},
                                          errbar_kwargs={'color': colors['stim'], 'alpha': 0.3},
                                          eventline_kwargs={'lw': 0})
                this_y_lim = ax.get_ylim()
                peri_event_time_histogram(spikes[probe].times, spikes[probe].clusters,
                                          trials.loc[trials['laser_stimulation'] == 0, 'goCue_times'],
                                          neuron_id, t_before=T_BEFORE, t_after=T_AFTER, bin_size=BIN_SIZE,
                                          include_raster=False, error_bars='sem', ax=ax,
                                          pethline_kwargs={'color': colors['no-stim'], 'lw': 1},
                                          errbar_kwargs={'color': colors['no-stim'], 'alpha': 0.3},
                                          eventline_kwargs={'lw': 0})
                ax.set(ylim=[np.min([this_y_lim[0], ax.get_ylim()[0]]),
                             np.max([this_y_lim[1], ax.get_ylim()[1]]) + np.max([this_y_lim[1], ax.get_ylim()[1]]) * 0.2])
                ax.set(ylabel='spikes/s', xlabel='Time (s)',
                       yticks=np.linspace(0, np.round(ax.get_ylim()[1]), 3))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                plt.tight_layout()
                plt.savefig(join(fig_path, cluster_regions[neuron_ids == neuron_id][0],
                                 f'{subject}_{date}_{probe}_neuron{neuron_id}.pdf'))
                plt.close(p)

stim_neurons.to_csv(join(save_path, 'stim_modulated_neurons.csv'))
