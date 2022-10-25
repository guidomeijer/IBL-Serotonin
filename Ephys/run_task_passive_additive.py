#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 10:26:50 2022
By: Guido Meijer
"""

import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.objects as so
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from os.path import join
from brainbox.io.one import SpikeSortingLoader
from brainbox.singlecell import calculate_peths
from serotonin_functions import (paths, figure_style, get_full_region_name, load_subjects,
                                 high_level_regions, load_trials, get_neuron_qc,
                                 load_passive_opto_times)
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()
fig_path, save_path = paths()

# Settings
PRE_TIME = 1
POST_TIME = 2
BIN_SIZE = 0.05
SMOOTHING = 0.025
BASELINE = [-0.6, -0.1]
PLOT = True

# Load in results
stim_neurons = pd.read_csv(join(save_path, 'task_modulated_neurons.csv'))
light_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))
all_neurons = pd.merge(stim_neurons, light_neurons,
                       on=['subject', 'date', 'neuron_id', 'eid', 'pid', 'region', 'probe'])
all_neurons['high_level_region'] = high_level_regions(all_neurons['region'])
all_neurons['passive_task_mod'] = all_neurons['modulated'] | all_neurons['opto_modulated']
all_neurons.loc[all_neurons['modulated'] & all_neurons['opto_modulated'], 'modulation_type'] = 'Both'
all_neurons.loc[~all_neurons['modulated'] & all_neurons['opto_modulated'], 'modulation_type'] = 'Task'
all_neurons.loc[all_neurons['modulated'] & ~all_neurons['opto_modulated'], 'modulation_type'] = 'Passive'

for i, pid in enumerate(np.unique(all_neurons['pid'])):

    # Get session details
    eid = one.pid2eid(pid)[0]
    ses_details = one.get_details(eid)
    subject = ses_details['subject']
    date = ses_details['start_time'][:10]
    print(f'Starting {subject}, {date}')

    try:
        # Load trials dataframe
        trials = load_trials(eid, laser_stimulation=True, patch_old_opto=False, one=one)

        # Load in passive pulses
        opto_times, _ = load_passive_opto_times(eid, one=one)

        # Load in spikes
        sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
        spikes, clusters, channels = sl.load_spike_sorting()
        clusters = sl.merge_clusters(spikes, clusters, channels)
    except Exception as err:
        print(err)

    # Filter neurons that pass QC
    qc_metrics = get_neuron_qc(pid, one=one, ba=ba)
    clusters_pass = np.where(qc_metrics['label'] == 1)[0]
    spikes.times = spikes.times[np.isin(spikes.clusters, clusters_pass)]
    spikes.clusters = spikes.clusters[np.isin(spikes.clusters, clusters_pass)]
    if len(spikes.clusters) == 0:
        print('No neurons found, skipping')
        continue

    # Get modulated neurons
    mod_neurons = all_neurons.loc[(all_neurons['pid'] == pid) & all_neurons['passive_task_mod'],
                                  'neuron_id'].values
    mod_type = all_neurons.loc[(all_neurons['pid'] == pid) & all_neurons['passive_task_mod'],
                               'modulation_type'].values

    # Get PSTH of passive and task
    peth_passive_bl, _ = calculate_peths(spikes.times, spikes.clusters, mod_neurons, opto_times,
                                         pre_time=PRE_TIME, post_time=POST_TIME, bin_size=BIN_SIZE,
                                         smoothing=SMOOTHING)
    peth_task_bl, _ = calculate_peths(spikes.times, spikes.clusters, mod_neurons,
                                      trials.loc[trials['laser_stimulation'] == 0, 'stimOn_times'],
                                      pre_time=PRE_TIME, post_time=POST_TIME, bin_size=BIN_SIZE,
                                      smoothing=SMOOTHING)
    peth_opto_task_bl, _ = calculate_peths(spikes.times, spikes.clusters, mod_neurons,
                                           trials.loc[trials['laser_stimulation'] == 1, 'stimOn_times'],
                                           pre_time=PRE_TIME, post_time=POST_TIME, bin_size=BIN_SIZE,
                                           smoothing=SMOOTHING)
    tscale = peth_passive_bl['tscale']

    # Baseline subtraction
    for j, neuron_id in enumerate(mod_neurons):
        baseline_ind = (tscale > BASELINE[0]) & (tscale < BASELINE[1])
        peth_passive_bl['means'][j, :] = (peth_passive_bl['means'][j, :]
                                          - np.mean(peth_passive_bl['means'][j, baseline_ind]))
        peth_task_bl['means'][j, :] = (peth_task_bl['means'][j, :]
                                       - np.mean(peth_task_bl['means'][j, baseline_ind]))
        peth_opto_task_bl['means'][j, :] = (peth_opto_task_bl['means'][j, :]
                                            - np.mean(peth_opto_task_bl['means'][j, baseline_ind]))

    peth_passive, _ = calculate_peths(spikes.times, spikes.clusters, mod_neurons, opto_times,
                                      pre_time=PRE_TIME, post_time=POST_TIME, bin_size=BIN_SIZE,
                                     smoothing=SMOOTHING)
    peth_task, _ = calculate_peths(spikes.times, spikes.clusters, mod_neurons,
                                   trials.loc[trials['laser_stimulation'] == 0, 'stimOn_times'],
                                   pre_time=PRE_TIME, post_time=POST_TIME, bin_size=BIN_SIZE,
                                   smoothing=SMOOTHING)
    peth_opto_task, _ = calculate_peths(spikes.times, spikes.clusters, mod_neurons,
                                        trials.loc[trials['laser_stimulation'] == 1, 'stimOn_times'],
                                        pre_time=PRE_TIME, post_time=POST_TIME, bin_size=BIN_SIZE,
                                        smoothing=SMOOTHING)

    # Plot
    if PLOT:
        colors, dpi = figure_style()
        for j, neuron_id in enumerate(mod_neurons):
            f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7, 1.75), dpi=dpi)

            ax1.plot(tscale, peth_passive['means'][j, :], color=colors['general'], label='Only opto')
            err_bars = peth_passive.stds[0, :] / np.sqrt(len(opto_times))
            ax1.fill_between(tscale, peth_passive['means'][j, :]-err_bars,
                            peth_passive['means'][j, :]+err_bars, color=colors['general'], alpha=0.3)

            ax1.plot(tscale, peth_task['means'][j, :], color='grey', label='Only task')
            err_bars = peth_task.stds[0, :] / np.sqrt(np.sum(trials['laser_stimulation'] == 0))
            ax1.fill_between(tscale, peth_task['means'][j, :]-err_bars,
                            peth_task['means'][j, :]+err_bars, color='grey', alpha=0.3)

            ax1.plot(tscale, peth_opto_task['means'][j, :], color=colors['stim'], label='Task+opto')
            err_bars = peth_opto_task.stds[0, :] / np.sqrt(np.sum(trials['laser_stimulation'] == 1))
            ax1.fill_between(tscale, peth_opto_task['means'][j, :]-err_bars,
                            peth_opto_task['means'][j, :]+err_bars, color=colors['stim'], alpha=0.3)

            ax1.set(ylabel='Firing rate (spks/s)', xlabel='Time (s)', xticks=[-1, 0, 1, 2],
                    yticks=[0, np.ceil(ax1.get_ylim()[1])/2, np.ceil(ax1.get_ylim()[1])],
                    ylim=[0, np.ceil(ax1.get_ylim()[1])])
            ax1.legend(frameon=False, prop={'size': 5}, bbox_to_anchor=(1.4, 1))

            ax2.plot(tscale, peth_passive_bl['means'][j, :], color=colors['general'], label='Only opto')
            err_bars = peth_passive_bl.stds[0, :] / np.sqrt(len(opto_times))
            ax2.fill_between(tscale, peth_passive_bl['means'][j, :]-err_bars,
                            peth_passive_bl['means'][j, :]+err_bars, color=colors['general'], alpha=0.3)

            ax2.plot(tscale, peth_task_bl['means'][j, :], color='grey', label='Only task')
            err_bars = peth_task.stds[0, :] / np.sqrt(np.sum(trials['laser_stimulation'] == 0))
            ax2.fill_between(tscale, peth_task_bl['means'][j, :]-err_bars,
                            peth_task_bl['means'][j, :]+err_bars, color='grey', alpha=0.3)

            ax2.plot(tscale, peth_opto_task_bl['means'][j, :], color=colors['stim'], label='Task+opto')
            err_bars = peth_opto_task_bl.stds[0, :] / np.sqrt(np.sum(trials['laser_stimulation'] == 1))
            ax2.fill_between(tscale, peth_opto_task_bl['means'][j, :]-err_bars,
                            peth_opto_task_bl['means'][j, :]+err_bars, color=colors['stim'], alpha=0.3)

            ax2.set(ylabel='Baseline subtracted (spks/s)', xlabel='Time (s)', xticks=[-1, 0, 1, 2],
                    yticks=[np.floor(ax2.get_ylim()[0]), 0, np.ceil(ax2.get_ylim()[1])],
                    ylim=[np.floor(ax2.get_ylim()[0]), np.ceil(ax2.get_ylim()[1])])
            ax2.legend(frameon=False, prop={'size': 5}, bbox_to_anchor=(1.4, 1))

            ax3.plot(tscale, peth_task_bl['means'][j, :], color='grey', label='Task')
            err_bars = peth_task_bl.stds[0, :] / np.sqrt(np.sum(trials['laser_stimulation'] == 1))
            ax3.fill_between(tscale, peth_task_bl['means'][j, :]-err_bars,
                            peth_task_bl['means'][j, :]+err_bars, color='grey', alpha=0.3)

            ax3.plot(tscale, peth_opto_task_bl['means'][j, :] - peth_passive_bl['means'][j, :],
                     color='hotpink', label='(task+opto)\n-only opto')

            ax3.set(ylabel='Baseline subtracted (spks/s)', xlabel='Time (s)', xticks=[-1, 0, 1, 2],
                    yticks=[np.floor(ax3.get_ylim()[0]), 0, np.ceil(ax3.get_ylim()[1])],
                    ylim=[np.floor(ax3.get_ylim()[0]), np.ceil(ax3.get_ylim()[1])])
            ax3.legend(frameon=False, prop={'size': 5}, bbox_to_anchor=(1.4, 1))

            sns.despine(trim=True)
            plt.tight_layout()

            plt.savefig(join(fig_path, 'Ephys', 'SingleNeurons', 'PassiveTaskAdditive', mod_type[j],
                             f'{subject}_{date}_{neuron_id}.jpg'), dpi=600)

            plt.close(f)




