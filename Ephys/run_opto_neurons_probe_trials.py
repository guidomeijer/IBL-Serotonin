#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 14:00:33 2021
By: Guido Meijer
"""

import pandas as pd
from os.path import join
import numpy as np
from brainbox.task.closed_loop import differentiate_units
from brainbox.plot import peri_event_time_histogram
import brainbox.io.one as bbone
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from serotonin_functions import (query_ephys_sessions, load_trials, paths, remap, load_subjects,
                                 behavioral_criterion, figure_style)
from one.api import ONE
one = ONE()

# Settings
ARTIFACT_CUTOFF = 0.48
NEURON_QC = False
PLOT = True
TEST_PRE_TIME = 0
TEST_POST_TIME = 0.3
PLOT_PRE_TIME = 0.5
PLOT_POST_TIME = 2
BIN_SIZE = 0.05
MIN_TRIALS = 300
_, fig_path, save_path = paths()

# Load in data
eids, _, ses_subjects = query_ephys_sessions(one=one, return_subjects=True)
light_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))
subjects = load_subjects()

# Apply behavioral criterion
eids = behavioral_criterion(eids, min_trials=MIN_TRIALS)

# Loop over sessions
results_df = pd.DataFrame()
for i, eid in enumerate(eids):

    # Get session details
    ses_details = one.get_details(eid)
    subject = ses_details['subject']
    date = ses_details['start_time'][:10]
    sert_cre = subjects.loc[subjects['subject'] == subject, 'sert-cre'].values[0]
    print(f'Starting {subject}, {date}')

    # Load in trials
    try:
        if sert_cre:
            trials = load_trials(eid, laser_stimulation=True)
        else:
            trials = load_trials(eid, patch_old_opto=False, laser_stimulation=True)
    except:
        print('Could not load trials')
        continue
    if trials.shape[0] < MIN_TRIALS:
        continue

    # Select probe trials
    trials = trials[trials['signed_contrast'] == 0]
    block_id = (trials['probabilityLeft'] == 0.8).astype(int).values

    # Load in neural data
    spikes, clusters, channels = bbone.load_spike_sorting_with_channel(eid, aligned=True, one=one)

    for p, probe in enumerate(spikes.keys()):
        if 'acronym' not in clusters[probe].keys():
            print(f'No brain regions found for {eid}')
            continue

        # Filter neurons that pass QC
        if ('metrics' not in clusters[probe].keys()) or (NEURON_QC == False):
            print('No neuron QC, using all neurons')
            clusters_pass = np.unique(spikes[probe].clusters)
        else:
            clusters_pass = np.where(clusters[probe]['metrics']['label'] == 1)[0]
        spikes[probe].times = spikes[probe].times[np.isin(spikes[probe].clusters, clusters_pass)]
        spikes[probe].clusters = spikes[probe].clusters[np.isin(spikes[probe].clusters, clusters_pass)]
        if len(spikes[probe].clusters) == 0:
            continue

        # Exclude artifact neurons
        artifact_neurons = light_neurons.loc[(light_neurons['eid'] == eid) & (light_neurons['probe'] == probe)
                                             & (light_neurons['roc_auc'] > ARTIFACT_CUTOFF), 'cluster_id'].values
        spikes[probe].times = spikes[probe].times[~np.isin(spikes[probe].clusters, artifact_neurons)]
        spikes[probe].clusters = spikes[probe].clusters[~np.isin(spikes[probe].clusters, artifact_neurons)]
        print(f'Excluded {len(artifact_neurons)} light artifact neurons')

        # Remap to beryl atlas
        clusters[probe]['acronym'] = remap(clusters[probe]['atlas_id'])

        # Get neurons that show a different respons when stimulated in stim blocks or non-stim blocks
        sig_units = differentiate_units(spikes[probe].times, spikes[probe].clusters,
                                        trials['goCue_times'], trials['laser_stimulation'],
                                        pre_time=TEST_PRE_TIME, post_time=TEST_POST_TIME)[0]
        print(f'Found {len(sig_units)} laser modulated neurons')

        # Plot over significant units
        if PLOT:
            for n, neuron_id in enumerate(sig_units):

                # Plot PSTH
                colors, dpi = figure_style()
                p, ax = plt.subplots(1, 1, figsize=(2, 2), dpi=dpi)
                peri_event_time_histogram(spikes[probe].times, spikes[probe].clusters,
                                          trials.loc[trials['laser_stimulation'] == 1, 'goCue_times'],
                                          neuron_id, t_before=PLOT_PRE_TIME, t_after=PLOT_POST_TIME,
                                          bin_size=BIN_SIZE, error_bars='sem', ax=ax,
                                          pethline_kwargs={'color': colors['stim'], 'lw': 1},
                                          errbar_kwargs={'color': colors['stim'], 'alpha': 0.3},
                                          eventline_kwargs={'lw': 0})
                this_y_lim = ax.get_ylim()
                peri_event_time_histogram(spikes[probe].times, spikes[probe].clusters,
                                          trials.loc[trials['laser_stimulation'] == 0, 'goCue_times'],
                                          neuron_id, t_before=PLOT_PRE_TIME, t_after=PLOT_POST_TIME,
                                          bin_size=BIN_SIZE, error_bars='sem', ax=ax,
                                          pethline_kwargs={'color': colors['no-stim'], 'lw': 1},
                                          errbar_kwargs={'color': colors['no-stim'], 'alpha': 0.3},
                                          eventline_kwargs={'lw': 0})
                ax.set(ylim=[np.min([this_y_lim[0], ax.get_ylim()[0]]),
                             np.max([this_y_lim[1], ax.get_ylim()[1]]) + np.max([this_y_lim[1], ax.get_ylim()[1]]) * 0.2])
                ax.set(ylabel='spikes/s', xlabel='Time (s)',
                       yticks=np.linspace(0, np.round(ax.get_ylim()[1]), 3))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                plt.tight_layout()
                region = clusters[probe]['acronym'][neuron_id]
                plt.savefig(join(fig_path, 'Ephys', 'SingleNeurons', 'BlankTrialStim',
                                 f'{region}_{subject}_{date}_{probe}_neuron{neuron_id}'))
                plt.close(p)

        # Get neurons that show a different respons when stimulated in stim blocks or non-stim blocks
        sig_units = differentiate_units(spikes[probe].times, spikes[probe].clusters,
                                        trials.loc[trials['laser_stimulation'] == 1, 'goCue_times'],
                                        trials.loc[trials['laser_stimulation'] == 1, 'probe_trial'],
                                        pre_time=TEST_PRE_TIME, post_time=TEST_POST_TIME)[0]
        print(f'Found {len(sig_units)} block vs probe stimulated neurons')

        # Plot over significant units
        if PLOT:
            for n, neuron_id in enumerate(sig_units):

                # Plot PSTH
                colors, dpi = figure_style()
                p, ax = plt.subplots(1, 1, figsize=(2, 2), dpi=dpi)
                peri_event_time_histogram(spikes[probe].times, spikes[probe].clusters,
                                          trials.loc[(trials['laser_stimulation'] == 1) & (trials['probe_trial'] == 1), 'goCue_times'],
                                          neuron_id, t_before=PLOT_PRE_TIME, t_after=PLOT_POST_TIME,
                                          bin_size=BIN_SIZE, error_bars='sem', ax=ax,
                                          pethline_kwargs={'color': colors['probe'], 'lw': 1},
                                          errbar_kwargs={'color': colors['probe'], 'alpha': 0.3},
                                          eventline_kwargs={'lw': 0})
                peri_event_time_histogram(spikes[probe].times, spikes[probe].clusters,
                                          trials.loc[(trials['laser_stimulation'] == 1) & (trials['probe_trial'] == 0), 'goCue_times'],
                                          neuron_id, t_before=PLOT_PRE_TIME, t_after=PLOT_POST_TIME,
                                          bin_size=BIN_SIZE, error_bars='sem', ax=ax,
                                          pethline_kwargs={'color': colors['block'], 'lw': 1},
                                          errbar_kwargs={'color': colors['block'], 'alpha': 0.3},
                                          eventline_kwargs={'lw': 0})
                ax.set(ylim=[ax.get_ylim()[0], ax.get_ylim()[1] + ax.get_ylim()[1] * 0.2])
                ax.set(ylabel='spikes/s', xlabel='Time (s)',
                       yticks=np.linspace(0, np.round(ax.get_ylim()[1]), 3))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                plt.tight_layout()
                region = clusters[probe]['acronym'][neuron_id]
                plt.savefig(join(fig_path, 'Ephys', 'SingleNeurons', 'BlockProbeStim',
                                 f'{region}_{subject}_{date}_{probe}_neuron{neuron_id}'))
                plt.close(p)

        # Get neurons that show a different respons when stimulated in stim blocks or non-stim blocks
        sig_units = differentiate_units(spikes[probe].times, spikes[probe].clusters,
                                        trials.loc[trials['laser_stimulation'] == 0, 'goCue_times'],
                                        trials.loc[trials['laser_stimulation'] == 0, 'probe_trial'],
                                        pre_time=TEST_PRE_TIME, post_time=TEST_POST_TIME)[0]
        print(f'Found {len(sig_units)} block vs probe non-stimulated neurons')

        # Plot over significant units
        if PLOT:
            for n, neuron_id in enumerate(sig_units):

                # Plot PSTH
                colors, dpi = figure_style()
                p, ax = plt.subplots(1, 1, figsize=(2, 2), dpi=dpi)
                peri_event_time_histogram(spikes[probe].times, spikes[probe].clusters,
                                          trials.loc[(trials['laser_stimulation'] == 0) & (trials['probe_trial'] == 1), 'goCue_times'],
                                          neuron_id, t_before=PLOT_PRE_TIME, t_after=PLOT_POST_TIME,
                                          bin_size=BIN_SIZE, error_bars='sem', ax=ax,
                                          pethline_kwargs={'color': colors['probe'], 'lw': 1},
                                          errbar_kwargs={'color': colors['probe'], 'alpha': 0.3},
                                          eventline_kwargs={'lw': 0})
                peri_event_time_histogram(spikes[probe].times, spikes[probe].clusters,
                                          trials.loc[(trials['laser_stimulation'] == 0) & (trials['probe_trial'] == 0), 'goCue_times'],
                                          neuron_id, t_before=PLOT_PRE_TIME, t_after=PLOT_POST_TIME,
                                          bin_size=BIN_SIZE, error_bars='sem', ax=ax,
                                          pethline_kwargs={'color': colors['block'], 'lw': 1},
                                          errbar_kwargs={'color': colors['block'], 'alpha': 0.3},
                                          eventline_kwargs={'lw': 0})
                ax.set(ylim=[ax.get_ylim()[0], ax.get_ylim()[1] + ax.get_ylim()[1] * 0.2])
                ax.set(ylabel='spikes/s', xlabel='Time (s)',
                       yticks=np.linspace(0, np.round(ax.get_ylim()[1]), 3))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                plt.tight_layout()
                region = clusters[probe]['acronym'][neuron_id]
                plt.savefig(join(fig_path, 'Ephys', 'SingleNeurons', 'BlockProbeNoStim',
                                 f'{region}_{subject}_{date}_{probe}_neuron{neuron_id}'))
                plt.close(p)


