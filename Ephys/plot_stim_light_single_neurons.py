#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 15:53:54 2021
By: Guido Meijer
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from os import mkdir
from os.path import join, isdir
from matplotlib.ticker import FormatStrFormatter
from serotonin_functions import paths, figure_style, get_full_region_name, load_trials
import brainbox.io.one as bbone
from brainbox.plot import peri_event_time_histogram
from oneibl.one import ONE
one = ONE()

# Settings
T_BEFORE = 1
T_AFTER = 2
BIN_SIZE = 0.05

# Paths
_, fig_path, save_path = paths()
fig_path_light = join(fig_path, '5HT', 'stim-light-modulated-neurons')
fig_path_stim = join(fig_path, '5HT', 'stim-modulated-neurons')
save_path = join(save_path, '5HT')

# Load in results
stim_neurons = pd.read_csv(join(save_path, 'stim_light_modulated_neurons.csv'))

# Drop root
stim_neurons = stim_neurons.reset_index(drop=True)
stim_neurons = stim_neurons.drop(index=[i for i, j in enumerate(stim_neurons['region']) if 'root' in j])

for i, eid in enumerate(np.unique(stim_neurons['eid'])):

    # Get session details
    ses_details = one.get_details(eid)
    subject = ses_details['subject']
    date = ses_details['start_time'][:10]

    # Load trials dataframe
    trials = load_trials(eid, laser_stimulation=True, one=one)

    # Load in spikes
    spikes, clusters, channels = bbone.load_spike_sorting_with_channel(eid, aligned=True, one=one)

    for p, probe in enumerate(spikes.keys()):

        # Stimulus evoked light modulated neurons
        mod_l_neuron_ids = stim_neurons.loc[(stim_neurons['mod_l_light'] == True)
                                            & (stim_neurons['eid'] == eid)
                                            & (stim_neurons['probe'] == probe), 'cluster_id']
        for n, cluster in enumerate(mod_l_neuron_ids):
            acronym = clusters[probe].acronym[cluster].replace('/', '-')
            if not isdir(join(fig_path_light, f'{acronym}')):
                mkdir(join(fig_path_light, f'{acronym}'))

            # Plot PSTH
            figure_style()
            p, ax = plt.subplots()
            peri_event_time_histogram(
                spikes[probe].times, spikes[probe].clusters,
                trials.loc[(trials['signed_contrast'] == -1) & (trials['laser_stimulation'] == 0), 'stimOn_times'],
                cluster, t_before=T_BEFORE, t_after=T_AFTER, bin_size=BIN_SIZE,
                error_bars='sem', ax=ax,
                pethline_kwargs={'color': 'black', 'lw': 2},
                errbar_kwargs={'color': 'black', 'alpha': 0.3})
            ylim_1 = ax.get_ylim()
            peri_event_time_histogram(
                spikes[probe].times, spikes[probe].clusters,
                trials.loc[(trials['signed_contrast'] == -1) & (trials['laser_stimulation'] == 1), 'stimOn_times'],
                cluster, t_before=T_BEFORE, t_after=T_AFTER, bin_size=BIN_SIZE,
                error_bars='sem', ax=ax,
                pethline_kwargs={'color': 'royalblue', 'lw': 2},
                errbar_kwargs={'color': 'royalblue', 'alpha': 0.3})
            max_ylim = np.max(np.vstack([ylim_1, ax.get_ylim()]), axis=0)
            ax.set(ylabel='spikes/s', xlabel='Time (s)', title='Left stimulus', ylim=max_ylim,
                   yticks=np.linspace(0, np.round(max_ylim[1]), 3))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            plt.tight_layout()
            plt.savefig(join(fig_path_light, acronym, f'{subject}_{date}_{probe}_neuron{cluster}_left-stim'))
            plt.close(p)
        mod_r_neuron_ids = stim_neurons.loc[(stim_neurons['mod_r_light'] == True)
                                               & (stim_neurons['eid'] == eid)
                                               & (stim_neurons['probe'] == probe), 'cluster_id']
        for n, cluster in enumerate(mod_l_neuron_ids):
            if not isdir(join(fig_path_light, f'{acronym}')):
                mkdir(join(fig_path_light, f'{acronym}'))

            # Plot PSTH
            figure_style()
            p, ax = plt.subplots()
            peri_event_time_histogram(
                spikes[probe].times, spikes[probe].clusters,
                trials.loc[(trials['signed_contrast'] == 1) & (trials['laser_stimulation'] == 0), 'stimOn_times'],
                cluster, t_before=T_BEFORE, t_after=T_AFTER, bin_size=BIN_SIZE,
                error_bars='sem', ax=ax,
                pethline_kwargs={'color': 'black', 'lw': 2},
                errbar_kwargs={'color': 'black', 'alpha': 0.3})
            ylim_1 = ax.get_ylim()
            peri_event_time_histogram(
                spikes[probe].times, spikes[probe].clusters,
                trials.loc[(trials['signed_contrast'] == 1) & (trials['laser_stimulation'] == 1), 'stimOn_times'],
                cluster, t_before=T_BEFORE, t_after=T_AFTER, bin_size=BIN_SIZE,
                error_bars='sem', ax=ax,
                pethline_kwargs={'color': 'royalblue', 'lw': 2},
                errbar_kwargs={'color': 'royalblue', 'alpha': 0.3})
            max_ylim = np.max(np.vstack([ylim_1, ax.get_ylim()]), axis=0)
            ax.set(ylabel='spikes/s', xlabel='Time (s)', title='Right stimulus', ylim=max_ylim,
                   yticks=np.linspace(0, np.round(max_ylim[1]), 3))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            plt.tight_layout()
            plt.savefig(join(fig_path_light, acronym, f'{subject}_{date}_{probe}_neuron{cluster}_right-stim'))
            plt.close(p)

        # Stimulus modulated neurons
        stim_neuron_ids = stim_neurons.loc[((stim_neurons['mod_l_stim'] == True) | ((stim_neurons['mod_r_stim'] == True)))
                                             & (stim_neurons['eid'] == eid)
                                             & (stim_neurons['probe'] == probe), 'cluster_id']
        for n, cluster in enumerate(stim_neuron_ids):
            if not isdir(join(fig_path_stim, f'{acronym}')):
                mkdir(join(fig_path_stim, f'{acronym}'))

            # Plot PSTH
            colors = figure_style(return_colors=True)
            p, ax = plt.subplots()
            peri_event_time_histogram(
                spikes[probe].times, spikes[probe].clusters,
                trials.loc[(trials['signed_contrast'] == -1), 'stimOn_times'],
                cluster, t_before=T_BEFORE, t_after=T_AFTER, bin_size=BIN_SIZE,
                error_bars='sem', ax=ax,
                pethline_kwargs={'color': colors['left'], 'lw': 2},
                errbar_kwargs={'color': colors['left'], 'alpha': 0.3})
            ylim_1 = ax.get_ylim()
            peri_event_time_histogram(
                spikes[probe].times, spikes[probe].clusters,
                trials.loc[(trials['signed_contrast'] == 1), 'stimOn_times'],
                cluster, t_before=T_BEFORE, t_after=T_AFTER, bin_size=BIN_SIZE,
                error_bars='sem', ax=ax,
                pethline_kwargs={'color': colors['right'], 'lw': 2},
                errbar_kwargs={'color': colors['right'], 'alpha': 0.3})
            max_ylim = np.max(np.vstack([ylim_1, ax.get_ylim()]), axis=0)
            ax.set(ylabel='spikes/s', xlabel='Time (s)', ylim=max_ylim,
                   yticks=np.linspace(0, np.round(max_ylim[1]), 3))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            plt.tight_layout()
            plt.savefig(join(fig_path_stim, acronym, f'{subject}_{date}_{probe}_neuron{cluster}'))
            plt.close(p)




