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
from brainbox.metrics.single_units import spike_sorting_metrics
from serotonin_functions import figure_style
import brainbox.io.one as bbone
from brainbox.plot import peri_event_time_histogram
from serotonin_functions import paths, remap, query_ephys_sessions, load_opto_times
from one.api import ONE
one = ONE()

# Settings
PLOT = True
OVERWRITE = True
NEURON_QC = False
T_BEFORE = 1  # for plotting
T_AFTER = 2
PRE_TIME = [1, 0]  # for significance testing
POST_TIME = [0, 1]
BIN_SIZE = 0.05
PERMUTATIONS = 500
_, fig_path, save_path = paths()
fig_path = join(fig_path, 'Ephys', 'SingleNeurons', 'LightModNeurons')

# Query sessions
eids, _ = query_ephys_sessions(one=one)

if OVERWRITE:
    light_neurons = pd.DataFrame()
else:
    light_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))
    eids = eids[~np.isin(eids, light_neurons['eid'])]
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
    spikes, clusters, channels = bbone.load_spike_sorting_with_channel(eid, aligned=True, one=one)

    for p, probe in enumerate(spikes.keys()):
        if 'acronym' not in clusters[probe].keys():
            print(f'No brain regions found for {eid}')
            continue

        # Select spikes of passive period
        start_passive = opto_train_times[0] - 360
        spikes[probe].clusters = spikes[probe].clusters[spikes[probe].times > start_passive]
        spikes[probe].times = spikes[probe].times[spikes[probe].times > start_passive]

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

        # Determine significant neurons
        print('Calculating significant neurons..')
        roc_auc, cluster_ids = roc_single_event(spikes[probe].times, spikes[probe].clusters,
                                                opto_train_times, pre_time=PRE_TIME, post_time=POST_TIME)
        roc_auc = 2 * (roc_auc - 0.5)

        roc_auc_permut = np.zeros([PERMUTATIONS, len(np.unique(spikes[probe].clusters))])
        for k in range(PERMUTATIONS):
            this_roc_auc_permut = roc_single_event(
                spikes[probe].times, spikes[probe].clusters,
                np.random.uniform(low=start_passive, high=opto_train_times[-1],
                                  size=opto_train_times.shape[0]),
                pre_time=PRE_TIME, post_time=POST_TIME)[0]
            roc_auc_permut[k, :] = 2 * (this_roc_auc_permut - 0.5)

        modulated = ((roc_auc > np.percentile(roc_auc_permut, 97.5, axis=0))
                       | (roc_auc < np.percentile(roc_auc_permut, 2.5, axis=0)))
        enhanced = roc_auc > np.percentile(roc_auc_permut, 97.5, axis=0)
        suppressed = roc_auc < np.percentile(roc_auc_permut, 2.5, axis=0)

        cluster_regions = remap(clusters[probe].atlas_id[cluster_ids])
        light_neurons = light_neurons.append(pd.DataFrame(data={
            'subject': subject, 'date': date, 'eid': eid, 'probe': probe,
            'region': cluster_regions, 'cluster_id': cluster_ids,
            'roc_auc': roc_auc, 'modulated': modulated, 'enhanced': enhanced, 'suppressed': suppressed}))

        # Plot light modulated units
        if PLOT:
            for n, cluster in enumerate(cluster_ids[modulated]):
                if not isdir(join(fig_path, f'{cluster_regions[cluster_ids == cluster][0]}')):
                    mkdir(join(fig_path, f'{cluster_regions[cluster_ids == cluster][0]}'))

                # Plot PSTH
                colors, dpi = figure_style()
                p, ax = plt.subplots(1, 1, figsize=(2, 2), dpi=dpi)
                peri_event_time_histogram(spikes[probe].times, spikes[probe].clusters, opto_train_times,
                                          cluster, t_before=T_BEFORE, t_after=T_AFTER, bin_size=BIN_SIZE,
                                          include_raster=True, error_bars='sem', ax=ax,
                                          pethline_kwargs={'color': 'black', 'lw': 1},
                                          errbar_kwargs={'color': 'black', 'alpha': 0.3},
                                          raster_kwargs={'color': 'black', 'lw': 0.3},
                                          eventline_kwargs={'lw': 0})
                ax.set(ylim=[ax.get_ylim()[0], ax.get_ylim()[1] + ax.get_ylim()[1] * 0.2])
                ax.plot([0, 1], [ax.get_ylim()[1] - ax.get_ylim()[1] * 0.05,
                                 ax.get_ylim()[1] - ax.get_ylim()[1] * 0.05], lw=2, color='royalblue')
                ax.set(ylabel='spikes/s', xlabel='Time (s)',
                       title=f'Modulation index: {roc_auc[cluster_ids == cluster][0]:.2f}',
                       yticks=np.linspace(0, np.round(ax.get_ylim()[1]), 3))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                plt.tight_layout()
                plt.savefig(join(fig_path, cluster_regions[cluster_ids == cluster][0],
                                 f'{subject}_{date}_{probe}_neuron{cluster}'))
                plt.close(p)


    light_neurons.to_csv(join(save_path, 'light_modulated_neurons.csv'))
light_neurons.to_csv(join(save_path, 'light_modulated_neurons.csv'))
