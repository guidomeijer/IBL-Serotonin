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
from serotonin_functions import paths, remap, query_ephys_sessions, load_opto_pulse_times, remove_artifact_neurons
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

# Settings
PLOT = True
NEURON_QC = True
T_BEFORE = 0.005  # for plotting
T_AFTER = 0.01
BIN_SIZE = 0.001
_, fig_path, save_path = paths()
fig_path = join(fig_path, 'Ephys', 'SingleNeurons', 'OptoTagged')

# Query sessions
eids = one.search(task_protocol='_iblrig_tasks_ephysChoiceWorld', project='serotonin_inference')

for i, eid in enumerate(eids):

    # Get session details
    ses_details = one.get_details(eid)
    subject = ses_details['subject']
    date = ses_details['start_time'][:10]

    print(f'Starting {subject}, {date}')

    # Load in laser pulse times
    try:
        opto_pulses = load_opto_pulse_times(eid, one=one)
    except:
        print('Session does not have passive laser pulses')
        continue
    if len(opto_pulses) == 0:
        print('Did not find ANY laser pulses!')
        continue
    else:
        print(f'Found {len(opto_pulses)} passive laser pulses')

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

        # Plot light modulated units
        if PLOT:
            colors, dpi = figure_style()
            for n, cluster in enumerate(clusters_pass):
                try:
                    # Plot PSTH
                    p, ax = plt.subplots(1, 1, figsize=(2, 2), dpi=dpi)
                    peri_event_time_histogram(spikes[probe].times, spikes[probe].clusters, opto_pulses,
                                              cluster, t_before=T_BEFORE, t_after=T_AFTER, bin_size=BIN_SIZE,
                                              include_raster=True, error_bars='sem', smoothing=0, ax=ax,
                                              pethline_kwargs={'color': 'black', 'lw': 1},
                                              errbar_kwargs={'color': 'black', 'alpha': 0.3},
                                              raster_kwargs={'color': 'black', 'lw': 0.3},
                                              eventline_kwargs={'lw': 0})
                    ax.set(ylim=[ax.get_ylim()[0], ax.get_ylim()[1] + ax.get_ylim()[1] * 0.2])
                    ax.plot([0, 0.001], [ax.get_ylim()[1] - ax.get_ylim()[1] * 0.05,
                                     ax.get_ylim()[1] - ax.get_ylim()[1] * 0.05], lw=2, color='royalblue')
                    ax.set(ylabel='spikes/s', xlabel='Time (s)',
                           yticks=np.linspace(0, np.round(ax.get_ylim()[1]), 3))
                    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                    plt.tight_layout()
                    if not isdir(join(fig_path, 'NoHistology', f'{subject}_{date}')):
                        mkdir(join(fig_path, 'NoHistology', f'{subject}_{date}'))
                    plt.savefig(join(fig_path, 'NoHistology', f'{subject}_{date}',
                                     f'{subject}_{date}_{probe}_neuron{cluster}'), dpi=300)
                    plt.close(p)
                except:
                    plt.close(p)
                    continue

