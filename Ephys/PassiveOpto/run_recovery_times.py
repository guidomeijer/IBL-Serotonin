#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 12:49:43 2022
By: Guido Meijer
"""


import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.optimize import curve_fit
from serotonin_functions import figure_style
from brainbox.io.one import SpikeSortingLoader
from brainbox.singlecell import calculate_peths
from matplotlib.patches import Rectangle
from serotonin_functions import (paths, load_passive_opto_times, combine_regions, load_subjects,
                                 high_level_regions, figure_style)
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

# Settings
PLOT = False
T_BEFORE = 1  # for plotting
T_AFTER = 4
BIN_SIZE = 0.1
SMOOTHING = 0.1
BASELINE = [-1, 0]
MIN_FR = 0.1
fig_path, save_path = paths()
fig_path = join(fig_path, 'Ephys', 'SingleNeurons', 'RecoveryTime')


def monoExp(x, m, t, b):
    return m * np.exp(-t * x) + b


# Load in light modulated neurons
light_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))
light_neurons['full_region'] = combine_regions(light_neurons['region'], split_thalamus=False)
light_neurons = light_neurons[light_neurons['full_region'] != 'root']

# Only select neurons from sert-cre mice
subjects = load_subjects()
light_neurons = light_neurons[light_neurons['subject'].isin(
    subjects.loc[subjects['sert-cre'] == 1, 'subject'].values)]

# %% Loop over sessions
recovery_df = pd.DataFrame()
for i, pid in enumerate(np.unique(light_neurons['pid'])):

    # Get session details
    eid = np.unique(light_neurons.loc[light_neurons['pid'] == pid, 'eid'])[0]
    probe = np.unique(light_neurons.loc[light_neurons['pid'] == pid, 'probe'])[0]
    subject = np.unique(light_neurons.loc[light_neurons['pid'] == pid, 'subject'])[0]
    date = np.unique(light_neurons.loc[light_neurons['pid'] == pid, 'date'])[0]
    print(f'Starting {subject}, {date}')

    # Load in laser pulse times
    opto_train_times, _ = load_passive_opto_times(eid, one=one)
    if len(opto_train_times) == 0:
        continue

    # Load in spikes
    try:
        sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
        spikes, clusters, channels = sl.load_spike_sorting()
        clusters = sl.merge_clusters(spikes, clusters, channels)
    except Exception as err:
        print(err)
        continue

    # Take slice of dataframe
    these_neurons = light_neurons[(light_neurons['modulated'] == 1)
                                  & (light_neurons['eid'] == eid)
                                  & (light_neurons['probe'] == probe)]

    # Get peri-event time histogram
    peths, _ = calculate_peths(spikes.times, spikes.clusters,
                               these_neurons['neuron_id'].values,
                               opto_train_times, T_BEFORE, T_AFTER, BIN_SIZE, SMOOTHING)
    tscale = peths['tscale']

    # Loop over neurons
    for n, index in enumerate(these_neurons.index.values):
        if np.mean(peths['means'][n, :]) > MIN_FR:

            # Calculate ratio change in firing rate
            peth_ratio = ((peths['means'][n, :]
                           - np.mean(peths['means'][n, ((tscale > BASELINE[0]) & (tscale < BASELINE[1]))]))
                          / (peths['means'][n, :]
                             + np.mean(peths['means'][n, ((tscale > BASELINE[0]) & (tscale < BASELINE[1]))])))
            peth_ratio = np.abs(peth_ratio)
            peak_ind = np.argmax(peth_ratio[tscale > 0]) + np.sum(tscale <= 0)
            through_ind = np.argmin(peth_ratio[peak_ind:]) + peak_ind

            try:
                params, cv = curve_fit(monoExp, tscale[peak_ind:through_ind], peth_ratio[peak_ind:through_ind],
                                       p0=(1, 1, 0), bounds=(0, 5))
            except:
                print('Fit failed')
                continue
            tauSec = 1 / params[1]

            # Add to dataframe
            recovery_df = pd.concat((recovery_df, pd.DataFrame(index=[recovery_df.shape[0]+1], data={
                'tau': 1 / params[1], 'peth_ratio': [peth_ratio], 'time': [tscale],
                'region': these_neurons.loc[index, 'full_region'], 'modulation': these_neurons.loc[index, 'mod_index_late'],
                'neuron_id': these_neurons.loc[index, 'neuron_id'], 'subject': these_neurons.loc[index, 'subject'],
                'eid': these_neurons.loc[index, 'eid'], 'acronym': these_neurons.loc[index, 'region']})))

            if PLOT:
                m, t, b = params
                colors, dpi = figure_style()
                f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
                ax1.add_patch(Rectangle((0, -1), 1, 2, color='royalblue', alpha=0.25, lw=0))
                ax1.plot(tscale, peth_ratio)
                ax1.plot(tscale[peak_ind:through_ind], monoExp(tscale[peak_ind:through_ind], m, t, b))
                ax1.set(ylim=[np.min(peth_ratio) - 0.04, np.max(peth_ratio) + 0.04],
                        title=f'tau: {tauSec:.2f} s', ylabel='Absolute ratio FR change',
                        xlabel='Time (s)', xticks=[-1, 0, 1, 2, 3, 4])
                plt.tight_layout()
                sns.despine(trim=True)
                plt.savefig(join(fig_path, f'{subject}_{date}_{probe}_{these_neurons.loc[index, "neuron_id"]}.pdf'))
                plt.close(f)

recovery_df.to_pickle(join(save_path, 'recovery_tau.pickle'))