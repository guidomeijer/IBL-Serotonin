# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 17:34:25 2022

@author: Guido
"""


import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Rectangle
import pandas as pd
import seaborn as sns
from serotonin_functions import figure_style
from brainbox.io.one import SpikeSortingLoader
from serotonin_functions import paths, peri_multiple_events_time_histogram, load_passive_opto_times
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()


# Settings
T_BEFORE = 1  # for plotting
T_AFTER = 2
BIN_SIZE = 0.05
SMOOTHING = 0.025
PLOT_LATENCY = False
OVERWRITE = True
fig_path, save_path = paths()
fig_path = join(fig_path, 'Ephys', 'SingleNeurons', 'AwakeAnesthesia')

# Load in data
anesthesia_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons_anesthesia.csv'))
light_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))
merged_neurons = pd.merge(anesthesia_neurons, light_neurons, on=[
    'pid', 'neuron_id', 'subject', 'probe', 'date', 'eid', 'region'])
mod_neurons = merged_neurons[merged_neurons['modulated_x'] & merged_neurons['modulated_y']]

for i, pid in enumerate(np.unique(mod_neurons['pid'])):
    
    # Get eid
    eid = np.unique(mod_neurons.loc[mod_neurons['pid'] == pid, 'eid'])[0]
    probe = np.unique(mod_neurons.loc[mod_neurons['pid'] == pid, 'probe'])[0]
    subject = np.unique(mod_neurons.loc[mod_neurons['pid'] == pid, 'subject'])[0]
    date = np.unique(mod_neurons.loc[mod_neurons['pid'] == pid, 'date'])[0]
    print(f'Starting {subject}, {date}, {probe}')
    
    # Get opto times
    awake_opto_times, _ = load_passive_opto_times(eid, one=one, anesthesia=False)
    anesthesia_opto_times, _ = load_passive_opto_times(eid, one=one, anesthesia=True)
    all_times = np.concatenate((awake_opto_times, anesthesia_opto_times))
    anesthesia_ind = np.concatenate((np.zeros(awake_opto_times.shape), np.ones(awake_opto_times.shape)))
    
    # Load in spikes
    sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
    spikes, clusters, channels = sl.load_spike_sorting()
    clusters = sl.merge_clusters(spikes, clusters, channels)
    
    # Take slice of dataframe
    modulated = mod_neurons[mod_neurons['pid'] == pid]
    
    colors, dpi = figure_style()
    for n, ind in enumerate(modulated.index.values):
        region = modulated.loc[ind, 'region']
        subject = modulated.loc[ind, 'subject']
        date = modulated.loc[ind, 'date']
        neuron_id = modulated.loc[ind, 'neuron_id']

        # Plot PSTH
        p, ax = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
        ax.add_patch(Rectangle((0, 0), 1, 100, color='royalblue', alpha=0.25, lw=0))
        ax.add_patch(Rectangle((0, 0), 1, -100, color='royalblue', alpha=0.25, lw=0))
        peri_multiple_events_time_histogram(
            spikes.times, spikes.clusters, all_times, anesthesia_ind,
            neuron_id, t_before=T_BEFORE, t_after=T_AFTER, bin_size=BIN_SIZE, ax=ax,
            pethline_kwargs=[{'color': colors['awake'], 'lw': 1}, {'color': colors['anesthesia'], 'lw': 1}],
            errbar_kwargs=[{'color': colors['awake'], 'alpha': 0.3}, {'color': colors['anesthesia'], 'alpha': 0.3}],
            raster_kwargs=[{'color': colors['awake'], 'lw': 0.5}, {'color': colors['anesthesia'], 'lw': 0.5}],
            eventline_kwargs={'lw': 0}, include_raster=True)
        ax.set(ylabel='Firing rate (spikes/s)', xlabel='Time from trial start (s)',
               yticks=np.linspace(0, np.ceil(ax.get_ylim()[1]), 3), xticks=[-1, 0, 1, 2])
        if np.round(ax.get_ylim()[1]) % 2 == 0:
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        else:
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        sns.despine(trim=False)
        plt.tight_layout()
        
        plt.savefig(join(fig_path, f'{region}_{subject}_{date}_{probe}_neuron{neuron_id}.jpg'), dpi=600)
        plt.close(p)
    
    
    
    

