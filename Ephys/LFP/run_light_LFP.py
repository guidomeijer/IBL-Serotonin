#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:17:26 2021
By: Guido Meijer
"""

import numpy as np
from os.path import join, isfile
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from serotonin_functions import figure_style
import brainbox.io.one as bbone
from scipy.signal import welch
from serotonin_functions import (paths, remap, query_ephys_sessions, load_passive_opto_times,
                                 load_lfp)
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

# Settings
DESTRIPED_LFP = False
# EXCLUDE = ['ZFM-02180_2021-05-19', 'ZFM-02600_2021-08-26']
PLOT = True
BASELINE = [-.5, 0]
STIM = [1.05, 1.55]
WINDOW_SIZE = 1024
fig_path, save_path = paths()
fig_path = join(fig_path, 'Ephys', 'LFP', 'RatioPostStim')

# Query sessions
rec = query_ephys_sessions(one=one)

lfp_df = pd.DataFrame()

for i in rec.index.values:

    # Get session details
    pid, eid, probe = rec.loc[i, 'pid'], rec.loc[i, 'eid'], rec.loc[i, 'probe']
    subject, date = rec.loc[i, 'subject'], rec.loc[i, 'date']
    print(f'Starting {subject}, {date}')

    # Load in laser pulse times
    try:
        opto_train_times, opto_on_times, opto_off_times = load_passive_opto_times(
                                                            eid, return_off_times=True, one=one)
    except:
        print('Session does not have passive laser pulses')
        continue
    if len(opto_train_times) == 0:
        print('Did not find ANY laser pulses!')
        continue
    else:
        print(f'Found {len(opto_train_times)} passive laser pulses')

    # Load in channels
    try:
        channels = bbone.load_channel_locations(eid, one=one)
    except:
        continue

    for p, probe in enumerate(channels.keys()):
        if 'acronym' not in channels[probe].keys():
            print(f'No brain regions found for {eid}')
            continue

        # Load in lfp
        if DESTRIPED_LFP and not isfile(join(save_path, f'{subject}_{date}_{probe}_cleaned_lfp.npy')):
            print(f'Artifact removal not run for {subject}, {date}')
            continue
        elif DESTRIPED_LFP:
            lfp = np.load(join(save_path, f'{subject}_{date}_{probe}_cleaned_lfp.npy'))
            time = np.load(join(save_path, f'{subject}_{date}_{probe}_timestamps.npy'))
        else:
            try:
                lfp, time = load_lfp(eid, probe, time_start=opto_on_times[0]-10,
                                     time_end=opto_on_times[-1]+10,
                                     relative_to='begin', one=one)
            except:
                continue

        # Load in channels
        collections = one.list_collections(eid)
        if f'alf/{probe}/pykilosort' in collections:
            collection = f'alf/{probe}/pykilosort'
        else:
            collection = f'alf/{probe}'
        chan_ind = one.load_dataset(eid, dataset='channels.rawInd.npy', collection=collection)

        # Remap to Beryl atlas
        channels[probe]['region'] = remap(channels[probe]['acronym'], combine=True)

        # Loop over laser pulse trains
        pulse_lfp_df = pd.DataFrame()
        for p, pulse_start in enumerate(opto_train_times):

            # Get baseline LFP power
            _, pBL = welch(lfp[:, (time >= pulse_start + BASELINE[0])
                               & (time <= pulse_start + BASELINE[1])],
                           fs=2500, window='hanning', nperseg=WINDOW_SIZE)
            freq, pStim = welch(lfp[:, (time >= pulse_start + STIM[0])
                                    & (time <= pulse_start + STIM[1])],
                                fs=2500, window='hanning', nperseg=WINDOW_SIZE)

            # Select frequencies of interest
            pBL = pBL[:, (freq >= 2) & (freq <= 200)]
            pStim = pStim[:, (freq >= 2) & (freq <= 200)]
            freq = freq[(freq >= 2) & (freq <= 200)]

            # Loop over brain regions
            for b, region in enumerate(np.unique(channels[probe]['region'])):
                if region == 'root':
                    continue

                # Get median LFP power over channels in region
                region_chan = chan_ind[channels[probe]['region'] == region]
                pulse_lfp_df = pulse_lfp_df.append(pd.DataFrame(data={
                    'baseline': 10 * np.log(np.median(pBL[region_chan, :], axis=0)),
                    'stim': 10 * np.log(np.median(pStim[region_chan, :], axis=0)),
                    'Hz': freq, 'region': region, 'pulse': p}), ignore_index=True)

        # Get ratio over stim/baseline (here it's inverted because both values are negative)
        pulse_lfp_df['ratio'] = pulse_lfp_df['baseline'] / pulse_lfp_df['stim']

        # Add to overall dataframe
        median_lfp_df = pulse_lfp_df.groupby(['Hz', 'region']).median().reset_index()
        median_lfp_df = median_lfp_df.drop(labels=['pulse'], axis=1)
        median_lfp_df['subject'] = subject
        median_lfp_df['date'] = date
        median_lfp_df['probe'] = probe
        lfp_df = lfp_df.append(median_lfp_df, ignore_index=True)

        # Plot this recording
        colors, dpi = figure_style()
        for r, region in enumerate(np.unique(pulse_lfp_df['region'])):
            f, ax1 = plt.subplots(1, 1, figsize=(2.5, 2), dpi=dpi)
            plt.plot([0, 100], [1, 1], ls='--', color='gray')
            sns.lineplot(data=pulse_lfp_df[pulse_lfp_df['region'] == region], x='Hz', y='ratio', palette='Set2', ax=ax1)
            ax1.set(ylabel='LFP ratio (stim/baseline)', xlabel='Frequency (Hz)', xlim=[0, 100])
            plt.tight_layout()
            sns.despine(trim=True)
            plt.savefig(join(fig_path, f'{region}_{subject}_{date}_{probe}'))
            plt.close(f)

# Save result
lfp_df.to_csv(join(save_path, 'LFP_ratio_opto_post_stim.csv'), index=False)
