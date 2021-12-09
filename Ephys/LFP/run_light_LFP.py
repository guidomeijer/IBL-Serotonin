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
from scipy.signal import periodogram
from serotonin_functions import (paths, remap, query_ephys_sessions, load_passive_opto_times,
                                 load_lfp, remap)
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

# Settings
EXCLUDE = ['ZFM-02180_2021-05-19', 'ZFM-02600_2021-08-26']
PLOT = True
T_BEFORE = 1  # for plotting
T_AFTER = 2
BIN_CENTERS = np.arange(-1, 2.01, 0.2)
BASELINE = [-1, 0]
BIN_SIZE = 0.25
THETA = [5, 15]
BETA = [15, 35]
GAMMA = [50, 120]
_, fig_path, save_path = paths()
fig_path = join(fig_path, 'Ephys', 'LFP')
save_path = join(save_path, 'LFP')

# Query sessions
eids, _, subjects = query_ephys_sessions(return_subjects=True, one=one)

all_lfp_df = pd.DataFrame()
for i, eid in enumerate(eids):

    # Get session details
    ses_details = one.get_details(eid)
    subject = ses_details['subject']
    date = ses_details['start_time'][:10]

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
    channels = bbone.load_channel_locations(eid, one=one)

    for p, probe in enumerate(channels.keys()):
        if 'acronym' not in channels[probe].keys():
            print(f'No brain regions found for {eid}')
            continue
        if not isfile(join(save_path, f'{subject}_{date}_{probe}_cleaned_lfp.npy')):
            print(f'Artifact removal not run for {subject}, {date}')
            continue

        # Load in lfp
        lfp = np.load(join(save_path, f'{subject}_{date}_{probe}_cleaned_lfp.npy'))
        time = np.load(join(save_path, f'{subject}_{date}_{probe}_timestamps.npy'))

        # Load in channels
        collections = one.list_collections(eid)
        if f'alf/{probe}/pykilosort' in collections:
            collection = f'alf/{probe}/pykilosort'
        else:
            collection = f'alf/{probe}'
        chan_ind = one.load_dataset(eid, dataset='channels.rawInd.npy', collection=collection)

        # Remap to Beryl atlas
        channels[probe]['acronym'] = remap(channels[probe]['atlas_id'])

        # Get LFP power per brain region
        for r, region in enumerate(np.unique(channels[probe]['acronym'])):
            lfp_df = pd.DataFrame()
            print(f'Processing {region}')
            region_chan = chan_ind[channels[probe]['acronym'] == region]
            for t, pulse_onset in enumerate(opto_train_times):
                theta, beta, gamma = np.zeros(BIN_CENTERS.shape), np.zeros(BIN_CENTERS.shape), np.zeros(BIN_CENTERS.shape)
                for b, bin_center in enumerate(BIN_CENTERS):
                    f, Pxx = periodogram(
                        lfp[np.ix_(np.isin(np.arange(lfp.shape[0]), region_chan),
                                   (time > pulse_onset + (bin_center - (BIN_SIZE / 2)))
                                   & (time < pulse_onset + (bin_center + (BIN_SIZE / 2))))],
                        fs=2500)
                    Pxx = np.mean(Pxx, axis=0)
                    theta[b] = Pxx[(f >= THETA[0]) & (f <= THETA[1])].mean()
                    beta[b] = Pxx[(f >= BETA[0]) & (f <= BETA[1])].mean()
                    gamma[b] = Pxx[(f >= GAMMA[0]) & (f <= GAMMA[1])].mean()

                # Baseline subtraction
                theta_p = ((theta - theta[(BIN_CENTERS >= BASELINE[0]) & (BIN_CENTERS <= BASELINE[1])].mean())
                           / theta[(BIN_CENTERS >= BASELINE[0]) & (BIN_CENTERS <= BASELINE[1])].mean()) * 100
                beta_p = ((beta - beta[(BIN_CENTERS >= BASELINE[0]) & (BIN_CENTERS <= BASELINE[1])].mean())
                          / beta[(BIN_CENTERS >= BASELINE[0]) & (BIN_CENTERS <= BASELINE[1])].mean()) * 100
                gamma_p = ((gamma - gamma[(BIN_CENTERS >= BASELINE[0]) & (BIN_CENTERS <= BASELINE[1])].mean())
                           / gamma[(BIN_CENTERS >= BASELINE[0]) & (BIN_CENTERS <= BASELINE[1])].mean()) * 100

                # Add to dataframe
                lfp_df = lfp_df.append(pd.DataFrame(data={
                    'theta': theta, 'theta_perc': theta_p, 'beta': beta, 'beta_perc': beta_p,
                    'gamma': gamma, 'gamma_perc': gamma_p, 'time': BIN_CENTERS}), ignore_index=True)

            # Add to overall dataframe
            #asd

            # Plot
            colors, dpi = figure_style()
            f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 3), dpi=dpi)
            sns.lineplot(x='time', y='theta_perc', data=lfp_df, ax=ax1, label='theta')
            sns.lineplot(x='time', y='beta_perc', data=lfp_df, ax=ax1, label='beta')
            ax1.legend(frameon=False)
            ax1.set(xlabel='Time (s)', ylabel='Change in LFP power (%)', title=f'{region}')

            sns.lineplot(x='time', y='gamma_perc', data=lfp_df, ax=ax2, label='gamma')
            ax2.legend(frameon=False)
            ax2.set(xlabel='Time (s)', ylabel='Change in LFP power (%)')

            ch_plot = np.random.choice(region_chan)
            plot_pulse_times = (opto_on_times - opto_on_times[0]) * 1000
            ax3.plot((time[((time > opto_on_times[0] - 0.01) & (time < opto_on_times[0] + 0.2))]\
                      - opto_on_times[0]) * 1000, lfp[ch_plot, ((time > opto_on_times[0] - 0.01)
                                                                   & (time < opto_on_times[0] + 0.2))],
                     zorder=2)
            y_lim = ax3.get_ylim()
            for pp in range(10):
                ax3.plot([plot_pulse_times[pp], plot_pulse_times[pp]], y_lim, ls='--', color='r',
                         lw=0.5, zorder=1)
            ax3.set(xlabel='Time (ms)', ylabel='uV', xlim=[-10, 200])

            plt.tight_layout()
            sns.despine(trim=True)
            plt.savefig(join(fig_path, f'{region}_{subject}_{date}_{probe}'))
            plt.close(f)
