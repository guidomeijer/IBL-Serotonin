#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 11:42:01 2021

@author: guido
"""

import numpy as np
from os.path import join
import pandas as pd
from dlc_functions import get_dlc_XYs, get_pupil_diameter
import matplotlib.pyplot as plt
from scipy.stats import zscore
import seaborn as sns
from serotonin_functions import load_trials, paths, query_opto_sessions
from one.api import ONE
one = ONE()

# Settings
TIME_BINS = np.arange(-1, 3, 0.1)
BIN_SIZE = 0.1
BASELINE = 0.5
_, fig_path, _ = paths()
fig_path = join(fig_path, 'opto-pupil')

subjects = pd.read_csv(join('..', 'subjects.csv'))
subjects = subjects[subjects['subject'] == 'ZFM-02600'].reset_index(drop=True)
results_df = pd.DataFrame()
for i, nickname in enumerate(subjects['subject']):
    print(f'Processing {nickname}..')

    # Query sessions
    eids = query_opto_sessions(nickname, one=one)

    # Loop over sessions
    pupil_size = pd.DataFrame()
    for j, eid in enumerate(eids):
        print(f'Processing session {j+1} of {len(eids)}')

        # Load in trials and video data
        try:
            trials = load_trials(eid, laser_stimulation=True, one=one)
            if trials is None:
                continue
            if 'laser_stimulation' not in trials.columns.values:
                continue
        except:
            print('could not load trials')

        # Load in camera timestamps and DLC output
        try:
            video_times, XYs = get_dlc_XYs(eid, one=one)
        except:
            print('Could not load video and/or DLC data')
            continue

        # If the difference between timestamps and video frames is too large, skip
        if np.abs(video_times.shape[0]
                  - XYs['pupil_left_r'].shape[len(XYs['pupil_left_r'].shape) - 1]) > 100:
            print('Timestamp mismatch, skipping..')
            continue

        # Get pupil diameter
        diameter = get_pupil_diameter(XYs)

        # Assume frames were dropped at the end
        if video_times.shape[0] > diameter.shape[0]:
            video_times = video_times[:diameter.shape[0]]
        elif diameter.shape[0] > video_times.shape[0]:
            diameter = diameter[:video_times.shape[0]]

        # z-score pupil
        diameter_zscore = zscore(diameter)

        # Get trial triggered baseline subtracted pupil diameter
        for t, trial_start in enumerate(trials['stimOn_times']):
            this_diameter = np.array([np.nan] * TIME_BINS.shape[0])
            baseline_subtracted = np.array([np.nan] * TIME_BINS.shape[0])
            baseline = np.mean(diameter_zscore[(video_times > (trial_start - BASELINE))
                                               & (video_times < trial_start)])
            for b, time_bin in enumerate(TIME_BINS):
                this_diameter[b] = np.mean(diameter_zscore[
                    (video_times > (trial_start + time_bin) - (BIN_SIZE / 2))
                    & (video_times < (trial_start + time_bin) + (BIN_SIZE / 2))])
                baseline_subtracted[b] = np.mean(diameter_zscore[
                    (video_times > (trial_start + time_bin) - (BIN_SIZE / 2))
                    & (video_times < (trial_start + time_bin) + (BIN_SIZE / 2))]) - baseline
            pupil_size = pupil_size.append(pd.DataFrame(data={
                'diameter': this_diameter, 'baseline_subtracted': baseline_subtracted, 'eid': eid,
                'subject': nickname, 'trial': t, 'contrast': trials.loc[t, 'signed_contrast'],
                'sert': subjects.loc[i, 'sert-cre'], 'laser': trials.loc[t, 'laser_stimulation'],
                'laser_prob': trials.loc[t, 'laser_probability'],
                'time': TIME_BINS}))

    # Plot this animal
    if pupil_size.shape[0] > 0:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True, dpi=300)
        lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='laser',
                               data=pupil_size[((pupil_size['laser_prob'] == 0.75) & (pupil_size['laser'] == 1))
                                               | ((pupil_size['laser_prob'] == 0.25) & (pupil_size['laser'] == 0))],
                               palette='colorblind', ci=68, ax=ax1)
        ax1.set(title='%s, sert: %d' % (nickname, subjects.loc[i, 'sert-cre']),
                ylabel='z-scored pupil diameter', xlabel='Time relative to trial start(s)')

        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles=handles, labels=['No stim', 'Stim'], frameon=False)

        sns.lineplot(x='time', y='baseline_subtracted', hue='laser',
                     data=pupil_size[((pupil_size['laser_prob'] == 0.75) & (pupil_size['laser'] == 0))
                                               | ((pupil_size['laser_prob'] == 0.25) & (pupil_size['laser'] == 1))],
                     palette='colorblind', ci=68, legend=None, ax=ax2)
        ax2.set(xlabel='Time relative to trial start(s)',
                title='Catch trials')

        plt.tight_layout()
        sns.despine(trim=True)
        plt.savefig(join(fig_path, f'{nickname}_pupil_opto'))

        # Add to overall dataframe
        results_df = results_df.append(pupil_size[pupil_size['laser'] == 0].groupby(['time', 'laser']).mean())
        results_df = results_df.append(pupil_size[pupil_size['laser'] == 1].groupby(['time', 'laser']).mean())
        results_df['nickname'] = nickname
