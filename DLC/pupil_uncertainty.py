#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 11:42:01 2021

@author: guido
"""

import numpy as np
from os.path import join
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
import seaborn as sns
from serotonin_functions import (load_trials, butter_filter, paths, px_to_mm, pupil_features)
from one.api import ONE
one = ONE()

# Settings
TIME_BINS = np.arange(-1, 3, 0.1)
BIN_SIZE = 0.1
_, fig_path, save_path = paths()
fig_path = join(fig_path, 'opto-pupil')

subjects = pd.read_csv(join('..', 'subjects.csv'))
#subjects = subjects[subjects['subject'] == 'ZFM-01802'].reset_index(drop=True)
results_df = pd.DataFrame()
for i, nickname in enumerate(subjects['subject']):
    print(f'Processing {nickname}..')

    # Query sessions
    if subjects.loc[i, 'date_range_probes'] == 'all':
        eids = one.search(subject=nickname, task_protocol='_iblrig_tasks_opto_biasedChoiceWorld')
    elif subjects.loc[i, 'date_range_probes'] == 'none':
        continue
    else:
        eids = one.search(subject=nickname, task_protocol='_iblrig_tasks_opto_biasedChoiceWorld',
                          date_range=[subjects.loc[i, 'date_range_probes'][:10],
                                      subjects.loc[i, 'date_range_probes'][11:]])

    # Loop over sessions
    pupil_size = pd.DataFrame()
    for j, eid in enumerate(eids):
        print(f'Processing session {j+1} of {len(eids)}')

        # Load in trials and video data
        try:
            trials = load_trials(eid, laser_stimulation=True, one=one)
        except:
            print('could not load trials')
        if trials is None:
            continue
        if 'laser_stimulation' not in trials.columns.values:
            continue
        if 'laser_probability' not in trials.columns.values:
            trials['laser_probability'] = trials['laser_stimulation']
        video_dlc, video_times = one.load(eid, dataset_types=['camera.dlc', 'camera.times'])
        if video_dlc is None:
            continue

        # Assume frames were dropped at the end
        if video_times.shape[0] > video_dlc.shape[0]:
            video_times = video_times[:video_dlc.shape[0]]
        else:
            video_dlc = video_dlc[:video_times.shape[0]]

        # Get pupil size
        video_dlc = px_to_mm(video_dlc)
        x, y, diameter = pupil_features(video_dlc)

        # Remove blinks
        likelihood = np.mean(np.vstack((video_dlc['pupil_top_r_likelihood'],
                                        video_dlc['pupil_bottom_r_likelihood'],
                                        video_dlc['pupil_left_r_likelihood'],
                                        video_dlc['pupil_right_r_likelihood'])), axis=0)
        diameter = diameter[likelihood > 0.8]
        video_times = video_times[likelihood > 0.8]

        # Remove outliers
        video_times = video_times[diameter < 10]
        diameter = diameter[diameter < 10]

        # Low pass filter trace
        fs = 1 / ((video_times[-1] - video_times[0]) / video_times.shape[0])
        diameter_filt = butter_filter(diameter, lowpass_freq=0.5, order=1, fs=int(fs))
        diameter_zscore = zscore(diameter_filt)

        # Get trial triggered pupil diameter
        for t, trial_start in enumerate(trials['goCue_times']):
            this_diameter = np.array([np.nan] * TIME_BINS.shape[0])
            for b, time_bin in enumerate(TIME_BINS):
                this_diameter[b] = np.mean(diameter_zscore[
                    (video_times > (trial_start + time_bin) - (BIN_SIZE / 2))
                    & (video_times < (trial_start + time_bin) + (BIN_SIZE / 2))])
            pupil_size = pupil_size.append(pd.DataFrame(data={
                'diameter': this_diameter, 'eid': eid, 'subject': nickname, 'trial': t,
                'sert': subjects.loc[i, 'sert-cre'], 'laser': trials.loc[t, 'laser_stimulation'],
                'laser_prob': trials.loc[t, 'laser_probability'],
                'time': TIME_BINS}))

    # Plot this animal
    if pupil_size.shape[0] > 0:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True, dpi=300)
        lineplt = sns.lineplot(x='time', y='diameter', hue='laser', data=pupil_size,
                               palette='colorblind', ci=68, ax=ax1)
        ax1.set(title='%s, sert: %d, only probes sessions' % (nickname, subjects.loc[i, 'sert-cre']),
                ylabel='z-scored pupil diameter', xlabel='Time relative to trial start(s)')

        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles=handles, labels=['No stim', 'Stim'], frameon=False)

        plt.tight_layout()
        sns.despine(trim=True)
        plt.savefig(join(fig_path, f'{nickname}_pupil_opto_probes'))

        # Add to overall dataframe
        results_df = results_df.append(pupil_size[pupil_size['laser'] == 0].groupby(['time', 'laser']).mean())
        results_df = results_df.append(pupil_size[pupil_size['laser'] == 1].groupby(['time', 'laser']).mean())
        results_df['nickname'] = nickname

results_df.to_pickle(join(save_path, 'pupil_opto_probes.p'))