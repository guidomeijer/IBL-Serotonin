#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 11:42:01 2021

@author: guido
"""

import numpy as np
from os.path import join
import pandas as pd
from dlc_functions import get_dlc_XYs, get_raw_and_smooth_pupil_dia
import matplotlib.pyplot as plt
import seaborn as sns
from serotonin_functions import paths, figure_style, load_passive_opto_times, load_subjects
from one.api import ONE
one = ONE()

# Settings
BEHAVIOR_CRIT = True
TIME_BINS = np.arange(-1, 3.2, 0.2)
BIN_SIZE = 0.2  # seconds
BASELINE = [1, 0]  # seconds
_, fig_path, _ = paths()
fig_path = join(fig_path, 'Pupil')

# Query and load data
eids = one.search(task_protocol='_iblrig_tasks_opto_ephysChoiceWorld',
                  dataset=['_ibl_leftCamera.dlc.pqt'])
subjects = load_subjects()

results_df, pupil_size = pd.DataFrame(), pd.DataFrame()
for i, eid in enumerate(eids):

    # Get session details
    ses_details = one.get_details(eid)
    nickname = ses_details['subject']
    date = ses_details['start_time'][:10]
    expression = subjects.loc[subjects['subject'] == nickname, 'expression'].values[0]
    print(f'Starting {nickname}, {date}')

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


    # Load in camera timestamps and DLC output
    try:
        video_times, XYs = get_dlc_XYs(one, eid)
    except:
        print('Could not load video and/or DLC data')
        continue

    # If the difference between timestamps and video frames is too large, skip
    if np.abs(video_times.shape[0] - XYs['pupil_left_r'].shape[0]) > 10000:
        print('Timestamp mismatch, skipping..')
        continue

    # Get pupil diameter
    print('Calculating smoothed pupil trace')
    raw_diameter, diameter = get_raw_and_smooth_pupil_dia(eid, 'left', one)

    # Assume frames were dropped at the end
    if video_times.shape[0] > diameter.shape[0]:
        video_times = video_times[:diameter.shape[0]]
    elif diameter.shape[0] > video_times.shape[0]:
        diameter = diameter[:video_times.shape[0]]

    # Calculate percentage change
    diameter_perc = ((diameter - np.percentile(diameter[~np.isnan(diameter)], 2))
                     / np.percentile(diameter[~np.isnan(diameter)], 2)) * 100

    # Get trial triggered baseline subtracted pupil diameter
    for t, trial_start in enumerate(opto_train_times):
        this_diameter = np.array([np.nan] * TIME_BINS.shape[0])
        baseline_subtracted = np.array([np.nan] * TIME_BINS.shape[0])
        baseline = np.nanmedian(diameter_perc[(video_times > (trial_start - BASELINE[0]))
                                              & (video_times < (trial_start - BASELINE[1]))])
        for b, time_bin in enumerate(TIME_BINS):
            this_diameter[b] = np.nanmedian(diameter_perc[
                (video_times > (trial_start + time_bin) - (BIN_SIZE / 2))
                & (video_times < (trial_start + time_bin) + (BIN_SIZE / 2))])
            baseline_subtracted[b] = np.nanmedian(diameter_perc[
                (video_times > (trial_start + time_bin) - (BIN_SIZE / 2))
                & (video_times < (trial_start + time_bin) + (BIN_SIZE / 2))]) - baseline
        pupil_size = pupil_size.append(pd.DataFrame(data={
            'diameter': this_diameter, 'baseline_subtracted': baseline_subtracted, 'eid': eid,
            'subject': nickname, 'trial': t, 'time': TIME_BINS, 'expression': expression,
            'date': date}))

    # Add to overal dataframe
    pupil_size = pupil_size.reset_index(drop=True)
    results_df = results_df.append(pd.DataFrame(data={
        'diameter': pupil_size.groupby('time').median()['diameter'],
        'baseline_subtracted': pupil_size.groupby('time').median()['baseline_subtracted'],
        'subject': nickname, 'expression': expression}))


# Plot
colors, dpi = figure_style()
for i, subject in enumerate(np.unique(pupil_size['subject'])):
    expression = subjects.loc[subjects['subject'] == subject, 'expression'].values[0]
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 2), dpi=dpi)

    lineplt = sns.lineplot(x='time', y='diameter', estimator=np.median,
                           data=pupil_size[pupil_size['subject'] == subject],
                           color=colors['stim'], ci=68, ax=ax1, legend=None)
    ax1.set(title='%s, expression: %d' % (subject, expression),
            ylabel='Pupil size (%)', xlabel='Time relative to trial start(s)',
            xticks=np.arange(-1, 3.1))

    lineplt = sns.lineplot(x='time', y='baseline_subtracted', estimator=np.median,
                           data=pupil_size[pupil_size['subject'] == subject],
                           color=colors['stim'], ci=68, ax=ax2, legend=None)
    ax2.set(title='%s, expression: %d' % (subject, expression),
            ylabel='Baseline subtracted\npupil size (%)', xlabel='Time relative to trial start(s)',
            xticks=np.arange(-1, 3.1))

    plt.tight_layout()
    sns.despine(trim=True)

    plt.savefig(join(fig_path, f'{subject}_pupil_opto_passive.png'))
    plt.savefig(join(fig_path, f'{subject}_pupil_opto_passive.pdf'))
