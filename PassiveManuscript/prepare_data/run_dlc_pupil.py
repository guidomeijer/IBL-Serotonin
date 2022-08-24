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
from serotonin_functions import paths, load_passive_opto_times, load_subjects
from one.api import ONE
one = ONE()

# Settings
TIME_BINS = np.arange(-0.5, 4.1, 0.1)
BIN_SIZE = 0.1  # seconds
BASELINE = [0.5, 0]  # seconds
fig_path, save_path = paths()

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
    if nickname not in subjects['subject'].values:
        continue
    expression = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
    print(f'Starting {nickname}, {date}')

    # Load in laser pulse times
    try:
        opto_train_times, _ = load_passive_opto_times(eid, one=one)
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
        pupil_size = pd.concat((pupil_size, pd.DataFrame(data={
            'diameter': this_diameter, 'baseline_subtracted': baseline_subtracted, 'eid': eid,
            'subject': nickname, 'trial': t, 'time': TIME_BINS, 'expression': expression,
            'date': date})))

    # Add to overal dataframe
    pupil_size = pupil_size.reset_index(drop=True)
    results_df = pd.concat((results_df, pd.DataFrame(data={
        'diameter': pupil_size[pupil_size['subject'] == nickname].groupby('time').median()['diameter'],
        'baseline_subtracted': pupil_size[pupil_size['subject'] == nickname].groupby('time').median()['baseline_subtracted'],
        'subject': nickname, 'expression': expression})))

# Save output
results_df.to_csv(join(save_path, 'pupil_passive.csv'))

