#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 11:10:21 2022
By: Guido Meijer
"""

import numpy as np
import pandas as pd
from os.path import join
from dlc_functions import get_dlc_XYs, smooth_interpolate_signal_sg
from serotonin_functions import (query_ephys_sessions, load_passive_opto_times, make_bins, paths,
                                 load_wheel_velocity)
from one.api import ONE
from ibllib.atlas import AllenAtlas
one = ONE()
ba = AllenAtlas()

# Settings
BINSIZE = 0.04
T_BEFORE = 1
T_AFTER = 2
STIM_DUR = 1
fig_path, save_path = paths()

# Query sessions
rec = query_ephys_sessions()

for i in rec.index.values:
    # Get session data
    pid, eid, probe = rec.loc[i, 'pid'], rec.loc[i, 'eid'], rec.loc[i, 'probe']
    subject, date = rec.loc[i, 'subject'], rec.loc[i, 'date']
    print(f'Processing {subject}, {date}')

    # Load in data
    opto_train_start, _ = load_passive_opto_times(eid, one=one)  # opto pulses
    video_times, XYs = get_dlc_XYs(one, eid)  # DLC
    mot_energy = one.load_datasets(eid, datasets=['bodyCamera.ROIMotionEnergy.npy',
                                                  'leftCamera.ROIMotionEnergy.npy',
                                                  'rightCamera.ROIMotionEnergy.npy'])[0]
    mot_body, mot_left, mot_right = mot_energy[0], mot_energy[1], mot_energy[2]
    cam_times = one.load_datasets(eid, datasets=['_ibl_bodyCamera.times.npy',
                                                 '_ibl_leftCamera.times.npy',
                                                 '_ibl_rightCamera.times.npy'])[0]
    times_body, times_left, times_right = cam_times[0], cam_times[1], cam_times[2]

    # Create boxcar predictor for opto stim
    time = np.linspace(0, T_BEFORE+T_AFTER+STIM_DUR, int((1/BINSIZE)*(T_BEFORE+T_AFTER+STIM_DUR)))
    opto_bc = np.zeros(time.shape)
    opto_bc[(time > T_BEFORE) & (time < T_BEFORE+STIM_DUR)] = 1

    # Smooth motion energy
    mot_body = smooth_interpolate_signal_sg(mot_body)
    mot_left = smooth_interpolate_signal_sg(mot_left)
    mot_right = smooth_interpolate_signal_sg(mot_right)

    # Get stim start and stop times
    opto_df = pd.DataFrame()
    opto_df['trial_start'] = opto_train_start - T_BEFORE
    opto_df['trial_end'] = opto_train_start + STIM_DUR + T_AFTER

    # Load in wheel velocity
    try:
        opto_df['wheel_velocity'] = load_wheel_velocity(eid, opto_df['trial_start'],
                                                        opto_df['trial_end'], BINSIZE, one=one)
    except:
        opto_df['wheel_velocity'] = np.nan

    # Create displacement of DLC markers
    print('Smoothing DLC traces')
    diff_video_times = (video_times[1:] + video_times[:-1]) / 2
    for i, dlc_key in enumerate(XYs.keys()):
        this_dist = np.linalg.norm(XYs[dlc_key][1:] - XYs[dlc_key][:-1], axis=1)
        this_dlc = smooth_interpolate_signal_sg(this_dist)
        opto_df[dlc_key] = make_bins(this_dlc, diff_video_times, opto_df['trial_start'],
                                     opto_df['trial_end'], BINSIZE)

    # Construct dataframe
    opto_df['opto_stim'] = [opto_bc] * opto_df.shape[0]
    opto_df['motion_energy_body'] = make_bins(mot_body, times_body, opto_df['trial_start'],
                                              opto_df['trial_end'], BINSIZE)
    opto_df['motion_energy_left'] = make_bins(mot_left, times_left, opto_df['trial_start'],
                                              opto_df['trial_end'], BINSIZE)
    opto_df['motion_energy_right'] = make_bins(mot_right, times_right, opto_df['trial_start'],
                                              opto_df['trial_end'], BINSIZE)

    # Save design matrix
    opto_df.to_pickle(join(save_path, 'GLM', f'dm_{subject}_{date}.pickle'))
    print('Saved output')

