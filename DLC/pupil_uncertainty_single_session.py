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
from oneibl.one import ONE
one = ONE()

# Settings
_, fig_path, save_path = paths()
fig_path = join(fig_path, 'opto-pupil')

eid = '15f742e1-1043-45c9-9504-f1e8a53c1744'

# Load in trials and video data
trials = load_trials(eid, one=one)
_, video_dlc, _, _, _, video_times = one.load(eid, dataset_types=['camera.dlc', 'camera.times'])

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