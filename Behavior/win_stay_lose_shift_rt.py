#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 10:53:33 2021
By: Guido Meijer
"""

from os.path import join
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from serotonin_functions import (paths, behavioral_criterion, load_trials, figure_style,
                                 query_opto_sessions, load_subjects)
from one.api import ONE
one = ONE()

# Settings
RT_CUTOFF = 0.5
subjects = load_subjects(behavior=True)
_, fig_path, save_path = paths()
fig_path = join(fig_path, 'Behavior', 'Models')

results_df = pd.DataFrame()
for i, nickname in enumerate(subjects['subject']):

    # Query sessions
    eids = query_opto_sessions(nickname, one=one)
    eids = behavioral_criterion(eids, one=one)

    # Get trials DataFrame
    trials = pd.DataFrame()
    for j, eid in enumerate(eids):
        try:
            if subjects.loc[i, 'sert-cre'] == 1:
                these_trials = load_trials(eid, laser_stimulation=True, patch_old_opto=True,
                                           invert_choice=True, one=one)
            else:
                these_trials = load_trials(eid, laser_stimulation=True, patch_old_opto=False,
                                           invert_choice=True, one=one)
        except:
            continue
        these_trials = these_trials.rename(columns={'feedbackType': 'trial_feedback_type'})
        these_trials['block_id'] = (these_trials['probabilityLeft'] == 0.2).astype(int)
        these_trials['stimulus_side'] = these_trials['stim_side'].copy()
        these_trials.loc[these_trials['signed_contrast'] == 0, 'stimulus_side'] = 0
        these_trials['contrast'] = these_trials['signed_contrast'].abs() * 100
        these_trials['previous_outcome'] = these_trials['trial_feedback_type'].shift(periods=1)
        these_trials['previous_choice'] = these_trials['choice'].shift(periods=1)
        trials = trials.append(these_trials, ignore_index=True)

    # Remove no-go trials
    trials = trials[trials['choice'] != 0]

    fast_rt_trials = trials[trials['reaction_times'] < RT_CUTOFF]
    win_stay_fast = (fast_rt_trials[((fast_rt_trials['previous_outcome'] == 1)
                                     & (fast_rt_trials['previous_choice'] == fast_rt_trials['choice']))].shape[0]
                     / fast_rt_trials.shape[0])
    lose_switch_fast = (fast_rt_trials[((fast_rt_trials['previous_outcome'] == -1)
                                        & (fast_rt_trials['previous_choice'] != fast_rt_trials['choice']))].shape[0]
                        / fast_rt_trials.shape[0])

    slow_rt_trials = trials[trials['reaction_times'] > RT_CUTOFF]
    win_stay_slow = (slow_rt_trials[((slow_rt_trials['previous_outcome'] == 1)
                                     & (slow_rt_trials['previous_choice'] == slow_rt_trials['choice']))].shape[0]
                     / slow_rt_trials.shape[0])
    lose_switch_slow = (slow_rt_trials[((slow_rt_trials['previous_outcome'] == -1)
                                        & (slow_rt_trials['previous_choice'] != slow_rt_trials['choice']))].shape[0]
                        / slow_rt_trials.shape[0])

    # Add to dataframe
    results_df = results_df.append(pd.DataFrame(data={
        'subject': nickname, 'sert-cre': subjects.loc[i, 'sert-cre'], 'win-stay': [win_stay_fast, win_stay_slow],
        'lose-switch': [lose_switch_fast, lose_switch_slow], 'rt': ['fast', 'slow']}), ignore_index=True)


# %% Plot

colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(2, 2), dpi=dpi)

sns.scatterplot(x='win-stay', y='lose-switch', data=results_df, hue='rt')


plt.tight_layout()
sns.despine(trim=True)


