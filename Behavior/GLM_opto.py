#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 10:53:33 2021
By: Guido Meijer
"""

from os.path import join
import pandas as pd
from serotonin_functions import (paths, behavioral_criterion, load_trials, figure_style,
                                 query_opto_sessions, load_subjects, fit_glm)
from one.api import ONE
one = ONE()

# Settings
subjects = load_subjects(behavior=True)
_, fig_path, save_path = paths()
fig_path = join(fig_path, 'Behavior', 'Models')

results_df = pd.DataFrame()
results_laser_df = pd.DataFrame()
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

    # Fit GLM
    params_all = fit_glm(trials, opto_stim=True)
    params_laser = fit_glm(trials[trials['laser_stimulation'] == 1])
    params_no_laser = fit_glm(trials[trials['laser_stimulation'] == 0])

    # Add to dataframe
    results_df = results_df.append(params_all, ignore_index=True)
    results_df.loc[results_df.shape[0]-1, 'subject'] = nickname
    results_df.loc[results_df.shape[0]-1, 'sert-cre'] = subjects.loc[i, 'sert-cre']
    results_laser_df = results_laser_df.append(params_laser, ignore_index=True)
    results_laser_df.loc[results_laser_df.shape[0]-1, 'subject'] = nickname
    results_laser_df.loc[results_laser_df.shape[0]-1, 'sert-cre'] = subjects.loc[i, 'sert-cre']
    results_laser_df.loc[results_laser_df.shape[0]-1, 'opto-stim'] = 1
    results_laser_df = results_laser_df.append(params_no_laser, ignore_index=True)
    results_laser_df.loc[results_laser_df.shape[0]-1, 'subject'] = nickname
    results_laser_df.loc[results_laser_df.shape[0]-1, 'sert-cre'] = subjects.loc[i, 'sert-cre']
    results_laser_df.loc[results_laser_df.shape[0]-1, 'opto-stim'] = 0


