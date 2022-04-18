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
import seaborn as sns
from serotonin_functions import (load_trials, plot_psychometric, paths, behavioral_criterion,
                                 fit_psychfunc, figure_style, query_opto_sessions, get_bias,
                                 load_subjects)
from one.api import ONE
one = ONE()

# Settings
PLOT_SINGLE_ANIMALS = True
fig_path, _ = paths()
fig_path = join(fig_path, 'Behavior', 'Psychometrics')
subjects = load_subjects()

switch_df = pd.DataFrame()
for i, nickname in enumerate(subjects['subject']):

    # Query sessions
    eids = query_opto_sessions(nickname, one=one)

    # Exclude the first opto sessions
    #eids = eids[:-1]

    # Apply behavioral criterion
    eids = behavioral_criterion(eids, one=one)
    if len(eids) == 0:
        continue

    # Get trials DataFrame
    trials = pd.DataFrame()
    ses_count = 0
    for j, eid in enumerate(eids):
        try:
            """
            if subjects.loc[i, 'sert-cre'] == 1:
                these_trials = load_trials(eid, laser_stimulation=True, one=one)
            else:
                these_trials = load_trials(eid, laser_stimulation=True, patch_old_opto=False, one=one)
            """
            these_trials = load_trials(eid, laser_stimulation=True, patch_old_opto=False, one=one)

            these_trials['session'] = ses_count
            trials = pd.concat((trials, these_trials), ignore_index=True)
            ses_count = ses_count + 1
        except:
            pass
    if len(trials) == 0:
        continue

    # Make array of after block switch trials
    trials['block_switch'] = np.zeros(trials.shape[0])
    trial_blocks = (trials['probabilityLeft'] == 0.2).astype(int)
    block_trans = np.append(np.array(np.where(np.diff(trial_blocks) != 0)) + 1, [trial_blocks.shape[0]])

    for t, trans in enumerate(block_trans[:-1]):
        r_choice = trials.loc[(trials['signed_contrast'] == 0) & (trials.index.values < block_trans[t+1])
                              & (trials.index.values >= block_trans[t]), 'right_choice'].reset_index(drop=True)
        switch_df = pd.concat((switch_df, pd.DataFrame(data={
            'right_choice': r_choice, 'trial': r_choice.index.values, 'opto': trials.loc[trans, 'laser_stimulation'],
            'switch_to': trials.loc[trans, 'probabilityLeft'], 'subject': nickname,
            'sert-cre': subjects.loc[i, 'sert-cre']})), ignore_index=True)

    if PLOT_SINGLE_ANIMALS:
        colors, dpi = figure_style()
        f, ax1 = plt.subplots(1, 1, figsize=(3, 3), dpi=dpi)
        sns.lineplot(x='trial', y='right_choice', data=switch_df, style='opto', hue='switch_to')
        ax1.set(xlim=[0, 15])

