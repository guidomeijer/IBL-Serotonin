#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 11:42:01 2021

@author: guido
"""

import numpy as np
from os.path import join
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from serotonin_functions import (paths, behavioral_criterion, load_trials, figure_style,
                                 load_subjects, fit_psychfunc, query_opto_sessions)
from one.api import ONE
one = ONE()

# Settings
BEHAVIOR_CRIT = True
_, fig_path, save_path = paths()
fig_path = join(fig_path, 'Behavior', 'ReactionTimes')

subjects = load_subjects()
subjects = subjects[subjects['subject'] != 'ZFM-02602']
subjects = subjects[subjects['subject'] != 'ZFM-02180']
subjects = subjects[subjects['subject'] != 'ZFM-01867']
subjects = subjects.reset_index()

results_df = pd.DataFrame()
for i, nickname in enumerate(subjects['subject']):
    print(f'Processing {nickname}')
    eids = query_opto_sessions(nickname, one=one)
    if len(eids) == 0:
        continue

    if BEHAVIOR_CRIT:
        eids = behavioral_criterion(eids, one=one)

    for j, eid in enumerate(eids):

        # Get trials DataFrame
        trials = pd.DataFrame()
        ses_count = 0
        for j, eid in enumerate(eids):
            try:
                if subjects.loc[i, 'sert-cre'] == 1:
                    these_trials = load_trials(eid, laser_stimulation=True, one=one)
                else:
                    these_trials = load_trials(eid, laser_stimulation=True, patch_old_opto=False, one=one)
                """
                these_trials = load_trials(eid, laser_stimulation=True, one=one)
                """
                these_trials['session'] = ses_count
                trials = trials.append(these_trials, ignore_index=True)
                ses_count = ses_count + 1
            except:
                pass
        if len(trials) == 0:
            continue

    perf_stim = (trials.loc[(trials['probe_trial'] == 0) & (trials['laser_stimulation'] == 1), 'correct'].sum()
                 / trials.loc[(trials['probe_trial'] == 0) & (trials['laser_stimulation'] == 1), 'correct'].shape[0])
    perf_no_stim = (trials.loc[(trials['probe_trial'] == 0) & (trials['laser_stimulation'] == 0), 'correct'].sum()
                    / trials.loc[(trials['probe_trial'] == 0) & (trials['laser_stimulation'] == 0), 'correct'].shape[0])
    perf_probe_stim = (trials.loc[(trials['probe_trial'] == 1) & (trials['laser_stimulation'] == 1), 'correct'].sum()
                       / trials.loc[(trials['probe_trial'] == 1) & (trials['laser_stimulation'] == 1), 'correct'].shape[0])
    perf_probe_no_stim = (trials.loc[(trials['probe_trial'] == 1) & (trials['laser_stimulation'] == 0), 'correct'].sum()
                          / trials.loc[(trials['probe_trial'] == 1) & (trials['laser_stimulation'] == 0), 'correct'].shape[0])
    perf_0_block_stim = (trials.loc[(trials['laser_probability'] == 0.75) & (trials['laser_stimulation'] == 1), 'correct'].sum()
                         / trials.loc[(trials['laser_probability'] == 0.75) & (trials['laser_stimulation'] == 1), 'correct'].shape[0])
    perf_0_block_no_stim = (trials.loc[(trials['laser_probability'] == 0.25) & (trials['laser_stimulation'] == 0), 'correct'].sum()
                            / trials.loc[(trials['laser_probability'] == 0.25) & (trials['laser_stimulation'] == 0), 'correct'].shape[0])
    perf_0_probe_stim = (trials.loc[(trials['laser_probability'] == 0.75) & (trials['laser_stimulation'] == 0), 'correct'].sum()
                         / trials.loc[(trials['laser_probability'] == 0.75) & (trials['laser_stimulation'] == 0), 'correct'].shape[0])
    perf_0_probe_no_stim = (trials.loc[(trials['laser_probability'] == 0.25) & (trials['laser_stimulation'] == 1), 'correct'].sum()
                            / trials.loc[(trials['laser_probability'] == 0.25) & (trials['laser_stimulation'] == 1), 'correct'].shape[0])



