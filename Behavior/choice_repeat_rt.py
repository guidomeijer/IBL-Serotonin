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
                                 query_opto_sessions, load_subjects, fit_glm)
from one.api import ONE
one = ONE()

# Settings
RT_CUTOFF = 0.5
subjects = load_subjects(behavior=True)
_, fig_path, save_path = paths()
fig_path = join(fig_path, 'Behavior', 'Models')

results_df = pd.DataFrame()
all_trials = pd.DataFrame()
for i, nickname in enumerate(subjects['subject']):

    # Query sessions
    eids = query_opto_sessions(nickname, one=one)
    eids = behavioral_criterion(eids, one=one)

    # Get trials DataFrame
    trials = pd.DataFrame()
    for j, eid in enumerate(eids):
        try:
            if subjects.loc[i, 'sert-cre'] == 1:
                these_trials = load_trials(eid, laser_stimulation=True, patch_old_opto=True, one=one)
            else:
                these_trials = load_trials(eid, laser_stimulation=True, patch_old_opto=False, one=one)
        except:
            continue
        these_trials['previous_choice'] = these_trials['choice'].shift(periods=1)
        trials = trials.append(these_trials, ignore_index=True)

    # Remove no-go trials
    trials[trials['choice'] == 0] = np.nan
    trials[trials['previous_choice'] == 0] = np.nan
    trials = trials.dropna(subset=['choice', 'previous_choice'])

    # Get probability of repeat split by reaction time
    p_repeat_short = (np.sum(trials.loc[trials['reaction_times'] < RT_CUTOFF, 'choice']
                             == trials.loc[trials['reaction_times'] < RT_CUTOFF, 'previous_choice'])
                      / trials[trials['reaction_times'] < RT_CUTOFF].shape[0])
    p_repeat_long = (np.sum(trials.loc[trials['reaction_times'] > RT_CUTOFF, 'choice']
                            == trials.loc[trials['reaction_times'] > RT_CUTOFF, 'previous_choice'])
                     / trials[trials['reaction_times'] > RT_CUTOFF].shape[0])


    # Add to dataframe
    results_df = results_df.append(pd.DataFrame(data={
        'subject': nickname, 'sert-cre': subjects.loc[i, 'sert-cre'],
        'p_repeat': [p_repeat_short, p_repeat_long], 'rt': ['short', 'long']}), ignore_index=True)


# %% Plot

colors, dpi = figure_style()
colors = [colors['wt'], colors['sert']]
f, ax1 = plt.subplots(1, 1, figsize=(2, 2), dpi=dpi)
sns.lineplot(x='rt', y='p_repeat', data=results_df, hue='sert-cre', estimator=None, units='subject',
             palette=colors, legend=None, dashes=False, markers=['o']*int(results_df.shape[0]/2), ax=ax1)

ax1.set(xlabel='', xticks=[0, 1], xticklabels=['Short RT', 'Long RT'], ylabel='P(choice repeat)',
        ylim=[0.6, 0.8])

plt.tight_layout()
sns.despine(trim=True)


