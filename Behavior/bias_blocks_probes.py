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
from serotonin_functions import paths, criteria_opto_eids, load_trials, figure_style
from one.api import ONE
one = ONE()

# Settings
_, fig_path, save_path = paths()
fig_path = join(fig_path, 'opto-behavior')

subjects = pd.read_csv(join('..', 'subjects.csv'))

results_df = pd.DataFrame()
accuracy_df = pd.DataFrame()
block_switches = pd.DataFrame()
for i, nickname in enumerate(subjects['subject']):

    # Query sessions
    if subjects.loc[i, 'date_range_blocks_good'] == 'all':
        eids = one.search(subject=nickname, task_protocol='_iblrig_tasks_opto_biasedChoiceWorld')
    elif subjects.loc[i, 'date_range_blocks_good'] == 'none':
        continue
    else:
        eids = one.search(subject=nickname, task_protocol='_iblrig_tasks_opto_biasedChoiceWorld',
                          date_range=[subjects.loc[i, 'date_range_blocks_good'][:10],
                                      subjects.loc[i, 'date_range_blocks_good'][11:]])
    #eids = criteria_opto_eids(eids, max_lapse=0.5, max_bias=0.5, min_trials=200, one=one)
    if len(eids) == 0:
        continue

    # Get trials DataFrame
    trials = pd.DataFrame()
    ses_count = 0
    for j, eid in enumerate(eids):
        these_trials = load_trials(eid, laser_stimulation=True, one=one)
        if these_trials is not None:
            these_trials['session'] = ses_count
            trials = trials.append(these_trials, ignore_index=True)
            ses_count = ses_count + 1
    if len(trials) == 0:
        continue
    if 'laser_probability' not in trials.columns:
        trials['laser_probability'] = trials['laser_stimulation'].copy()

    bias_no_stim = np.abs(trials[(trials['probabilityLeft'] == 0.8)
                               & (trials['signed_contrast'] == 0)
                               & (trials['laser_stimulation'] == 0)].mean()
                        - trials[(trials['probabilityLeft'] == 0.2)
                                 & (trials['signed_contrast'] == 0)
                                 & (trials['laser_stimulation'] == 0)].mean())['right_choice']
    bias_stim = np.abs(trials[(trials['probabilityLeft'] == 0.8)
                               & (trials['signed_contrast'] == 0)
                               & (trials['laser_stimulation'] == 1)].mean()
                        - trials[(trials['probabilityLeft'] == 0.2)
                                 & (trials['signed_contrast'] == 0)
                                 & (trials['laser_stimulation'] == 1)].mean())['right_choice']


    results_df = results_df.append(pd.DataFrame(data={'bias': [bias_no_stim, bias_stim],
                                                      'opto_stim': ['no stim', 'stim'],
                                                      'sert-cre': subjects.loc[i, 'sert-cre'],
                                                      'subject': nickname}))


# %% Plot
colors = figure_style(return_colors=True)
f, ax1 = plt.subplots(1, 1, figsize=(5, 5), dpi=150)

sns.lineplot(x='opto_stim', y='bias', hue='sert-cre', style='subject', estimator=None,
             data=results_df, dashes=False, markers=['o']*int(results_df.shape[0]/2),
             legend=False, lw=2, ms=8, palette=[colors['wt'], colors['sert']], ax=ax1)
ax1.set(xlabel='', title='Probe trials', ylabel='Bias')

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'bias-probe'))
