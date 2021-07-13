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
from serotonin_functions import paths, criteria_opto_eids, load_trials, figure_style
from one.api import ONE
one = ONE()

# Settings
N_SESSIONS = 5
_, fig_path, save_path = paths()
fig_path = join(fig_path, 'opto-behavior')

subjects = pd.read_csv(join('..', 'subjects.csv'))
subjects = subjects[~((subjects['date_range_blocks'] == 'none') & (subjects['date_range_probes'] == 'none')
                     & (subjects['date_range_half'] == 'none'))].reset_index(drop=True)

results_df = pd.DataFrame()
for i, nickname in enumerate(subjects['subject']):

    # Stimulated sessions
    eids, details = one.search(subject=nickname,
                               task_protocol='_iblrig_tasks_opto_biasedChoiceWorld', details=True)
    stim_dates = [i['date'] for i in details]
    trials = pd.DataFrame()
    stim_ses = np.arange(len(eids), len(eids) - N_SESSIONS, -1)

    for j, ind in enumerate(stim_ses):
        trials = load_trials(eids[ind], one=one)
        bias = np.abs(trials[(trials['probabilityLeft'] == 0.8)
                             & (trials['signed_contrast'] == 0)].mean()
                      - trials[(trials['probabilityLeft'] == 0.2)
                               & (trials['signed_contrast'] == 0)].mean())['right_choice']
        results_df = results_df.append(pd.DataFrame(index=[results_df.shape[0] + 1], data={
            'bias': bias, 'stim': 1, 'session': j, 'date': stim_dates[j],
            'sert-cre': subjects.loc[i, 'sert-cre'], 'subject': nickname}))

    # Pre unstimulated sessions
    eids, details = one.search(subject=nickname, task_protocol='_iblrig_tasks_biasedChoiceWorld',
                               details=True)
    no_stim_dates = [datetime.strptime(i, '%Y-%m-%d') for i in [j['start_time'][:10] for j in details]]
    pre_dates = [i for i in no_stim_dates if i < np.min(stim_dates)]
    eids = one.search(subject=nickname, task_protocol='_iblrig_tasks_biasedChoiceWorld',
                      date_range=[str(pre_dates[4])[:10], str(pre_dates[0])[:10]])
    trials = pd.DataFrame()
    ses_count = 0
    for j, eid in enumerate(eids):
        these_trials = load_trials(eid, laser_stimulation=False, one=one)
        if these_trials is not None:
            these_trials['session'] = ses_count
            trials = trials.append(these_trials, ignore_index=True)
            ses_count = ses_count + 1
    if len(trials) == 0:
        continue
    bias = np.abs(trials[(trials['probabilityLeft'] == 0.8)
                               & (trials['signed_contrast'] == 0)].mean()
                        - trials[(trials['probabilityLeft'] == 0.2)
                                 & (trials['signed_contrast'] == 0)].mean())['right_choice']
    results_df = results_df.append(pd.DataFrame(index=[results_df.shape[0] + 1], data={
        'bias': bias, 'sessions': 'pre',
        'sert-cre': subjects.loc[i, 'sert-cre'], 'subject': nickname}))

# %% Plot
colors = figure_style(return_colors=True)
f, ax1 = plt.subplots(1, 1, figsize=(5, 5), dpi=150)

sns.lineplot(x='sessions', y='bias', hue='sert-cre', style='subject', estimator=None,
             data=results_df, dashes=False, markers=['o']*int(results_df.shape[0]/2),
             legend=False, lw=2, ms=8, palette=[colors['wt'], colors['sert']], ax=ax1)
#ax1.set(xlabel='', title='Probe trials', ylabel='Bias')

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'stim-no-stim-bias'))

