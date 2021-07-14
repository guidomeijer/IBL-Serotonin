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
BASELINE = 3
_, fig_path, save_path = paths()
fig_path = join(fig_path, 'opto-behavior')

subjects = pd.read_csv(join('..', 'subjects.csv'))
subjects = subjects[~((subjects['date_range_blocks'] == 'none') & (subjects['date_range_probes'] == 'none')
                     & (subjects['date_range_half'] == 'none'))].reset_index(drop=True)

results_df = pd.DataFrame()
for i, nickname in enumerate(subjects['subject']):
    print(f'Processing {nickname}')
    # Query all sessions
    eids, details = one.search(subject=nickname,
                               task_protocol=['_iblrig_tasks_opto_biasedChoiceWorld',
                                              '_iblrig_tasks_biasedChoiceWorld'], details=True)
    all_dates = [i['date'] for i in details]

    # Find first stimulated session
    _, stim_details = one.search(subject=nickname,
                                 task_protocol=['_iblrig_tasks_opto_biasedChoiceWorld'],
                                 details=True)
    stim_dates = [i['date'] for i in stim_details]
    rel_ses = -(np.arange(len(all_dates)) - all_dates.index(np.min(stim_dates)))

    for j, eid in enumerate(eids):
        try:
            trials = load_trials(eid, one=one)
        except:
            continue
        bias = np.abs(trials[(trials['probabilityLeft'] == 0.8)
                             & (trials['signed_contrast'] == 0)].mean()
                      - trials[(trials['probabilityLeft'] == 0.2)
                               & (trials['signed_contrast'] == 0)].mean())['right_choice']
        results_df = results_df.append(pd.DataFrame(index=[results_df.shape[0] + 1], data={
            'bias': bias, 'stim': int('opto' in details[j]['task_protocol']),
            'session': rel_ses[j], 'date': details[j]['date'],
            'sert-cre': subjects.loc[i, 'sert-cre'], 'subject': nickname}))

    # Subtract baseline
    results_df.loc[results_df['subject'] == nickname, 'bias_baseline'] = (
        (results_df.loc[results_df['subject'] == nickname, 'bias']
        - (results_df.loc[(results_df['subject'] == nickname)
                         & (results_df['session'].between(-BASELINE, -1)), 'bias'].mean())))

# %% Plot
colors = figure_style(return_colors=True)
f, ax1 = plt.subplots(1, 1, figsize=(5, 5), dpi=150)
"""
sns.lineplot(x='session', y='bias', hue='sert-cre', style='subject', estimator=None,
             data=results_df, dashes=False, markers=['o']*int(results_df.shape[0]/2),
             legend=False, lw=2, ms=8, palette=[colors['wt'], colors['sert']], ax=ax1)
"""
"""
sns.lineplot(x='session', y='bias', hue='sert-cre', style='subject', dashes=False,
             estimator=None,
             data=results_df, legend=False, lw=2, ms=8, palette=[colors['wt'], colors['sert']], ax=ax1)
"""
sns.lineplot(x='session', y='bias', hue='sert-cre', ci=68,
             data=results_df, legend=False, lw=2, ms=8, palette=[colors['wt'], colors['sert']], ax=ax1)
#ax1.set(xlabel='', title='Probe trials', ylabel='Bia's')
ax1.set(xlim=[-10, 20])

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'stim-no-stim-bias'))

