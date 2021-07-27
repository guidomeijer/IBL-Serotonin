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
BASELINE = 10
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
colors, dpi = figure_style()
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7, 3.5), dpi=dpi)
"""
sns.lineplot(x='session', y='bias', hue='sert-cre', style='subject', estimator=None,
             data=results_df, dashes=False, markers=['o']*int(results_df.shape[0]/2),
             legend=False, lw=2, ms=8, palette=[colors['wt'], colors['sert']], ax=ax1)

sns.lineplot(x='session', y='bias', hue='sert-cre', style='subject', dashes=False,
             estimator=None,
             data=results_df, legend=False, lw=2, ms=8, palette=[colors['wt'], colors['sert']], ax=ax1)
"""
sns.lineplot(x='session', y='bias', hue='sert-cre', ci=68,
             data=results_df[results_df['sert-cre'] == 1], legend=False, lw=2, ms=8,
             palette=[colors['sert']], ax=ax1)

for i, nickname in enumerate(results_df.loc[results_df['sert-cre'] == 1, 'subject'].unique()):
    ax2.plot([0, 1], [results_df.loc[(results_df['subject'] == nickname)
                                     & (results_df['session'].between(-3, -1)), 'bias'].mean(),
                      results_df.loc[(results_df['subject'] == nickname)
                                     & (results_df['session'].between(0, 2)), 'bias'].mean()],
             color = 'k')
ax2.set(ylim=[0, 0.5])

for i, nickname in enumerate(results_df.loc[results_df['sert-cre'] == 1, 'subject'].unique()):
    ax3.plot([0, 1], [results_df.loc[(results_df['subject'] == nickname)
                                     & (results_df['session'].between(-3, -1)), 'bias'].mean(),
                      results_df.loc[(results_df['subject'] == nickname)
                                     & (results_df['session'].between(3, 5)), 'bias'].mean()],
             color = 'k')
ax3.set(ylim=[0, 0.5])

#ax1.set(xlabel='', title='Probe trials', ylabel='Bia's')
ax1.set(xlim=[-8, 20])

#sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'stim-no-stim-bias'))

