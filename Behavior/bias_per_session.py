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
                                 behavioral_criterion, get_bias, load_subjects)
from one.api import ONE
one = ONE()

# Settings
BASELINE = 5
_, fig_path, save_path = paths()
fig_path = join(fig_path, 'Behavior')
subjects = load_subjects()
subjects = subjects[subjects['subject'] != 'ZFM-02602']
subjects = subjects[subjects['subject'] != 'ZFM-02180']
subjects = subjects[subjects['subject'] != 'ZFM-01867']
subjects = subjects.reset_index()

results_df = pd.DataFrame()
for i, nickname in enumerate(subjects['subject']):
    print(f'Processing {nickname}')
    # Query all sessions
    eids, details = one.search(subject=nickname,
                               task_protocol=['_iblrig_tasks_opto_biasedChoiceWorld',
                                              '_iblrig_tasks_biasedChoiceWorld'], details=True)
    all_eids = []
    all_dates = []
    for k, eid in enumerate(eids):
        full_details = one.get_details(eid, full=True)
        if 'iblrig' in full_details['location']:
            all_eids.append(eid)
            all_dates.append(datetime.strptime(full_details['start_time'][:10], '%Y-%m-%d').date())

    # Find first stimulated session
    _, stim_details = one.search(subject=nickname,
                                 task_protocol=['_iblrig_tasks_opto_biasedChoiceWorld'],
                                 details=True)
    if stim_details is None:
        continue
    stim_dates = [i['date'] for i in stim_details]
    rel_ses = -(np.arange(len(all_dates)) - all_dates.index(np.min(stim_dates)))

    # Apply behavioral criterion
    # behavioral_criterion

    for j, eid in enumerate(all_eids):
        try:
            trials = load_trials(eid, one=one)
        except:
            continue
        if len(trials) < 300:
            continue
        if len(np.unique(trials['probabilityLeft'])) > 3:
            continue
        """
        bias = np.abs(trials[(trials['probabilityLeft'] == 0.8)
                             & (trials['signed_contrast'] == 0)].mean()
                      - trials[(trials['probabilityLeft'] == 0.2)
                               & (trials['signed_contrast'] == 0)].mean())['right_choice']
        """
        bias = get_bias(trials)
        results_df = results_df.append(pd.DataFrame(index=[results_df.shape[0] + 1], data={
            'bias': bias, 'stim': int('opto' in details[j]['task_protocol']),
            'session': rel_ses[j], 'date': details[j]['date'],
            'sert-cre': subjects.loc[i, 'sert-cre'],
            'expression': subjects.loc[i, 'expression'],'subject': nickname}))

    # Subtract baseline
    results_df.loc[results_df['subject'] == nickname, 'bias_baseline'] = (
        (results_df.loc[results_df['subject'] == nickname, 'bias']
        - (results_df.loc[(results_df['subject'] == nickname)
                         & (results_df['session'].between(-BASELINE, -1)), 'bias'].mean())))

# %% Plot
colors, dpi = figure_style()
f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(9, 2), dpi=dpi)
"""
sns.lineplot(x='session', y='bias', hue='sert-cre', style='subject', estimator=None,
             data=results_df, dashes=False, markers=['o']*int(results_df.shape[0]/2),
             legend=False, lw=2, ms=8, palette=[colors['wt'], colors['sert']], ax=ax1)
"""
sns.lineplot(x='session', y='bias', hue='sert-cre', style='subject', dashes=False,
             estimator=None, data=results_df, legend=False, lw=2, ms=8,
             palette=[colors['wt'], colors['sert']], ax=ax1)
"""
sns.lineplot(x='session', y='bias', hue='expression', ci=68,
             data=results_df[results_df['expression'] == 1], legend=False,
             palette=[colors['sert']], ax=ax1)
"""

ax1.set(ylabel='Bias', xlabel='Days relative to first opto day', ylim=[0, 0.61], xlim=[-BASELINE, 10])


sns.lineplot(x='session', y='bias', hue='expression', ci=68,
             data=results_df[results_df['expression'] == 0], legend=False,
             palette=[colors['wt']], ax=ax2)
ax2.set(ylabel='Bias', xlabel='Days relative to first opto day', ylim=[0, 0.61], xlim=[-BASELINE, 10])

for i, nickname in enumerate(results_df.loc[results_df['expression'] == 1, 'subject'].unique()):
    ax3.plot([0, 1], [results_df.loc[(results_df['subject'] == nickname)
                                     & (results_df['session'].between(-3, -1)), 'bias'].mean(),
                      results_df.loc[(results_df['subject'] == nickname)
                                     & (results_df['session'].between(0, 2)), 'bias'].mean()],
             color='gray', marker='o')
ax3.set(ylabel='Bias', ylim=[0, 0.61], xticks=[0, 1], xticklabels=['-3 to -1', '0 to 2'],
        xlabel='Days')

for i, nickname in enumerate(results_df.loc[results_df['expression'] == 1, 'subject'].unique()):
    ax4.plot([0, 1], [results_df.loc[(results_df['subject'] == nickname)
                                     & (results_df['session'].between(-3, -1)), 'bias'].mean(),
                      results_df.loc[(results_df['subject'] == nickname)
                                     & (results_df['session'].between(2, 4)), 'bias'].mean()],
             color=colors['sert'], marker='o')
for i, nickname in enumerate(results_df.loc[results_df['expression'] == 0, 'subject'].unique()):
    ax4.plot([0, 1], [results_df.loc[(results_df['subject'] == nickname)
                                     & (results_df['session'].between(-3, -1)), 'bias'].mean(),
                      results_df.loc[(results_df['subject'] == nickname)
                                     & (results_df['session'].between(2, 4)), 'bias'].mean()],
             color=colors['wt'], marker='o')
ax4.set(ylabel='Bias', ylim=[0, 0.61], xticks=[0, 1], xticklabels=['-3 to -1', '3 to 5'],
        xlabel='Days')

sns.swarmplot(x='expression', y='bias', data=results_df[results_df['session'] > 0].groupby('subject').mean(), ax=ax5)
ax5.set(ylabel='Bias during stimulated days', xticks=[0, 1], xticklabels=['WT', 'SERT'], xlabel='',
        ylim=[0, 0.61])

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'stim-no-stim-bias'))


