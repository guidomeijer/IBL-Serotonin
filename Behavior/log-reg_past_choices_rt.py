#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 10:53:33 2021
By: Guido Meijer
"""

from os.path import join
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from serotonin_functions import (paths, behavioral_criterion, load_trials, figure_style,
                                 query_opto_sessions, load_subjects, fit_glm)
from one.api import ONE
one = ONE()
log_reg = LogisticRegression(random_state=42)

# Settings
RT_CUTOFF = 0.5
REWARD_WIN = 10  # trials
MIN_TRIALS = 3  # for estimating reward bias
subjects = load_subjects(behavior=True)
_, fig_path, save_path = paths()
fig_path = join(fig_path, 'Behavior', 'ModelAgnostic')

results_df = pd.DataFrame()
for i, nickname in enumerate(subjects['subject']):

    # Query sessions
    eids = query_opto_sessions(nickname, one=one)
    eids = behavioral_criterion(eids, one=one)

    # Get trials DataFrame
    all_trials = pd.DataFrame()
    for j, eid in enumerate(eids):
        try:
            if subjects.loc[i, 'sert-cre'] == 1:
                trials = load_trials(eid, laser_stimulation=True, patch_old_opto=True, one=one)
            else:
                trials = load_trials(eid, laser_stimulation=True, patch_old_opto=False, one=one)
        except:
            continue

        for t in range(REWARD_WIN+1, trials.shape[0]):
            trials_slice = trials[t-(REWARD_WIN+1):t-1]

            # reward bias opto
            stim_trials = trials_slice[trials_slice['laser_stimulation'] == 1]
            if stim_trials.shape[0] >= MIN_TRIALS:
                trials.loc[t, 'past_choices_opto'] = np.sum(stim_trials['choice'])

            # reward bias no opto
            no_stim_trials = trials_slice[trials_slice['laser_stimulation'] == 0]
            if no_stim_trials.shape[0] >= MIN_TRIALS:
                trials.loc[t, 'past_choices_no_opto'] = np.sum(no_stim_trials['choice'])

        # Add to trial dataframe
        all_trials = all_trials.append(trials, ignore_index=True)

    # Predict choices with logistic regression
    trials_slice = all_trials.loc[~all_trials['past_choices_opto'].isnull()]
    log_reg.fit(trials_slice.loc[trials_slice['reaction_times'] > RT_CUTOFF, 'past_choices_opto'].values.reshape(-1, 1),
                trials_slice.loc[trials_slice['reaction_times'] > RT_CUTOFF, 'choice'])
    opto_long = log_reg.score(trials_slice.loc[trials_slice['reaction_times'] > RT_CUTOFF, 'past_choices_opto'].values.reshape(-1, 1),
                              trials_slice.loc[trials_slice['reaction_times'] > RT_CUTOFF, 'choice'])
    log_reg.fit(trials_slice.loc[trials_slice['reaction_times'] < RT_CUTOFF, 'past_choices_opto'].values.reshape(-1, 1),
                trials_slice.loc[trials_slice['reaction_times'] < RT_CUTOFF, 'choice'])
    opto_short = log_reg.score(trials_slice.loc[trials_slice['reaction_times'] < RT_CUTOFF, 'past_choices_opto'].values.reshape(-1, 1),
                                 trials_slice.loc[trials_slice['reaction_times'] < RT_CUTOFF, 'choice'])

    trials_slice = all_trials.loc[~all_trials['past_choices_no_opto'].isnull()]
    log_reg.fit(trials_slice.loc[trials_slice['reaction_times'] > RT_CUTOFF, 'past_choices_no_opto'].values.reshape(-1, 1),
                trials_slice.loc[trials_slice['reaction_times'] > RT_CUTOFF, 'choice'])
    no_opto_long = log_reg.score(trials_slice.loc[trials_slice['reaction_times'] > RT_CUTOFF, 'past_choices_no_opto'].values.reshape(-1, 1),
                                 trials_slice.loc[trials_slice['reaction_times'] > RT_CUTOFF, 'choice'])
    log_reg.fit(trials_slice.loc[trials_slice['reaction_times'] < RT_CUTOFF, 'past_choices_no_opto'].values.reshape(-1, 1),
                trials_slice.loc[trials_slice['reaction_times'] < RT_CUTOFF, 'choice'])
    no_opto_short = log_reg.score(trials_slice.loc[trials_slice['reaction_times'] < RT_CUTOFF, 'past_choices_no_opto'].values.reshape(-1, 1),
                                  trials_slice.loc[trials_slice['reaction_times'] < RT_CUTOFF, 'choice'])

    results_df = results_df.append(pd.DataFrame(data={
        'subject': nickname, 'sert-cre': subjects.loc[i, 'sert-cre'],
        'long_rt': [opto_long, no_opto_long], 'short_rt': [opto_short, no_opto_short],
        'opto': [1, 0]}), ignore_index=True)

# %% Plot

plot_df = results_df.copy()
plot_df.loc[plot_df['sert-cre'] == 1, 'sert-cre'] = 'Sert'
plot_df.loc[plot_df['sert-cre'] == 0, 'sert-cre'] = 'WT'
plot_df.loc[plot_df['opto'] == 1, 'opto'] = 'Stim'
plot_df.loc[plot_df['opto'] == 0, 'opto'] = 'No stim'

colors, dpi = figure_style()
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(4, 4), dpi=dpi)
sns.lineplot(x='opto', y='long_rt', data=plot_df, hue='sert-cre', estimator=None, units='subject',
             palette=[colors['sert'], colors['wt']], legend='brief', dashes=False,
             markers=['o']*int(plot_df.shape[0]/2), ax=ax1)
ax1.legend(frameon=False)
ax1.set(xlabel='', xticks=[0, 1], xticklabels=['No stim', 'Stim'], ylabel='Corr. reward bias vs choice bias',
        title=f'Long RT (> {RT_CUTOFF}s)', ylim=[0, 1])

sns.lineplot(x='opto', y='short_rt', data=plot_df, hue='sert-cre', estimator=None, units='subject',
             palette=[colors['sert'], colors['wt']], legend=None, dashes=False,
             markers=['o']*int(plot_df.shape[0]/2), ax=ax2)
ax2.set(xlabel='', xticks=[0, 1], xticklabels=['No stim', 'Stim'], ylabel='Corr. reward bias vs choice bias',
        title=f'Short RT (< {RT_CUTOFF}s)', ylim=[0, 1])

sns.barplot(x='sert-cre', y='long_rt', data=plot_df, hue='opto', palette=[colors['no-stim'], colors['stim']],
            order=['WT', 'Sert'], ax=ax3)
ax3.set(xlabel='', ylabel='Corr. reward bias vs choice bias', ylim=[0, 1])
ax3.legend(frameon=False, bbox_to_anchor=(0.65, 0.8))
#ax3.get_legend().remove()

sns.barplot(x='sert-cre', y='short_rt', data=plot_df, hue='opto', palette=[colors['no-stim'], colors['stim']],
            order=['WT', 'Sert'], ax=ax4)
ax4.set(xlabel='', ylabel='Corr. reward bias vs choice bias', ylim=[0, 1])
ax4.get_legend().remove()

plt.tight_layout(pad=3)
sns.despine(trim=True)
plt.savefig(join(fig_path, 'past_choices_pred'), dpi=300)

