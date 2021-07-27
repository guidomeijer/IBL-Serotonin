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
import torch
from models.expSmoothing_prevAction_SE import expSmoothing_prevAction_SE as exp_prev_action
from serotonin_functions import paths, criteria_opto_eids, load_exp_smoothing_trials
from oneibl.one import ONE
one = ONE()

# Settings
REMOVE_OLD_FIT = False
PRE_TRIALS = 5
POST_TRIALS = 16
POSTERIOR = 'posterior_mean'
_, fig_path, save_path = paths()
fig_path = join(fig_path, '5HT', 'opto-behavior-first-session')

subjects = pd.read_csv(join('..', 'subjects.csv'))

results_df = pd.DataFrame()
accuracy_df = pd.DataFrame()
block_switches = pd.DataFrame()
for i, nickname in enumerate(subjects['subject']):

    # Query sessions
    if subjects.loc[i, 'date_range_blocks'] == 'all':
        eids, ses_details = one.search(subject=nickname,
                                       task_protocol='_iblrig_tasks_opto_biasedChoiceWorld',
                                       details=True)
    elif subjects.loc[i, 'date_range_blocks'] == 'none':
        continue
    else:
        eids, ses_details = one.search(subject=nickname, task_protocol='_iblrig_tasks_opto_biasedChoiceWorld',
                                       date_range=[subjects.loc[i, 'date_range_blocks'][:10],
                                                   subjects.loc[i, 'date_range_blocks'][11:]],
                                       details=True)

    #eids = criteria_opto_eids(eids, max_lapse=0.5, max_bias=0.5, min_trials=200, one=one)
    if len(eids) == 0:
        continue

    # Get session dates as datetime object
    ses_dates = [datetime.strptime(i, '%Y-%m-%d') for i in [j['start_time'][:10] for j in ses_details]]
    ses_dates.sort()
    first_day = ses_dates[0]

    # Fit model
    eid = one.search(subject=nickname, task_protocol='_iblrig_tasks_opto_biasedChoiceWorld',
                                   date_range=[str(first_day)[:10]])[0]
    actions, stimuli, stim_side, prob_left, stimulated, session_uuids = load_exp_smoothing_trials(
        [eid], laser_stimulation=True, one=one)
    if len(session_uuids) > 0:
        model = exp_prev_action('./model_fit_results/', session_uuids, f'{nickname}_first_day',
                                actions, stimuli, stim_side, torch.tensor(stimulated))
        model.load_or_train(nb_steps=2000, remove_old=REMOVE_OLD_FIT)
        params = model.get_parameters(parameter_type=POSTERIOR)
        priors = model.compute_signal(signal='prior', act=actions, stim=stimuli, side=stim_side)['prior']
        results_df = results_df.append(pd.DataFrame(data={
            'tau': [1/params[0], 1/params[1]], 'opto_stim': ['no stim', 'stim'],
            'sert-cre': subjects.loc[i, 'sert-cre'], 'subject': nickname}))

    # Add prior around block switch non-stimulated
    transitions = np.array(np.where(np.diff(prob_left[0]) != 0)[0])[:-1] + 1
    for t, trans in enumerate(transitions):
        if trans >= PRE_TRIALS:
            if stimulated[0][trans] == 1:
                opto = 'stim'
            elif stimulated[0][trans] == 0:
                opto = 'no stim'
            block_switches = block_switches.append(pd.DataFrame(data={
                        'prior': priors[trans-PRE_TRIALS:trans+POST_TRIALS],
                        'trial': np.append(np.arange(-PRE_TRIALS, 0), np.arange(0, POST_TRIALS)),
                        'change_to': prob_left[0][trans],
                        'opto': opto,
                        'sert_cre': subjects.loc[i, 'sert-cre'],
                        'subject': nickname}))

    # Plot for this animal
    sns.set(context='talk', style='ticks', font_scale=1.5)
    f, ax1 = plt.subplots(1, 1, figsize=(10, 10))
    sns.lineplot(x='trial', y='prior', data=block_switches[block_switches['subject'] == nickname],
             hue='change_to', style='opto', palette='colorblind', ax=ax1, ci=68)
    #plt.plot([0, 0], [0, 1], color=[0.5, 0.5, 0.5], ls='--')
    handles, labels = ax1.get_legend_handles_labels()
    labels = ['', 'Change to R', 'Change to L', '', 'No stim', 'Stim']
    ax1.legend(handles, labels, frameon=False, prop={'size': 20})
    ax1.set(ylabel='Prior', xlabel='Trials relative to block switch',
            title='Tau stim: %.2f, Tau no stim: %.2f' % (1/params[1], 1/params[0]))
    sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig(join(fig_path, '%s_model_prevactions' % nickname))

# %% Plot

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(25, 10))
sns.lineplot(x='opto_stim', y='tau', hue='sert-cre', style='subject', estimator=None,
             data=results_df, dashes=False, markers=['o']*int(results_df.shape[0]/2),
             legend=False, ax=ax1)
ax1.set(xlabel='', ylabel='Lenght of integration window (tau)',
        title='Exponential smoothed previous actions model')

sns.lineplot(x='trial', y='prior', data=block_switches[block_switches['sert_cre'] == 1],
             hue='change_to', style='opto', palette='colorblind', ax=ax2, ci=68)
#plt.plot([0, 0], [0, 1], color=[0.5, 0.5, 0.5], ls='--')
handles, labels = ax2.get_legend_handles_labels()
labels = ['', 'Change to R', 'Change to L', '', 'No stim', 'Stim']
ax2.legend(handles, labels, frameon=False, prop={'size': 20})
ax2.set(ylabel='Prior', xlabel='Trials relative to block switch', title='Sert-Cre', ylim=[0, 1])

sns.lineplot(x='trial', y='prior', data=block_switches[block_switches['sert_cre'] == 0],
             hue='change_to', style='opto', palette='colorblind', ax=ax3, ci=68)
#plt.plot([0, 0], [0, 1], color=[0.5, 0.5, 0.5], ls='--')
handles, labels = ax2.get_legend_handles_labels()
labels = ['', 'Change to R', 'Change to L', '', 'No stim', 'Stim']
ax3.legend(handles, labels, frameon=False, prop={'size': 20})
ax3.set(ylabel='Prior', xlabel='Trials relative to block switch', title='WT control', ylim=[0, 1])

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'model_prevaction_opto_behavior'))

