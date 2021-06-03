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
from behavior_models import utils
from models.expSmoothing_stimside import expSmoothing_stimside as exp_stimside
from models.expSmoothing_prevAction import expSmoothing_prevAction as exp_prev_action
from my_functions import paths, criteria_opto_eids
from oneibl.one import ONE
one = ONE()

# Settings
REMOVE_OLD_FIT = False
PRE_TRIALS = 5
POST_TRIALS = 11
POSTERIOR = 'maximum_a_posteriori'
_, fig_path, save_path = paths()
fig_path = join(fig_path, '5HT', 'opto-behavior')

subjects = pd.read_csv('subjects.csv')

results_df = pd.DataFrame()
block_switches = pd.DataFrame()
for i, nickname in enumerate(subjects['subject']):

    # Query sessions
    if subjects.loc[i, 'date_range'] == 'all':
        eids = one.search(subject=nickname, task_protocol='_iblrig_tasks_opto_biasedChoiceWorld')
    else:
        eids = one.search(subject=nickname, task_protocol='_iblrig_tasks_opto_biasedChoiceWorld',
                          date_range=[subjects.loc[i, 'date_range'][:10], subjects.loc[i, 'date_range'][11:]])
    #eids = criteria_opto_eids(eids, max_lapse=0.3, max_bias=0.5, min_trials=200)
    if len(eids) == 0:
        continue

    # Get trial data
    stimuli_arr, actions_arr, stim_sides_arr, prob_left, session_uuids = [], [], [], [], []
    ses_count = 0
    for j, eid in enumerate(eids):
        try:
            data = utils.load_session(eid)
            laser_stimulation = one.load(eid, dataset_types=['_ibl_trials.laser_stimulation'])[0]
            stim_side, stimuli, actions, pLeft_oracle = utils.format_data(data)
            stimuli_arr.append(stimuli)
            actions_arr.append(actions)
            stim_sides_arr.append(stim_side)
            prob_left.append(data['probabilityLeft'])
            session_uuids.append(eid)
        except Exception:
            print('Could not load trials for %s' % eid)

    # get maximum number of trials across sessions
    max_len = np.array([len(stimuli_arr[k]) for k in range(len(stimuli_arr))]).max()

    # pad with 0 such that we obtain nd arrays of size nb_sessions x nb_trials and convert to arrays
    stimuli = np.array([np.concatenate((stimuli_arr[k], np.zeros(max_len-len(stimuli_arr[k]))))
                        for k in range(len(stimuli_arr))])
    actions = np.array([np.concatenate((actions_arr[k], np.zeros(max_len-len(actions_arr[k]))))
                        for k in range(len(actions_arr))])
    stim_side = np.array([np.concatenate((stim_sides_arr[k], np.zeros(max_len-len(stim_sides_arr[k]))))
                          for k in range(len(stim_sides_arr))])
    session_uuids = np.array(session_uuids)

    # Fit models
    model = exp_stimside('./model_fit_results/', session_uuids, '%s_1_tau' % nickname,
                         actions, stimuli, stim_side)
    model.load_or_train(nb_steps=2000, remove_old=REMOVE_OLD_FIT)
    acc_stimside = model.compute_signal(signal='score', act=actions, stim=stimuli,
                                         side=stim_side)['accuracy']
    results_df = results_df.append(pd.DataFrame(index=[len(results_df)],
                                                data={'accuracy': acc_stimside, 'model': 'stimside',
                                                      'sert-cre': subjects.loc[i, 'sert-cre'],
                                                      'subject': nickname}))

    model = exp_prev_action('./model_fit_results/', session_uuids, '%s_1_tau' % nickname,
                            actions, stimuli, stim_side)
    model.load_or_train(nb_steps=2000, remove_old=REMOVE_OLD_FIT)
    acc_prevaction = model.compute_signal(signal='score', act=actions, stim=stimuli,
                                          side=stim_side)['accuracy']
    results_df = results_df.append(pd.DataFrame(index=[len(results_df)],
                                                data={'accuracy': acc_prevaction, 'model': 'prevaction',
                                                      'sert-cre': subjects.loc[i, 'sert-cre'],
                                                      'subject': nickname}))


# %% Plot
sns.set(context='talk', style='ticks', font_scale=1.5)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

sns.lineplot(x='model', y='accuracy', hue='sert-cre', style='subject', estimator=None,
             data=results_df, dashes=False, markers=['o']*int(results_df.shape[0]/2),
             legend=False, ax=ax1)
ax1.set(xlabel='', ylabel='Lenght of integration window (tau)')

"""
sns.lineplot(x='trial', y='prior_stimside', data=block_switches,
             hue='change_to', style='opto', palette='colorblind', ax=ax2, ci=68)
#plt.plot([0, 0], [0, 1], color=[0.5, 0.5, 0.5], ls='--')
handles, labels = ax2.get_legend_handles_labels()
labels = ['', 'Change to R', 'Change to L', '', 'No stim', 'Stim']
ax2.legend(handles, labels, frameon=False, prop={'size': 20})
ax2.set(ylabel='Prior', xlabel='Trials relative to block switch',
        title='Exponential smoothed previous stimulus side model')
"""

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'model_prevaction_opto_behavior'))