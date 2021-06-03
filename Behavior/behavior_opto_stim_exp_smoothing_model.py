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
subjects = subjects[subjects['subject'] == 'ZFM-01867'].reset_index(drop=True)

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
    stimuli_arr_ns, actions_arr_ns, stim_sides_arr_ns, prob_left_ns, session_uuids = [], [], [], [], []
    stimuli_arr_s, actions_arr_s, stim_sides_arr_s, prob_left_s = [], [], [], []
    ses_count = 0
    for j, eid in enumerate(eids):
        try:
            data = utils.load_session(eid)
            laser_stimulation = one.load(eid, dataset_types=['_ibl_trials.laser_stimulation'])[0]
            stim_side, stimuli, actions, pLeft_oracle = utils.format_data(data)
            stimuli_arr_ns.append(stimuli[laser_stimulation == 0])
            actions_arr_ns.append(actions[laser_stimulation == 0])
            stim_sides_arr_ns.append(stim_side[laser_stimulation == 0])
            prob_left_ns.append(data['probabilityLeft'][laser_stimulation == 0])
            stimuli_arr_s.append(stimuli[laser_stimulation == 1])
            actions_arr_s.append(actions[laser_stimulation == 1])
            stim_sides_arr_s.append(stim_side[laser_stimulation == 1])
            prob_left_s.append(data['probabilityLeft'][laser_stimulation == 1])
            session_uuids.append(eid)
        except Exception:
            print('Could not load trials for %s' % eid)

    # get maximum number of trials across sessions
    max_len_ns = np.array([len(stimuli_arr_ns[k]) for k in range(len(stimuli_arr_ns))]).max()
    max_len_s = np.array([len(stimuli_arr_s[k]) for k in range(len(stimuli_arr_s))]).max()

    # pad with 0 such that we obtain nd arrays of size nb_sessions x nb_trials and convert to arrays
    stimuli_ns = np.array([np.concatenate((stimuli_arr_ns[k], np.zeros(max_len_ns-len(stimuli_arr_ns[k]))))
                           for k in range(len(stimuli_arr_ns))])
    actions_ns = np.array([np.concatenate((actions_arr_ns[k], np.zeros(max_len_ns-len(actions_arr_ns[k]))))
                           for k in range(len(actions_arr_ns))])
    stim_side_ns = np.array([np.concatenate((stim_sides_arr_ns[k], np.zeros(max_len_ns-len(stim_sides_arr_ns[k]))))
                             for k in range(len(stim_sides_arr_ns))])
    stimuli_s = np.array([np.concatenate((stimuli_arr_s[k], np.zeros(max_len_s-len(stimuli_arr_s[k]))))
                          for k in range(len(stimuli_arr_ns))])
    actions_s = np.array([np.concatenate((actions_arr_s[k], np.zeros(max_len_s-len(actions_arr_s[k]))))
                          for k in range(len(actions_arr_s))])
    stim_side_s = np.array([np.concatenate((stim_sides_arr_s[k], np.zeros(max_len_s-len(stim_sides_arr_s[k]))))
                            for k in range(len(stim_sides_arr_s))])
    session_uuids = np.array(session_uuids)

    # Fit models
    model = exp_stimside('./model_fit_results/', session_uuids, '%s_no_stim' % nickname,
                         actions_ns, stimuli_ns, stim_side_ns)
    model.load_or_train(nb_steps=2000, remove_old=REMOVE_OLD_FIT)
    param_ss_ns = model.get_parameters(parameter_type=POSTERIOR)
    priors_stimside = model.compute_signal(signal='prior', act=actions_ns, stim=stimuli_ns,
                                           side=stim_side_ns)['prior']

    model = exp_prev_action('./model_fit_results/', session_uuids, '%s_no_stim' % nickname,
                            actions_ns, stimuli_ns, stim_side_ns)
    model.load_or_train(nb_steps=2000, remove_old=REMOVE_OLD_FIT)
    param_pa_ns = model.get_parameters(parameter_type=POSTERIOR)
    priors_prevaction = model.compute_signal(signal='prior', act=actions_ns, stim=stimuli_ns,
                                             side=stim_side_ns)['prior']
    results_df = results_df.append(pd.DataFrame(index=[len(results_df)],
                                                data={'tau_ss': 1/param_ss_ns[0],
                                                      'tau_pa': 1/param_pa_ns[0],
                                                      'opto_stim': 'no stim',
                                                      'sert-cre': subjects.loc[i, 'sert-cre'],
                                                      'subject': nickname}))

    # Add prior around block switch non-stimulated
    for k in range(len(priors_stimside)):
        transitions = np.array(np.where(np.diff(prob_left_ns[k]) != 0)[0]) + 1
        for t, trans in enumerate(transitions):
            if trans >= PRE_TRIALS:
                block_switches = block_switches.append(pd.DataFrame(data={
                            'prior_stimside': priors_stimside[k][trans-PRE_TRIALS:trans+POST_TRIALS],
                            'prior_prevaction': priors_prevaction[k][trans-PRE_TRIALS:trans+POST_TRIALS],
                            'trial': np.append(np.arange(-PRE_TRIALS, 0), np.arange(0, POST_TRIALS)),
                            'change_to': prob_left_ns[k][trans],
                            'opto': 'no stim',
                            'sert_cre': subjects.loc[i, 'sert-cre'],
                            'subject': nickname}))

    model = exp_stimside('./model_fit_results/', session_uuids, '%s_stim' % nickname,
                         actions_s, stimuli_s, stim_side_s)
    model.load_or_train(nb_steps=2000, remove_old=REMOVE_OLD_FIT)
    param_ss_s = model.get_parameters(parameter_type=POSTERIOR)
    priors_stimside = model.compute_signal(signal='prior', act=actions_s, stim=stimuli_s,
                                           side=stim_side_s)['prior']
    model = exp_prev_action('./model_fit_results/', session_uuids, '%s_stim' % nickname,
                            actions_s, stimuli_s, stim_side_s)
    model.load_or_train(nb_steps=2000, remove_old=REMOVE_OLD_FIT)
    param_pa_s = model.get_parameters(parameter_type=POSTERIOR)
    priors_prevaction = model.compute_signal(signal='prior', act=actions_s, stim=stimuli_s,
                                             side=stim_side_s)['prior']
    results_df = results_df.append(pd.DataFrame(index=[len(results_df)],
                                                data={'tau_ss': 1/param_ss_s[0],
                                                      'tau_pa': 1/param_pa_s[0],
                                                      'opto_stim': 'stim',
                                                      'sert-cre': subjects.loc[i, 'sert-cre'],
                                                      'subject': nickname}))

    # Add prior around block switch stimulated
    for k in range(len(priors_stimside)):
        transitions = np.array(np.where(np.diff(prob_left_s[k]) != 0)[0]) + 1
        for t, trans in enumerate(transitions):
            if PRE_TRIALS > trans:
                pre_trials = trans
            else:
                pre_trials = PRE_TRIALS
            if POST_TRIALS + trans > priors_stimside[k].shape[0]:
                post_trials = POST_TRIALS - (priors_stimside[k].shape[0] - trans)
            else:
                post_trials = POST_TRIALS
            block_switches = block_switches.append(pd.DataFrame(data={
                        'prior_stimside': priors_stimside[k][trans-pre_trials:trans+post_trials],
                        'prior_prevaction': priors_prevaction[k][trans-pre_trials:trans+post_trials],
                        'trial': np.append(np.arange(-pre_trials, 0), np.arange(0, post_trials)),
                        'change_to': prob_left_s[k][trans],
                        'opto': 'stim',
                        'sert_cre': subjects.loc[i, 'sert-cre'],
                        'subject': nickname}))

    # Plot for this animal
    sns.set(context='talk', style='ticks', font_scale=1.5)
    f, ax1 = plt.subplots(1, 1, figsize=(10, 10))
    sns.lineplot(x='trial', y='prior_stimside', data=block_switches[block_switches['subject'] == nickname],
             hue='change_to', style='opto', palette='colorblind', ax=ax1, ci=68)
    #plt.plot([0, 0], [0, 1], color=[0.5, 0.5, 0.5], ls='--')
    handles, labels = ax1.get_legend_handles_labels()
    labels = ['', 'Change to R', 'Change to L', '', 'No stim', 'Stim']
    ax1.legend(handles, labels, frameon=False, prop={'size': 20})
    ax1.set(ylabel='Prior', xlabel='Trials relative to block switch',
            title='Tau stim: %.2f, Tau no stim: %.2f' % (1/param_ss_s[0], 1/param_ss_ns[0]))
    sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig(join(fig_path, '%s_model_stimside' % nickname))

    # Plot for this animal
    f, ax1 = plt.subplots(1, 1, figsize=(10, 10))
    sns.lineplot(x='trial', y='prior_prevaction', data=block_switches[block_switches['subject'] == nickname],
             hue='change_to', style='opto', palette='colorblind', ax=ax1, ci=68)
    #plt.plot([0, 0], [0, 1], color=[0.5, 0.5, 0.5], ls='--')
    handles, labels = ax1.get_legend_handles_labels()
    labels = ['', 'Change to R', 'Change to L', '', 'No stim', 'Stim']
    ax1.legend(handles, labels, frameon=False, prop={'size': 20})
    ax1.set(ylabel='Prior', xlabel='Trials relative to block switch',
            title='Tau stim: %.2f, Tau no stim: %.2f' % (1/param_pa_s[0], 1/param_pa_ns[0]))
    sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig(join(fig_path, '%s_model_prevaction' % nickname))


# %% Plot
sns.set(context='talk', style='ticks', font_scale=1.5)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

sns.lineplot(x='opto_stim', y='tau_ss', hue='sert-cre', style='subject', estimator=None,
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

sns.lineplot(x='opto_stim', y='tau_ss', hue='sert-cre', style='subject', estimator=None,
             data=results_df, dashes=False, markers=['o']*int(results_df.shape[0]/2),
             legend=False, ax=ax1)
ax1.set(xlabel='', ylabel='Lenght of integration window (tau)')


sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'model_stimside_opto_behavior'))


sns.set(context='talk', style='ticks', font_scale=1.5)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

sns.lineplot(x='opto_stim', y='tau_pa', hue='sert-cre', style='subject', estimator=None,
             data=results_df, dashes=False, markers=['o']*int(results_df.shape[0]/2),
             legend=False, ax=ax1)
ax1.set(xlabel='', ylabel='Lenght of integration window (tau)')

sns.lineplot(x='trial', y='prior_prevaction', data=block_switches,
             hue='change_to', style='opto', palette='colorblind', ax=ax2, ci=68)
#plt.plot([0, 0], [0, 1], color=[0.5, 0.5, 0.5], ls='--')
handles, labels = ax2.get_legend_handles_labels()
labels = ['', 'Change to R', 'Change to L', '', 'No stim', 'Stim']
ax2.legend(handles, labels, frameon=False, prop={'size': 20})
ax2.set(ylabel='Prior', xlabel='Trials relative to block switch',
        title='Exponential smoothed previous actions model')

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'model_prevaction_opto_behavior'))