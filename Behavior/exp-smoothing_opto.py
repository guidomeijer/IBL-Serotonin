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
import torch
from matplotlib.patches import Rectangle
from models.expSmoothing_stimside_SE import expSmoothing_stimside_SE as exp_stimside
from models.expSmoothing_prevAction_SE import expSmoothing_prevAction_SE as exp_prev_action
from serotonin_functions import (paths, behavioral_criterion, load_exp_smoothing_trials, figure_style,
                                 query_opto_sessions, load_subjects)
from one.api import ONE
one = ONE()

# Settings
PRE_TRIALS = 5
POST_TRIALS = 20
PLOT_EXAMPLES = True
INCLUDE_EPHYS = True
REMOVE_OLD_FIT = False
POSTERIOR = 'posterior_mean'
STIM = 20
fig_path, save_path = paths()
fig_path = join(fig_path, 'Behavior', 'Models')

subjects = load_subjects()

results_df = pd.DataFrame()
block_switches = pd.DataFrame()
for i, nickname in enumerate(subjects['subject']):

    # Query sessions
    eids = query_opto_sessions(nickname, include_ephys=INCLUDE_EPHYS, one=one)
    eids = behavioral_criterion(eids, verbose=False, one=one)
    if len(eids) == 0:
        continue
    if len(eids) > 10:
        eids = eids[:10]

    # Get trial data
    if subjects.loc[i, 'sert-cre'] == 1:
        actions, stimuli, stim_side, prob_left, stim_trials, session_uuids = load_exp_smoothing_trials(
            eids, stimulated=STIM, patch_old_opto=True, one=one)
    else:
        actions, stimuli, stim_side, prob_left, stim_trials, session_uuids = load_exp_smoothing_trials(
            eids, stimulated=STIM, patch_old_opto=False, one=one)

    # Fit model
    model = exp_prev_action('./model_fit_results/', session_uuids, '%s_%s' % (nickname, STIM),
                            actions, stimuli, stim_side, torch.tensor(stim_trials))
    model.load_or_train(nb_steps=2000, remove_old=REMOVE_OLD_FIT)
    param_prevaction = model.get_parameters(parameter_type=POSTERIOR)
    priors_prevaction = model.compute_signal(signal='prior', act=actions, stim=stimuli, side=stim_side)['prior']

    # Add to df
    results_df = results_df.append(pd.DataFrame(data={'tau_pa': [1/param_prevaction[0], 1/param_prevaction[1]],
                                                      'opto_stim': ['no stim', 'stim'],
                                                      'sert-cre': subjects.loc[i, 'sert-cre'],
                                                      'subject': nickname}))

    # Add prior around block switches
    if prob_left.shape[0] == 1:
        priors_prevaction = [priors_prevaction]
    for k in range(len(priors_prevaction)):
        transitions = np.array(np.where(np.diff(prob_left[k]) != 0)[0])[:-1] + 1
        for t, trans in enumerate(transitions):
            if trans >= PRE_TRIALS:
                if stim_trials[k][trans] == 1:
                    opto = 'stim'
                elif stim_trials[k][trans] == 0:
                    opto = 'no stim'
                block_switches = pd.concat((block_switches, pd.DataFrame(data={
                            'prior_prevaction': priors_prevaction[k][trans-PRE_TRIALS:trans+POST_TRIALS],
                            'trial': np.append(np.arange(-PRE_TRIALS, 0), np.arange(0, POST_TRIALS)),
                            'change_to': prob_left[k][trans],
                            'opto': opto,
                            'sert_cre': subjects.loc[i, 'sert-cre'],
                            'subject': nickname})))
    block_switches = block_switches.reset_index(drop=True)

    # Plot for this animal
    if PLOT_EXAMPLES:
        colors, dpi = figure_style()
        f, ax1 = plt.subplots(1, 1, figsize=(3, 2), dpi=dpi)
        sns.lineplot(x='trial', y='prior_prevaction', data=block_switches[block_switches['subject'] == nickname],
                 hue='change_to', style='opto', palette='colorblind', ax=ax1, errorbar='se')
        #plt.plot([0, 0], [0, 1], color=[0.5, 0.5, 0.5], ls='--')
        handles, labels = ax1.get_legend_handles_labels()
        labels = ['', 'Change to R', 'Change to L', '', 'No stim', 'Stim']
        ax1.legend(handles, labels, frameon=False, bbox_to_anchor=(1, 1))
        ax1.set(ylabel='Prior', xlabel='Trials relative to block switch', ylim=[0, 1],
                title='Tau stim: %.2f, Tau no stim: %.2f' % (1/param_prevaction[1], 1/param_prevaction[0]))
        sns.despine(trim=True)
        plt.tight_layout()
        plt.savefig(join(fig_path, '%s_model_prevaction' % nickname), dpi=300)

        # Plot priors of example session
        if np.sum(stim_side[0] == 0) == 0:
            these_priors = priors_prevaction[0]
            these_p_left = prob_left[0]
            these_stim = stim_trials[0]
        else:
            these_priors = priors_prevaction[0][:np.where(stim_side[0] == 0)[0][0] - 1]
            these_p_left = prob_left[0][:np.where(stim_side[0] == 0)[0][0] - 1]
            these_stim = stim_trials[0][:np.where(stim_side[0] == 0)[0][0] - 1]
        BLOCK_COLORS = (colors['left'], colors['right'])
        f, ax1 = plt.subplots(1, 1, figsize=(6, 3), dpi=dpi)
        trial_blocks = (these_p_left == 0.2).astype(int)
        block_trans = np.append([0], np.array(np.where(np.diff(these_p_left) != 0)) + 1)
        block_trans = np.append(block_trans, [these_p_left.shape[0]])
        for j, trans in enumerate(block_trans[:-1]):
            p = Rectangle((trans, -0.05), block_trans[j+1] - trans, 1.05, alpha=0.5,
                          color=BLOCK_COLORS[trial_blocks[trans]])
            ax1.add_patch(p)
        ax1.plot(np.arange(1, these_priors.shape[0] + 1), these_priors, color='k')
        these_stim[these_stim == 0] = np.nan
        these_stim[these_stim == 1] = 1.1
        ax1.plot(np.arange(1, these_priors.shape[0] + 1), these_stim, lw=2, color=colors['stim'])
        ax1.set(ylabel='Prior', xlabel='Trials', title=f'{nickname}')
        sns.despine(trim=True)
        plt.tight_layout()
        plt.savefig(join(fig_path, '%s_prevaction_example_session' % nickname), dpi=300)


# %% Plot

colors, dpi = figure_style()
colors = [colors['wt'], colors['sert']]
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(3, 2), dpi=dpi, sharey=False)

for i, subject in enumerate(results_df['subject']):
    ax1.plot([1, 2], results_df.loc[(results_df['subject'] == subject), 'tau_pa'],
             color = colors[results_df.loc[results_df['subject'] == subject, 'sert-cre'].unique()[0]],
             marker='o', ms=2)

handles, labels = ax1.get_legend_handles_labels()
labels = ['', 'WT', 'SERT']
ax1.legend(handles[:3], labels[:3], frameon=False, prop={'size': 7}, loc='center left', bbox_to_anchor=(1, .5))
ax1.set(xlabel='', ylabel='Length of integration window (tau)', title='Previous actions',
        xticks=[1, 2], xticklabels=['No stim', 'Stim'], ylim=[0, 15])

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, f'exp-smoothing_opto_{STIM}'), dpi=300)
