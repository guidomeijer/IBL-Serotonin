#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 11:42:01 2021

@author: guido
"""

import numpy as np
from os.path import join
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from serotonin_functions import (load_trials, plot_psychometric, paths, behavioral_criterion,
                                 fit_psychfunc, figure_style, query_opto_sessions, get_bias,
                                 load_subjects)
from one.api import ONE
one = ONE()

# Settings
RT_CUTOFF = 0.5
PLOT_SINGLE_ANIMALS = True
fig_path, _ = paths()
fig_path = join(fig_path, 'Behavior', 'Psychometrics')
subjects = load_subjects(behavior=True)

bias_df, lapse_df, psy_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
for i, nickname in enumerate(subjects['subject']):

    # Query sessions
    eids = query_opto_sessions(nickname, one=one)

    # Exclude the first opto sessions
    #eids = eids[:-2]

    # Apply behavioral criterion
    eids = behavioral_criterion(eids, one=one)
    if len(eids) == 0:
        continue

    # Get trials DataFrame
    trials = pd.DataFrame()
    for j, eid in enumerate(eids):
        try:
            if subjects.loc[i, 'sert-cre'] == 1:
                these_trials = load_trials(eid, laser_stimulation=True, patch_old_opto=False, one=one)
            else:
                these_trials = load_trials(eid, laser_stimulation=True, patch_old_opto=False, one=one)
        except:
            continue
        trials = trials.append(these_trials, ignore_index=True)
    if len(trials) < 400:
        continue

    # Get fit parameters
    these_trials = trials[(trials['probabilityLeft'] == 0.8) & (trials['laser_stimulation'] == 1)
                          & (trials['reaction_times'] < RT_CUTOFF)]
    pars = fit_psychfunc(np.sort(these_trials['signed_contrast'].unique()),
                         these_trials.groupby('signed_contrast').size(),
                         these_trials.groupby('signed_contrast').mean()['right_choice'])
    psy_df = psy_df.append(pd.DataFrame(index=[len(psy_df)+1], data={
        'subject': nickname, 'sert-cre': subjects.loc[i, 'sert-cre'], 'opto_stim': 1, 'prob_left': 0.8, 'rt': 'fast',
        'bias': pars[0], 'threshold': pars[1], 'lapse_l': pars[2], 'lapse_r': pars[3]}))

    these_trials = trials[(trials['probabilityLeft'] == 0.2) & (trials['laser_stimulation'] == 1)
                          & (trials['reaction_times'] > RT_CUTOFF)]
    pars = fit_psychfunc(np.sort(these_trials['signed_contrast'].unique()),
                         these_trials.groupby('signed_contrast').size(),
                         these_trials.groupby('signed_contrast').mean()['right_choice'])
    psy_df = psy_df.append(pd.DataFrame(index=[len(psy_df)+1], data={
        'subject': nickname, 'sert-cre': subjects.loc[i, 'sert-cre'], 'opto_stim': 1, 'prob_left': 0.2, 'rt': 'slow',
        'bias': pars[0], 'threshold': pars[1], 'lapse_l': pars[2], 'lapse_r': pars[3]}))

    these_trials = trials[(trials['probabilityLeft'] == 0.8) & (trials['laser_stimulation'] == 0)
                          & (trials['reaction_times'] < RT_CUTOFF)]
    pars = fit_psychfunc(np.sort(these_trials['signed_contrast'].unique()),
                         these_trials.groupby('signed_contrast').size(),
                         these_trials.groupby('signed_contrast').mean()['right_choice'])
    psy_df = psy_df.append(pd.DataFrame(index=[len(psy_df)+1], data={
        'subject': nickname, 'sert-cre': subjects.loc[i, 'sert-cre'], 'opto_stim': 0, 'prob_left': 0.8, 'rt': 'fast',
        'bias': pars[0], 'threshold': pars[1], 'lapse_l': pars[2], 'lapse_r': pars[3]}))

    these_trials = trials[(trials['probabilityLeft'] == 0.2) & (trials['laser_stimulation'] == 0)
                          & (trials['reaction_times'] > RT_CUTOFF)]
    pars = fit_psychfunc(np.sort(these_trials['signed_contrast'].unique()),
                         these_trials.groupby('signed_contrast').size(),
                         these_trials.groupby('signed_contrast').mean()['right_choice'])
    psy_df = psy_df.append(pd.DataFrame(index=[len(psy_df)+1], data={
        'subject': nickname, 'sert-cre': subjects.loc[i, 'sert-cre'], 'opto_stim': 0, 'prob_left': 0.2, 'rt': 'slow',
        'bias': pars[0], 'threshold': pars[1], 'lapse_l': pars[2], 'lapse_r': pars[3]}))

    # Plot
    if PLOT_SINGLE_ANIMALS:
        colors, dpi = figure_style()
        f, ax1 = plt.subplots(1, 1, figsize=(2, 2), dpi=dpi)

        # plot_psychometric(trials[trials['probabilityLeft'] == 0.5], ax=ax1, color='k')
        plot_psychometric(trials[(trials['probabilityLeft'] == 0.8)
                                 & (trials['laser_stimulation'] == 0)
                                 & (trials['reaction_times'] > RT_CUTOFF)], ax=ax1, color=colors['left'])
        plot_psychometric(trials[(trials['probabilityLeft'] == 0.8)
                                 & (trials['laser_stimulation'] == 1)
                                 & (trials['reaction_times'] > RT_CUTOFF)], ax=ax1,
                          color=colors['left'], linestyle='--')
        plot_psychometric(trials[(trials['probabilityLeft'] == 0.2)
                                 & (trials['laser_stimulation'] == 0)
                                 & (trials['reaction_times'] > RT_CUTOFF)], ax=ax1, color=colors['right'])
        plot_psychometric(trials[(trials['probabilityLeft'] == 0.2)
                                 & (trials['laser_stimulation'] == 1)
                                 & (trials['reaction_times'] > RT_CUTOFF)], ax=ax1,
                          color=colors['right'], linestyle='--')
        #ax1.text(-20, 0.75, '80% right', color=colors['right'])
        #ax1.text(20, 0.25, '80% left', color=colors['left'])
        ax1.set(title='dashed line = opto stim')

        sns.despine(trim=True)
        plt.tight_layout()

        plt.savefig(join(fig_path, '%s_opto_rt_slow_psycurve.png' % nickname), dpi=300)
        plt.savefig(join(fig_path, '%s_opto_rt_slow_psycurve.pdf' % nickname))


psy_df['bias_abs'] = psy_df['bias'].abs()
psy_sum_df = psy_df.groupby(['subject', 'opto_stim', 'rt']).sum().reset_index()

# %% Plot
colors, dpi = figure_style()
colors = [colors['wt'], colors['sert']]

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2), dpi=dpi)
sns.lineplot(x='rt', y='bias_abs', data=psy_sum_df, hue='sert-cre', style='opto_stim',
             estimator=None, units='subject', palette=colors, ax=ax1)

plt.tight_layout()
sns.despine(trim=True)
#plt.savefig(join(fig_path, 'summary_psycurve.pdf'))

