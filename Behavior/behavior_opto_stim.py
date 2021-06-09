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
from serotonin_functions import (load_trials, plot_psychometric, paths, criteria_opto_eids,
                                 fit_psychfunc, figure_style)
from oneibl.one import ONE
one = ONE()

# Settings
_, fig_path, _ = paths()
fig_path = join(fig_path, '5HT', 'opto-behavior')

subjects = pd.read_csv(join('..', 'subjects.csv'))

# testing
#subjects = subjects[subjects['subject'] == 'ZFM-01867'].reset_index(drop=True)

bias_df, lapse_df, psy_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
for i, nickname in enumerate(subjects['subject']):

    # Query sessions
    if subjects.loc[i, 'date_range'] == 'all':
        eids = one.search(subject=nickname, task_protocol='_iblrig_tasks_opto_biasedChoiceWorld')
    elif subjects.loc[i, 'date_range'] == 'none':
        continue
    else:
        eids = one.search(subject=nickname, task_protocol='_iblrig_tasks_opto_biasedChoiceWorld',
                          date_range=[subjects.loc[i, 'date_range'][:10], subjects.loc[i, 'date_range'][11:]])
    #eids = criteria_opto_eids(eids, max_lapse=0.3, max_bias=0.5, min_trials=300, one=one)

    # Get trials DataFrame
    trials = pd.DataFrame()
    ses_count = 0
    for j, eid in enumerate(eids):
        these_trials = load_trials(eid, laser_stimulation=True, one=one)
        if these_trials is not None:
            these_trials['session'] = ses_count
            trials = trials.append(these_trials, ignore_index=True)
            ses_count = ses_count + 1
    if len(trials) == 0:
        continue
    if 'laser_probability' not in trials.columns:
        trials['laser_probability'] = trials['laser_stimulation'].copy()

    # Get bias shift
    bias_no_stim = np.abs(trials[(trials['probabilityLeft'] == 0.8)
                                 & (trials['signed_contrast'] == 0)
                                 & (trials['laser_stimulation'] == 0)
                                 & (trials['laser_probability'] != 0.75)].mean()
                          - trials[(trials['probabilityLeft'] == 0.2)
                                   & (trials['signed_contrast'] == 0)
                                   & (trials['laser_stimulation'] == 0)
                                   & (trials['laser_probability'] != 0.75)].mean())['right_choice']
    bias_stim = np.abs(trials[(trials['probabilityLeft'] == 0.8)
                              & (trials['signed_contrast'] == 0)
                              & (trials['laser_stimulation'] == 1)
                              & (trials['laser_probability'] != 0.25)].mean()
                       - trials[(trials['probabilityLeft'] == 0.2)
                                & (trials['signed_contrast'] == 0)
                                & (trials['laser_stimulation'] == 1)
                                & (trials['laser_probability'] != 0.25)].mean())['right_choice']
    bias_catch_stim = np.abs(trials[(trials['probabilityLeft'] == 0.8)
                                    & (trials['laser_stimulation'] == 1)
                                    & (trials['laser_probability'] == 0.25)].mean()
                             - trials[(trials['probabilityLeft'] == 0.2)
                                      & (trials['laser_stimulation'] == 1)
                                      & (trials['laser_probability'] == 0.25)].mean())['right_choice']
    bias_catch_no_stim = np.abs(trials[(trials['probabilityLeft'] == 0.8)
                                       & (trials['laser_stimulation'] == 0)
                                       & (trials['laser_probability'] == 0.75)].mean()
                                - trials[(trials['probabilityLeft'] == 0.2)
                                         & (trials['laser_stimulation'] == 0)
                                         & (trials['laser_probability'] == 0.75)].mean())['right_choice']
    bias_df = bias_df.append(pd.DataFrame(data={
        'subject': nickname, 'sert-cre': subjects.loc[i, 'sert-cre'],
        'bias': [bias_no_stim, bias_stim, bias_catch_stim, bias_catch_no_stim],
        'opto_stim': [0, 1, 1, 0], 'catch_trial': [0, 0, 1, 1]}))

    # Get lapse rates
    lapse_l_l_ns = trials.loc[(trials['probabilityLeft'] == 0.8) & (trials['signed_contrast'] == -1)
                              & (trials['laser_probability'] == 0), 'correct'].mean()
    lapse_r_l_ns = trials.loc[(trials['probabilityLeft'] == 0.8) & (trials['signed_contrast'] == 1)
                              & (trials['laser_probability'] == 0), 'correct'].mean()
    lapse_l_r_ns = trials.loc[(trials['probabilityLeft'] == 0.2) & (trials['signed_contrast'] == -1)
                              & (trials['laser_probability'] == 0), 'correct'].mean()
    lapse_r_r_ns = trials.loc[(trials['probabilityLeft'] == 0.2) & (trials['signed_contrast'] == 1)
                              & (trials['laser_probability'] == 0), 'correct'].mean()
    lapse_l_l_s = trials.loc[(trials['probabilityLeft'] == 0.8) & (trials['signed_contrast'] == -1)
                              & (trials['laser_probability'] == 1), 'correct'].mean()
    lapse_r_l_s = trials.loc[(trials['probabilityLeft'] == 0.8) & (trials['signed_contrast'] == 1)
                              & (trials['laser_probability'] == 1), 'correct'].mean()
    lapse_l_r_s = trials.loc[(trials['probabilityLeft'] == 0.2) & (trials['signed_contrast'] == -1)
                              & (trials['laser_probability'] == 1), 'correct'].mean()
    lapse_r_r_s = trials.loc[(trials['probabilityLeft'] == 0.2) & (trials['signed_contrast'] == 1)
                              & (trials['laser_probability'] == 1), 'correct'].mean()
    lapse_df = lapse_df.append(pd.DataFrame(data={
        'subject': nickname, 'sert-cre': subjects.loc[i, 'sert-cre'],
        'lapse': [lapse_l_l_ns, lapse_r_l_ns, lapse_l_r_ns, lapse_r_r_ns,
                  lapse_l_l_s, lapse_r_l_s, lapse_l_r_s, lapse_r_r_s],
        'opto_stim': [0, 0, 0, 0, 1, 1, 1, 1],
        'stim_side': ['l', 'r', 'l', 'r', 'l', 'r', 'l', 'r'],
        'bias_side': ['l', 'l', 'r', 'r', 'l', 'l', 'r', 'r']}))

    # Get fit parameters
    these_trials = trials[(trials['probabilityLeft'] == 0.8) & (trials['laser_stimulation'] == 0)
                          & (trials['laser_probability'] != 0.75)]
    stim_levels = np.sort(these_trials['signed_contrast'].unique())
    pars = fit_psychfunc(stim_levels, these_trials.groupby('signed_contrast').size(),
                         these_trials.groupby('signed_contrast').mean()['right_choice'])
    psy_df = psy_df.append(pd.DataFrame(index=[len(psy_df)+1], data={
        'subject': nickname, 'sert-cre': subjects.loc[i, 'sert-cre'], 'opto_stim': 0, 'prob_left': 0.8,
        'bias': pars[0], 'threshold': pars[1], 'lapse_l': pars[2], 'lapse_r': pars[3]}))
    these_trials = trials[(trials['probabilityLeft'] == 0.2) & (trials['laser_stimulation'] == 0)
                          & (trials['laser_probability'] != 0.75)]
    stim_levels = np.sort(these_trials['signed_contrast'].unique())
    pars = fit_psychfunc(stim_levels, these_trials.groupby('signed_contrast').size(),
                         these_trials.groupby('signed_contrast').mean()['right_choice'])
    psy_df = psy_df.append(pd.DataFrame(index=[len(psy_df)+1], data={
        'subject': nickname, 'sert-cre': subjects.loc[i, 'sert-cre'], 'opto_stim': 0, 'prob_left': 0.2,
        'bias': pars[0], 'threshold': pars[1], 'lapse_l': pars[2], 'lapse_r': pars[3]}))
    these_trials = trials[(trials['probabilityLeft'] == 0.8) & (trials['laser_stimulation'] == 1)
                          & (trials['laser_probability'] != 0.25)]
    stim_levels = np.sort(these_trials['signed_contrast'].unique())
    pars = fit_psychfunc(stim_levels, these_trials.groupby('signed_contrast').size(),
                         these_trials.groupby('signed_contrast').mean()['right_choice'])
    psy_df = psy_df.append(pd.DataFrame(index=[len(psy_df)+1], data={
        'subject': nickname, 'sert-cre': subjects.loc[i, 'sert-cre'], 'opto_stim': 1, 'prob_left': 0.8,
        'bias': pars[0], 'threshold': pars[1], 'lapse_l': pars[2], 'lapse_r': pars[3]}))
    these_trials = trials[(trials['probabilityLeft'] == 0.2) & (trials['laser_stimulation'] == 1)
                          & (trials['laser_probability'] != 0.25)]
    stim_levels = np.sort(these_trials['signed_contrast'].unique())
    pars = fit_psychfunc(stim_levels, these_trials.groupby('signed_contrast').size(),
                         these_trials.groupby('signed_contrast').mean()['right_choice'])
    psy_df = psy_df.append(pd.DataFrame(index=[len(psy_df)+1], data={
        'subject': nickname, 'sert-cre': subjects.loc[i, 'sert-cre'], 'opto_stim': 1, 'prob_left': 0.2,
        'bias': pars[0], 'threshold': pars[1], 'lapse_l': pars[2], 'lapse_r': pars[3]}))

    # Plot
    sns.set(context='talk', style='ticks', font_scale=1.5)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), sharey=True)

    # plot_psychometric(trials[trials['probabilityLeft'] == 0.5], ax=ax1, color='k')
    plot_psychometric(trials[(trials['probabilityLeft'] == 0.8)
                             & (trials['laser_stimulation'] == 0)
                             & (trials['laser_probability'] != 0.75)], ax=ax1, color='b')
    plot_psychometric(trials[(trials['probabilityLeft'] == 0.8)
                             & (trials['laser_stimulation'] == 1)
                             & (trials['laser_probability'] != 0.25)], ax=ax1,
                      color='b', linestyle='--')
    plot_psychometric(trials[(trials['probabilityLeft'] == 0.2)
                             & (trials['laser_stimulation'] == 0)
                             & (trials['laser_probability'] != 0.75)], ax=ax1, color='r')
    plot_psychometric(trials[(trials['probabilityLeft'] == 0.2)
                             & (trials['laser_stimulation'] == 1)
                             & (trials['laser_probability'] != 0.25)], ax=ax1,
                      color='r', linestyle='--')
    ax1.text(-25, 0.75, '20:80', color='r')
    ax1.text(25, 0.25, '80:20', color='b')
    ax1.set(title='dashed line = opto stim')

    catch_trials = trials[((trials['laser_probability'] == 0.75) & (trials['laser_stimulation'] == 0))
                          | ((trials['laser_probability'] == 0.25) & (trials['laser_stimulation'] == 1))]

    ax2.errorbar([0, 1],
                 [trials[(trials['probabilityLeft'] == 0.2) & (trials['signed_contrast'] == 0)
                         & (trials['laser_stimulation'] == 1) & (trials['laser_probability'] != 0.25)]['right_choice'].mean(),
                  catch_trials[(catch_trials['probabilityLeft'] == 0.2) & (catch_trials['laser_stimulation'] == 1)]['right_choice'].mean()],
                 [trials[(trials['probabilityLeft'] == 0.2) & (trials['signed_contrast'] == 0)
                         & (trials['laser_stimulation'] == 1) & (trials['laser_probability'] != 0.25)]['right_choice'].sem(),
                  catch_trials[(catch_trials['probabilityLeft'] == 0.2) & (catch_trials['laser_stimulation'] == 1)]['right_choice'].sem()],
                 marker='o', label='Stim', color='r', ls='--')
    ax2.errorbar([0, 1],
                 [trials[(trials['probabilityLeft'] == 0.2) & (trials['signed_contrast'] == 0)
                         & (trials['laser_stimulation'] == 0) & (trials['laser_probability'] != 0.75)]['right_choice'].mean(),
                  catch_trials[(catch_trials['probabilityLeft'] == 0.2) & (catch_trials['laser_stimulation'] == 0)]['right_choice'].mean()],
                 [trials[(trials['probabilityLeft'] == 0.2) & (trials['signed_contrast'] == 0)
                         & (trials['laser_stimulation'] == 0) & (trials['laser_probability'] != 0.75)]['right_choice'].sem(),
                  catch_trials[(catch_trials['probabilityLeft'] == 0.2) & (catch_trials['laser_stimulation'] == 0)]['right_choice'].sem()],
                 marker='o', label='No stim', color='r')
    ax2.errorbar([0, 1],
                 [trials[(trials['probabilityLeft'] == 0.8) & (trials['signed_contrast'] == 0)
                         & (trials['laser_stimulation'] == 1) & (trials['laser_probability'] != 0.25)]['right_choice'].mean(),
                  catch_trials[(catch_trials['probabilityLeft'] == 0.8) & (catch_trials['laser_stimulation'] == 1)]['right_choice'].mean()],
                 [trials[(trials['probabilityLeft'] == 0.8) & (trials['signed_contrast'] == 0)
                         & (trials['laser_stimulation'] == 1) & (trials['laser_probability'] != 0.25)]['right_choice'].sem(),
                  catch_trials[(catch_trials['probabilityLeft'] == 0.8) & (catch_trials['laser_stimulation'] == 1)]['right_choice'].sem()],
                 marker='o', label='Stim', color='b', ls='--')
    ax2.errorbar([0, 1],
                 [trials[(trials['probabilityLeft'] == 0.8) & (trials['signed_contrast'] == 0)
                         & (trials['laser_stimulation'] == 0) & (trials['laser_probability'] != 0.75)]['right_choice'].mean(),
                  catch_trials[(catch_trials['probabilityLeft'] == 0.8) & (catch_trials['laser_stimulation'] == 0)]['right_choice'].mean()],
                 [trials[(trials['probabilityLeft'] == 0.8) & (trials['signed_contrast'] == 0)
                         & (trials['laser_stimulation'] == 0) & (trials['laser_probability'] != 0.75)]['right_choice'].sem(),
                  catch_trials[(catch_trials['probabilityLeft'] == 0.8) & (catch_trials['laser_stimulation'] == 0)]['right_choice'].sem()],
                 marker='o', label='No stim', color='b')
    ax2.set(xticks=[0, 1], xticklabels=['Normal trials', 'Catch trials'], title='0% contrast trials')

    sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig(join(fig_path, '%s_opto_behavior_psycurve' % nickname))

# %% Plot
colors = [sns.color_palette('colorblind')[0], sns.color_palette('colorblind')[7]]

psy_avg_block_df = psy_df.groupby(['subject', 'opto_stim']).mean()
psy_avg_block_df['lapse_both'] = psy_avg_block_df.loc[:, 'lapse_l':'lapse_r'].mean(axis=1)
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10), dpi=150)
sns.lineplot(x='opto_stim', y='bias', hue='sert-cre', style='subject', estimator=None,
             data=bias_df[bias_df['catch_trial'] == 0], dashes=False,
             markers=['o']*int(bias_df.shape[0]/4), palette=colors, hue_order=[1, 0],
             legend=False, ax=ax1)
ax1.set(xlabel='', xticks=[0, 1], xticklabels=['No stim', 'Stim'], ylabel='Bias', ylim=[0, 0.7])

sns.lineplot(x='opto_stim', y='bias', hue='sert-cre', style='subject', estimator=None,
             data=bias_df[bias_df['catch_trial'] == 1], dashes=False,
             markers=['o']*int(bias_df.shape[0]/4), palette=colors, hue_order=[1, 0],
             legend=False, ax=ax2)
ax2.set(xlabel='', xticks=[0, 1], xticklabels=['No stim', 'Stim'], ylabel='Bias', ylim=[0, 0.7],
        title='Catch trials')

sns.lineplot(x='opto_stim', y='threshold', hue='sert-cre', style='subject', estimator=None,
             data=psy_df.groupby(['subject', 'opto_stim']).mean(), dashes=False,
             markers=['o']*int(bias_df.shape[0]/4), palette=colors, hue_order=[1, 0],
             legend=False, ax=ax3)
ax3.set(xlabel='', xticks=[0, 1], xticklabels=['No stim', 'Stim'], ylabel='Threshold')

sns.lineplot(x='opto_stim', y='lapse_both', hue='sert-cre', style='subject', estimator=None,
             data=psy_avg_block_df, dashes=False,
             markers=['o']*int(bias_df.shape[0]/4), palette=colors, hue_order=[1, 0],
             legend=False, ax=ax4)
ax4.set(xlabel='', xticks=[0, 1], xticklabels=['No stim', 'Stim'], ylabel='Lapse rate')

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'summary_psycurve'))

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), dpi=150)
delta_lapse_l_l_s = (psy_df.loc[(psy_df['opto_stim'] == 1) & (psy_df['prob_left'] == 0.8)
                                & (psy_df['sert-cre'] == 1), 'lapse_l'].values
                     - psy_df.loc[(psy_df['opto_stim'] == 0) & (psy_df['prob_left'] == 0.8)
                              & (psy_df['sert-cre'] == 1), 'lapse_l'].values)
delta_lapse_l_r_s = (psy_df.loc[(psy_df['opto_stim'] == 1) & (psy_df['prob_left'] == 0.2)
                                & (psy_df['sert-cre'] == 1), 'lapse_l'].values
                     - psy_df.loc[(psy_df['opto_stim'] == 0) & (psy_df['prob_left'] == 0.2)
                              & (psy_df['sert-cre'] == 1), 'lapse_l'].values)
delta_lapse_r_l_s = (psy_df.loc[(psy_df['opto_stim'] == 1) & (psy_df['prob_left'] == 0.8)
                                & (psy_df['sert-cre'] == 1), 'lapse_r'].values
                     - psy_df.loc[(psy_df['opto_stim'] == 0) & (psy_df['prob_left'] == 0.8)
                                  & (psy_df['sert-cre'] == 1), 'lapse_r'].values)
delta_lapse_r_r_s = (psy_df.loc[(psy_df['opto_stim'] == 1) & (psy_df['prob_left'] == 0.2)
                                    & (psy_df['sert-cre'] == 1), 'lapse_r'].values
                     - psy_df.loc[(psy_df['opto_stim'] == 0) & (psy_df['prob_left'] == 0.2)
                                  & (psy_df['sert-cre'] == 1), 'lapse_r'].values)
delta_lapse_l_l_wt = (psy_df.loc[(psy_df['opto_stim'] == 1) & (psy_df['prob_left'] == 0.8)
                                 & (psy_df['sert-cre'] == 0), 'lapse_l'].values
                      - psy_df.loc[(psy_df['opto_stim'] == 0) & (psy_df['prob_left'] == 0.8)
                                  & (psy_df['sert-cre'] == 0), 'lapse_l'].values)
delta_lapse_l_r_wt = (psy_df.loc[(psy_df['opto_stim'] == 1) & (psy_df['prob_left'] == 0.2)
                                 & (psy_df['sert-cre'] == 0), 'lapse_l'].values
                      - psy_df.loc[(psy_df['opto_stim'] == 0) & (psy_df['prob_left'] == 0.2)
                                   & (psy_df['sert-cre'] == 0), 'lapse_l'].values)
delta_lapse_r_l_wt = (psy_df.loc[(psy_df['opto_stim'] == 1) & (psy_df['prob_left'] == 0.8)
                                 & (psy_df['sert-cre'] == 0), 'lapse_r'].values
                      - psy_df.loc[(psy_df['opto_stim'] == 0) & (psy_df['prob_left'] == 0.8)
                                   & (psy_df['sert-cre'] == 0), 'lapse_r'].values)
delta_lapse_r_r_wt = (psy_df.loc[(psy_df['opto_stim'] == 1) & (psy_df['prob_left'] == 0.2)
                                 & (psy_df['sert-cre'] == 0), 'lapse_r'].values
                      - psy_df.loc[(psy_df['opto_stim'] == 0) & (psy_df['prob_left'] == 0.2)
                                   & (psy_df['sert-cre'] == 0), 'lapse_r'].values)

ax1.plot([-.5, 1.5], [0, 0], ls='--', color=[.5, .5, .5], lw=2)
ax1.plot(np.zeros(len(delta_lapse_l_l_s)), delta_lapse_l_l_s, 'o', color=colors[0])
ax1.plot(np.zeros(len(delta_lapse_l_l_wt)), delta_lapse_l_l_wt, 'o', color=colors[1])
ax1.plot(np.ones(len(delta_lapse_r_l_s)), delta_lapse_r_l_s, 'o', color=colors[0])
ax1.plot(np.ones(len(delta_lapse_r_l_wt)), delta_lapse_r_l_wt, 'o', color=colors[1])
ax1.set(xticks=[0, 1], xticklabels=['L', 'R'], ylabel='delta lapse rate \n (stim minus non-stim)',
        ylim=[-.1, .1], title='80:20 blocks')

ax2.plot([-.5, 1.5], [0, 0], ls='--', color=[.5, .5, .5], lw=2)
ax2.plot(np.zeros(len(delta_lapse_l_r_s)), delta_lapse_l_r_s, 'o', color=colors[0])
ax2.plot(np.zeros(len(delta_lapse_l_r_wt)), delta_lapse_l_r_wt, 'o', color=colors[1])
ax2.plot(np.ones(len(delta_lapse_r_r_s)), delta_lapse_r_r_s, 'o', color=colors[0])
ax2.plot(np.ones(len(delta_lapse_r_r_wt)), delta_lapse_r_r_wt, 'o', color=colors[1])
ax2.set(xticks=[0, 1], xticklabels=['L', 'R'], ylabel='delta lapse rate \n (stim minus non-stim)',
        ylim=[-.1, .1], title='20:80 blocks')
plt.tight_layout()






