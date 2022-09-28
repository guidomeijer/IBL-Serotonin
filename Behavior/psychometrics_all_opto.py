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
PLOT_SINGLE_ANIMALS = True
fig_path, _ = paths()
fig_path = join(fig_path, 'Behavior', 'Psychometrics')
subjects = load_subjects()

bias_df, lapse_df, psy_df, ses_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
for i, nickname in enumerate(subjects['subject']):

    # Query sessions
    eids = query_opto_sessions(nickname, include_ephys=True, one=one)

    # Apply behavioral criterion
    eids = behavioral_criterion(eids, one=one)
    if len(eids) == 0:
        continue

    # Get trials DataFrame
    trials = pd.DataFrame()
    ses_count = 0
    for j, eid in enumerate(eids):
        try:
            these_trials = load_trials(eid, laser_stimulation=True, patch_old_opto=False, one=one)
        except:
            continue
        these_trials['session'] = ses_count
        trials = pd.concat((trials, these_trials), ignore_index=True)
        ses_count = ses_count + 1

    ses_df = pd.concat((ses_df, pd.DataFrame(index=[ses_df.shape[0]+1], data={
        'subject': nickname, 'n_sessions': ses_count, 'n_trials': trials.shape[0],
        'sert-cre': subjects.loc[i, 'sert-cre']})))

    if len(trials) == 0:
        continue

    # Get bias from fitted curves
    bias_fit_stim = get_bias(trials.loc[(trials['laser_stimulation'] == 1) & (trials['probe_trial'] == 0)])
    bias_fit_no_stim = get_bias(trials.loc[(trials['laser_stimulation'] == 0) & (trials['probe_trial'] == 0)])

    # Get bias strength 0% contrast trials
    bias_no_stim = np.abs(trials[(trials['probabilityLeft'] == 0.8)
                                 & (trials['laser_stimulation'] == 0)
                                 & (trials['laser_probability'] == 0.25)].mean()
                          - trials[(trials['probabilityLeft'] == 0.2)
                                   & (trials['laser_stimulation'] == 0)
                                   & (trials['laser_probability'] == 0.25)].mean())['right_choice']
    bias_stim = np.abs(trials[(trials['probabilityLeft'] == 0.8)
                              & (trials['laser_stimulation'] == 1)
                              & (trials['laser_probability'] == 0.75)].mean()
                       - trials[(trials['probabilityLeft'] == 0.2)
                                & (trials['laser_stimulation'] == 1)
                                & (trials['laser_probability'] == 0.75)].mean())['right_choice']

    bias_probe_stim = np.abs(trials[(trials['probabilityLeft'] == 0.8)
                                    & (trials['laser_stimulation'] == 1)
                                    & (trials['laser_probability'] == 0.25)].mean()
                             - trials[(trials['probabilityLeft'] == 0.2)
                                      & (trials['laser_stimulation'] == 1)
                                      & (trials['laser_probability'] == 0.25)].mean())['right_choice']
    bias_probe_no_stim = np.abs(trials[(trials['probabilityLeft'] == 0.8)
                                       & (trials['laser_stimulation'] == 0)
                                       & (trials['laser_probability'] == 0.75)].mean()
                                - trials[(trials['probabilityLeft'] == 0.2)
                                         & (trials['laser_stimulation'] == 0)
                                         & (trials['laser_probability'] == 0.75)].mean())['right_choice']

    # Get side bias 0% contrast trials
    side_bias_no_stim = 0.5 - (np.sum(trials.loc[(trials['laser_probability'] == 0.25)
                                                 & (trials['laser_stimulation'] == 0), 'right_choice'] == 1)
                               / np.shape(trials.loc[(trials['laser_probability'] == 0.25)
                                                     & (trials['laser_stimulation'] == 0), 'right_choice'] == 1)[0])
    side_bias_stim = 0.5 - (np.sum(trials.loc[(trials['laser_probability'] == 0.75)
                                              & (trials['laser_stimulation'] == 1), 'right_choice'] == 1)
                            / np.shape(trials.loc[(trials['laser_probability'] == 0.75)
                                                  & (trials['laser_stimulation'] == 1), 'right_choice'] == 1)[0])
    side_bias_probe_stim = 0.5 - (np.sum(trials.loc[(trials['laser_probability'] == 0.25)
                                                 & (trials['laser_stimulation'] == 1), 'right_choice'] == 1)
                               / np.shape(trials.loc[(trials['laser_probability'] == 0.25)
                                                     & (trials['laser_stimulation'] == 1), 'right_choice'] == 1)[0])
    side_bias_probe_no_stim = 0.5 - (np.sum(trials.loc[(trials['laser_probability'] == 0.75)
                                                       & (trials['laser_stimulation'] == 0), 'right_choice'] == 1)
                                     / np.shape(trials.loc[(trials['laser_probability'] == 0.75)
                                                           & (trials['laser_stimulation'] == 0), 'right_choice'] == 1)[0])

    # Get performance
    perf_opto = (trials.loc[trials['laser_stimulation'] == 1, 'correct'].sum()
                 / trials.loc[trials['laser_stimulation'] == 1, 'correct'].shape[0]) * 100
    perf_no_opto = (trials.loc[trials['laser_stimulation'] == 0, 'correct'].sum()
                    / trials.loc[trials['laser_stimulation'] == 0, 'correct'].shape[0]) * 100

    # Get RT
    rt_no_stim = trials[(trials['laser_stimulation'] == 0)
                        & (trials['laser_probability'] == 0.25)].median()['reaction_times']
    rt_stim = trials[(trials['laser_stimulation'] == 1)
                     & (trials['laser_probability'] == 0.75)].median()['reaction_times']
    rt_catch_no_stim = trials[(trials['laser_stimulation'] == 0)
                              & (trials['laser_probability'] == 0.25)].median()['reaction_times']
    rt_catch_stim = trials[(trials['laser_stimulation'] == 1)
                           & (trials['laser_probability'] == 0.25)].median()['reaction_times']
    bias_df = bias_df.append(pd.DataFrame(index=[bias_df.shape[0] + 1], data={
        'subject': nickname, 'sert-cre': subjects.loc[i, 'sert-cre'],
        'bias_no_stim': bias_no_stim, 'bias_stim': bias_stim, 'bias_probe_stim': bias_probe_stim,
        'bias_probe_no_stim': bias_probe_no_stim, 'side_bias_no_stim': side_bias_no_stim,
        'side_bias_stim': side_bias_stim, 'side_bias_probe_stim': side_bias_probe_stim,
        'side_bias_probe_no_stim': side_bias_probe_no_stim, 'rt_no_stim': rt_no_stim,
        'rt_stim': rt_stim, 'rt_catch_no_stim': rt_catch_no_stim, 'rt_catch_stim': rt_catch_stim,
        'bias_fit_stim': bias_fit_stim, 'bias_fit_no_stim': bias_fit_no_stim,
        'perf_stim': perf_opto, 'perf_no_stim': perf_no_opto}))

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
        'expression': subjects.loc[i, 'expression'],
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
        'subject': nickname, 'sert-cre': subjects.loc[i, 'sert-cre'],
        'expression': subjects.loc[i, 'expression'],
        'opto_stim': 1, 'prob_left': 0.2,
        'bias': pars[0], 'threshold': pars[1], 'lapse_l': pars[2], 'lapse_r': pars[3]}))

    # Plot
    if PLOT_SINGLE_ANIMALS:
        colors, dpi = figure_style()
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 2), dpi=dpi, sharey=True)

        # plot_psychometric(trials[trials['probabilityLeft'] == 0.5], ax=ax1, color='k')
        plot_psychometric(trials[(trials['probabilityLeft'] == 0.8)
                                 & (trials['laser_stimulation'] == 0)
                                 & (trials['laser_probability'] != 0.75)], ax=ax1, color=colors['left'])
        plot_psychometric(trials[(trials['probabilityLeft'] == 0.8)
                                 & (trials['laser_stimulation'] == 1)
                                 & (trials['laser_probability'] != 0.25)], ax=ax1,
                          color=colors['left'], linestyle='--')
        plot_psychometric(trials[(trials['probabilityLeft'] == 0.2)
                                 & (trials['laser_stimulation'] == 0)
                                 & (trials['laser_probability'] != 0.75)], ax=ax1, color=colors['right'])
        plot_psychometric(trials[(trials['probabilityLeft'] == 0.2)
                                 & (trials['laser_stimulation'] == 1)
                                 & (trials['laser_probability'] != 0.25)], ax=ax1,
                          color=colors['right'], linestyle='--')
        #ax1.text(-20, 0.75, '80% right', color=colors['right'])
        #ax1.text(20, 0.25, '80% left', color=colors['left'])
        ax1.set(title=f'{nickname}, SERT: {subjects.loc[i, "sert-cre"]}')

        catch_trials = trials[((trials['laser_probability'] == 0.75) & (trials['laser_stimulation'] == 0))
                              | ((trials['laser_probability'] == 0.25) & (trials['laser_stimulation'] == 1))]

        ax2.errorbar([0, 1],
                     [trials[(trials['probabilityLeft'] == 0.2) & (trials['signed_contrast'] == 0)
                             & (trials['laser_stimulation'] == 1) & (trials['laser_probability'] != 0.25)]['right_choice'].mean(),
                      catch_trials[(catch_trials['probabilityLeft'] == 0.2) & (catch_trials['laser_stimulation'] == 1)]['right_choice'].mean()],
                     [trials[(trials['probabilityLeft'] == 0.2) & (trials['signed_contrast'] == 0)
                             & (trials['laser_stimulation'] == 1) & (trials['laser_probability'] != 0.25)]['right_choice'].sem(),
                      catch_trials[(catch_trials['probabilityLeft'] == 0.2) & (catch_trials['laser_stimulation'] == 1)]['right_choice'].sem()],
                     marker='o', label='Stim', color=colors['right'], ls='--')
        ax2.errorbar([0, 1],
                     [trials[(trials['probabilityLeft'] == 0.2) & (trials['signed_contrast'] == 0)
                             & (trials['laser_stimulation'] == 0) & (trials['laser_probability'] != 0.75)]['right_choice'].mean(),
                      catch_trials[(catch_trials['probabilityLeft'] == 0.2) & (catch_trials['laser_stimulation'] == 0)]['right_choice'].mean()],
                     [trials[(trials['probabilityLeft'] == 0.2) & (trials['signed_contrast'] == 0)
                             & (trials['laser_stimulation'] == 0) & (trials['laser_probability'] != 0.75)]['right_choice'].sem(),
                      catch_trials[(catch_trials['probabilityLeft'] == 0.2) & (catch_trials['laser_stimulation'] == 0)]['right_choice'].sem()],
                     marker='o', label='No stim', color=colors['right'])
        ax2.errorbar([0, 1],
                     [trials[(trials['probabilityLeft'] == 0.8) & (trials['signed_contrast'] == 0)
                             & (trials['laser_stimulation'] == 1) & (trials['laser_probability'] != 0.25)]['right_choice'].mean(),
                      catch_trials[(catch_trials['probabilityLeft'] == 0.8) & (catch_trials['laser_stimulation'] == 1)]['right_choice'].mean()],
                     [trials[(trials['probabilityLeft'] == 0.8) & (trials['signed_contrast'] == 0)
                             & (trials['laser_stimulation'] == 1) & (trials['laser_probability'] != 0.25)]['right_choice'].sem(),
                      catch_trials[(catch_trials['probabilityLeft'] == 0.8) & (catch_trials['laser_stimulation'] == 1)]['right_choice'].sem()],
                     marker='o', label='Stim', color=colors['left'], ls='--')
        ax2.errorbar([0, 1],
                     [trials[(trials['probabilityLeft'] == 0.8) & (trials['signed_contrast'] == 0)
                             & (trials['laser_stimulation'] == 0) & (trials['laser_probability'] != 0.75)]['right_choice'].mean(),
                      catch_trials[(catch_trials['probabilityLeft'] == 0.8) & (catch_trials['laser_stimulation'] == 0)]['right_choice'].mean()],
                     [trials[(trials['probabilityLeft'] == 0.8) & (trials['signed_contrast'] == 0)
                             & (trials['laser_stimulation'] == 0) & (trials['laser_probability'] != 0.75)]['right_choice'].sem(),
                      catch_trials[(catch_trials['probabilityLeft'] == 0.8) & (catch_trials['laser_stimulation'] == 0)]['right_choice'].sem()],
                     marker='o', label='No stim', color=colors['left'])
        ax2.set(xticks=[0, 1], xticklabels=['Normal trials', 'Catch trials'], title='0% contrast trials')

        sns.despine(trim=True)
        plt.tight_layout()

        plt.savefig(join(fig_path, '%s_opto_behavior_psycurve.jpg' % nickname), dpi=600)

# %% Plot

psy_avg_block_df = psy_df.groupby(['subject', 'opto_stim']).mean()
psy_avg_block_df['lapse_both'] = psy_avg_block_df.loc[:, 'lapse_l':'lapse_r'].mean(axis=1)
psy_avg_block_df = psy_avg_block_df.reset_index()
colors, dpi = figure_style()
colors = [colors['wt'], colors['sert']]
f, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, figsize=(6.5, 3.5), dpi=dpi)
for i, subject in enumerate(bias_df['subject']):
    sert_cre = bias_df.loc[bias_df['subject'] == subject, 'sert-cre'].unique()[0]
    if sert_cre == 1:
        label = 'SERT'
    else:
        label = 'WT'
    ax1.plot([1, 2], [bias_df.loc[bias_df['subject'] == subject, 'bias_fit_no_stim'],
                      bias_df.loc[bias_df['subject'] == subject, 'bias_fit_stim']],
             color = colors[sert_cre], label=label, marker='o', ms=2)
ax1.set(xlabel='', xticks=[1, 2], xticklabels=['Non-stimulated\ntrials', 'Stimulated\ntrials'],
        ylabel='Bias', ylim=[-0.1, 0.7])
#ax1.legend(frameon=False)


for i, subject in enumerate(bias_df['subject']):
    ax2.plot([1, 2], [bias_df.loc[bias_df['subject'] == subject, 'bias_probe_no_stim'],
                      bias_df.loc[bias_df['subject'] == subject, 'bias_probe_stim']],
             color = colors[bias_df.loc[bias_df['subject'] == subject, 'sert-cre'].values[0]], marker='o', ms=2)
ax2.set(xlabel='', xticks=[1, 2], xticklabels=['Non-stimulated\ntrials', 'Stimulated\ntrials'],
        ylabel='Total bias', ylim=[-0.1, 0.7],
        title='Probe trials')

for i, subject in enumerate(bias_df['subject']):
    ax3.plot([1, 2], [bias_df.loc[bias_df['subject'] == subject, 'side_bias_no_stim'],
                      bias_df.loc[bias_df['subject'] == subject, 'side_bias_stim']],
             color = colors[bias_df.loc[bias_df['subject'] == subject, 'sert-cre'].values[0]], marker='o', ms=2)
ax3.set(xlabel='', xticks=[1, 2], xticklabels=['Non-stimulated\ntrials', 'Stimulated\ntrials'],
        ylabel='Side bias', ylim=[-0.25, 0.25])

for i, subject in enumerate(bias_df['subject']):
    ax4.plot([1, 2], [bias_df.loc[bias_df['subject'] == subject, 'side_bias_probe_no_stim'],
                      bias_df.loc[bias_df['subject'] == subject, 'side_bias_probe_stim']],
             color = colors[bias_df.loc[bias_df['subject'] == subject, 'sert-cre'].values[0]], marker='o', ms=2)
ax4.set(xlabel='', xticks=[1, 2], xticklabels=['Non-stimulated\ntrials', 'Stimulated\ntrials'],
        ylabel='Side bias', title='Probe trials', ylim=[-0.25, 0.25])

"""
delta_block = (bias_df.loc[(bias_df['opto_stim'] == 1) & (bias_df['catch_trial'] == 0), 'bias'].values -
               bias_df.loc[(bias_df['opto_stim'] == 0) & (bias_df['catch_trial'] == 0), 'bias'].values)
delta_probe = (bias_df.loc[(bias_df['opto_stim'] == 1) & (bias_df['catch_trial'] == 1), 'bias'].values -
               bias_df.loc[(bias_df['opto_stim'] == 0) & (bias_df['catch_trial'] == 1), 'bias'].values)
sert_cre = bias_df.loc[(bias_df['opto_stim'] == 1) & (bias_df['catch_trial'] == 0), 'sert-cre'].values
ax3.plot([0, 0], [-0.2, 0.2], ls='--', color='gray')
ax3.plot([-0.2, 0.2], [0, 0], ls='--', color='gray')
ax3.scatter(delta_block[sert_cre == 1], delta_probe[sert_cre ==1], color=colors[0])
ax3.scatter(delta_block[sert_cre == 0], delta_probe[sert_cre ==0], color=colors[1])
ax3.set(xlim=(-0.2, 0.2), ylim=(-0.2, 0.2), xlabel='Bias change block trials', ylabel='Bias change probe trials')
"""

for i, subject in enumerate(psy_avg_block_df['subject']):
    ax6.plot([1, 2], psy_avg_block_df.loc[(psy_avg_block_df['subject'] == subject), 'threshold'],
             color = colors[int(psy_avg_block_df.loc[psy_avg_block_df['subject'] == subject, 'expression'].unique()[1])],
             marker='o', ms=2)
ax5.set(xlabel='', xticks=[0, 1], xticklabels=['No stim', 'Stim'], ylabel='Threshold')

for i, subject in enumerate(psy_avg_block_df['subject']):
    ax6.plot([1, 2], psy_avg_block_df.loc[(psy_avg_block_df['subject'] == subject), 'lapse_both'],
             color = colors[int(psy_avg_block_df.loc[psy_avg_block_df['subject'] == subject, 'expression'].unique()[1])],
             marker='o', ms=2)
ax6.set(xlabel='', xticks=[1, 2], xticklabels=['No stim', 'Stim'], ylabel='Lapse rate')

for i, subject in enumerate(bias_df['subject']):
    ax7.plot([1, 2], [bias_df.loc[bias_df['subject'] == subject, 'rt_no_stim'],
                      bias_df.loc[bias_df['subject'] == subject, 'rt_stim']],
             color = colors[bias_df.loc[bias_df['subject'] == subject, 'sert-cre'].values[0]], marker='o', ms=2)
ax7.set(xlabel='', xticks=[1, 2], xticklabels=['No stim', 'Stim'], ylabel='Median reaction time',
        yscale='log', yticks=[1], yticklabels=[1])

for i, subject in enumerate(bias_df['subject']):
    ax8.plot([1, 2], [bias_df.loc[bias_df['subject'] == subject, 'perf_no_stim'],
                      bias_df.loc[bias_df['subject'] == subject, 'perf_stim']],
             color = colors[bias_df.loc[bias_df['subject'] == subject, 'sert-cre'].values[0]], marker='o', ms=2)
ax8.set(xlabel='', xticks=[1, 2], xticklabels=['No stim', 'Stim'], ylabel='Performance (%)',
        ylim=[65, 85])

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'summary_psycurve.jpg'))

# %%
"""
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5), dpi=dpi)
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
"""