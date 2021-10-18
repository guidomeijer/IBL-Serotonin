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
                                 load_subjects, fit_psychfunc, query_opto_sessions)
from one.api import ONE
one = ONE()

# Settings
_, fig_path, save_path = paths()
fig_path = join(fig_path, 'Behavior', 'ReactionTimes')
PLOT = True

subjects = load_subjects(behavior=True)
subjects = subjects.reset_index()

results_df = pd.DataFrame()
for i, nickname in enumerate(subjects['subject']):
    print(f'Processing {nickname}')
    eids = query_opto_sessions(nickname, one=one)
    if len(eids) == 0:
        continue

    for j, eid in enumerate(eids):

        # Get trials DataFrame
        trials = pd.DataFrame()
        ses_count = 0
        for j, eid in enumerate(eids):
            try:
                if subjects.loc[i, 'sert-cre'] == 1:
                    these_trials = load_trials(eid, laser_stimulation=True, one=one)
                else:
                    these_trials = load_trials(eid, laser_stimulation=True, patch_old_opto=False, one=one)
                """
                these_trials = load_trials(eid, laser_stimulation=True, one=one)
                """
                these_trials['session'] = ses_count
                trials = trials.append(these_trials, ignore_index=True)
                ses_count = ses_count + 1
            except:
                pass
        if len(trials) == 0:
            continue

    trials['rt_log'] = np.log10(trials['reaction_times'])

    if PLOT:
        colors, dpi = figure_style()
        XTICKS = [0.01, 0.05, 0.5, 5, 10, 60]
        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(4, 4), dpi=dpi)

        ax1.hist(trials.loc[(trials['laser_probability'] == 0.75) & (trials['laser_stimulation'] == 1),
                            'rt_log'], label='Stim', color=colors['stim'], histtype='step', lw=1.5)
        ax1.hist(trials.loc[(trials['laser_probability'] == 0.25) & (trials['laser_stimulation'] == 0),
                            'rt_log'], label='No stim', color=colors['no-stim'], histtype='step', lw=1.5)
        ax1.plot([trials.loc[(trials['laser_probability'] == 0.75) & (trials['laser_stimulation'] == 1), 'rt_log'].median(),
                  trials.loc[(trials['laser_probability'] == 0.75) & (trials['laser_stimulation'] == 1), 'rt_log'].median()],
                 ax1.get_ylim(), ls='--', color=colors['stim'])
        ax1.plot([trials.loc[(trials['laser_probability'] == 0.25) & (trials['laser_stimulation'] == 0), 'rt_log'].median(),
                  trials.loc[(trials['laser_probability'] == 0.25) & (trials['laser_stimulation'] == 0), 'rt_log'].median()],
                 ax1.get_ylim(), ls='--', color=colors['no-stim'])
        ax1.set(title='0% contrast (stim. blocks)', xticks=np.log10(XTICKS), xticklabels=XTICKS, ylabel='Trial count',
                xlabel='log-transformed reaction times (s)', xlim=np.log10([np.min(XTICKS), np.max(XTICKS)]))

        ax2.hist(trials.loc[(trials['laser_probability'] == 0.25) & (trials['laser_stimulation'] == 1),
                            'rt_log'], label='Stim', color=colors['stim'], histtype='step', lw=1.5)
        ax2.hist(trials.loc[(trials['laser_probability'] == 0.75) & (trials['laser_stimulation'] == 0),
                            'rt_log'], label='No stim', color=colors['no-stim'], histtype='step', lw=1.5)
        ax2.plot([trials.loc[(trials['laser_probability'] == 0.25) & (trials['laser_stimulation'] == 1), 'rt_log'].median(),
                  trials.loc[(trials['laser_probability'] == 0.25) & (trials['laser_stimulation'] == 1), 'rt_log'].median()],
                 ax2.get_ylim(), ls='--', color=colors['stim'])
        ax2.plot([trials.loc[(trials['laser_probability'] == 0.75) & (trials['laser_stimulation'] == 0), 'rt_log'].median(),
                  trials.loc[(trials['laser_probability'] == 0.75) & (trials['laser_stimulation'] == 0), 'rt_log'].median()],
                 ax2.get_ylim(), ls='--', color=colors['no-stim'])
        ax2.set(title='0% contrast (probe trials)', xticks=np.log10(XTICKS), xticklabels=XTICKS, ylabel='Trial count',
                xlabel='log-transformed reaction times (s)', xlim=np.log10([np.min(XTICKS), np.max(XTICKS)]))

        ax3.hist(trials.loc[(trials['signed_contrast'].abs() >= 0.25) & (trials['laser_stimulation'] == 1),
                            'rt_log'], label='Stim', color=colors['stim'], histtype='step', lw=2)
        ax3.hist(trials.loc[(trials['signed_contrast'].abs() >= 0.25) & (trials['laser_stimulation'] == 0),
                            'rt_log'], label='No stim', color=colors['no-stim'], histtype='step', lw=1.5)
        ax3.plot([trials.loc[(trials['signed_contrast'].abs() >= 0.25) & (trials['laser_stimulation'] == 1), 'rt_log'].median(),
                  trials.loc[(trials['signed_contrast'].abs() >= 0.25) & (trials['laser_stimulation'] == 1), 'rt_log'].median()],
                 ax3.get_ylim(), ls='--', color=colors['stim'])
        ax3.plot([trials.loc[(trials['signed_contrast'].abs() >= 0.25) & (trials['laser_stimulation'] == 0), 'rt_log'].median(),
                  trials.loc[(trials['signed_contrast'].abs() >= 0.25) & (trials['laser_stimulation'] == 0), 'rt_log'].median()],
                 ax3.get_ylim(), ls='--', color=colors['no-stim'])
        ax3.set(title='High contrast stim', xticks=np.log10(XTICKS), xticklabels=XTICKS, ylabel='Trial count',
                xlabel='log-transformed reaction times (s)', xlim=np.log10([np.min(XTICKS), np.max(XTICKS)]))

        ax4.hist(trials.loc[((trials['signed_contrast'].abs() < 0.25) & ((trials['signed_contrast'].abs() > 0)))
                            & (trials['laser_stimulation'] == 1),
                            'rt_log'], label='Stim', color=colors['stim'], histtype='step', lw=1.5)
        ax4.hist(trials.loc[((trials['signed_contrast'].abs() < 0.25) & ((trials['signed_contrast'].abs() > 0)))
                            & (trials['laser_stimulation'] == 0),
                            'rt_log'], label='Stim', color=colors['no-stim'], histtype='step', lw=1.5)
        ax4.plot([trials.loc[(trials['signed_contrast'].abs() < 0.25) & (trials['laser_stimulation'] == 1), 'rt_log'].median(),
                  trials.loc[(trials['signed_contrast'].abs() < 0.25) & (trials['laser_stimulation'] == 1), 'rt_log'].median()],
                 ax4.get_ylim(), ls='--', color=colors['stim'])
        ax4.plot([trials.loc[(trials['signed_contrast'].abs() < 0.25) & (trials['laser_stimulation'] == 0), 'rt_log'].median(),
                  trials.loc[(trials['signed_contrast'].abs() < 0.25) & (trials['laser_stimulation'] == 0), 'rt_log'].median()],
                 ax4.get_ylim(), ls='--', color=colors['no-stim'])
        ax4.set(title='Low contrast stim', xticks=np.log10(XTICKS), xticklabels=XTICKS, ylabel='Trial count',
                xlabel='log-transformed reaction times (s)', xlim=np.log10([np.min(XTICKS), np.max(XTICKS)]))

        plt.tight_layout()
        sns.despine(trim=True)
        plt.savefig(join(fig_path, f'log_rt_distribution_{nickname}.png'))
        plt.savefig(join(fig_path, f'log_rt_distribution_{nickname}.pdf'))

        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(4, 4), dpi=dpi)

        ax1.hist(trials.loc[(trials['laser_probability'] == 0.75) & (trials['laser_stimulation'] == 1),
                            'reaction_times'], label='Stim', color=colors['stim'], histtype='step', lw=1.5, bins=200)
        ax1.hist(trials.loc[(trials['laser_probability'] == 0.25) & (trials['laser_stimulation'] == 0),
                            'reaction_times'], label='No stim', color=colors['no-stim'], histtype='step', lw=1.5, bins=200)
        ax1.plot([trials.loc[(trials['laser_probability'] == 0.75) & (trials['laser_stimulation'] == 1), 'reaction_times'].median(),
                  trials.loc[(trials['laser_probability'] == 0.75) & (trials['laser_stimulation'] == 1), 'reaction_times'].median()],
                 ax1.get_ylim(), ls='--', color=colors['stim'])
        ax1.plot([trials.loc[(trials['laser_probability'] == 0.25) & (trials['laser_stimulation'] == 0), 'reaction_times'].median(),
                  trials.loc[(trials['laser_probability'] == 0.25) & (trials['laser_stimulation'] == 0), 'reaction_times'].median()],
                 ax1.get_ylim(), ls='--', color=colors['no-stim'])
        ax1.set(title='0% contrast (stim. blocks)', ylabel='Trial count', xlabel='Reaction times (s)', xlim=[-0.5, 1])

        ax2.hist(trials.loc[(trials['laser_probability'] == 0.25) & (trials['laser_stimulation'] == 1),
                            'reaction_times'], label='Stim', color=colors['stim'], histtype='step', lw=1.5, bins=200)
        ax2.hist(trials.loc[(trials['laser_probability'] == 0.75) & (trials['laser_stimulation'] == 0),
                            'reaction_times'], label='No stim', color=colors['no-stim'], histtype='step', lw=1.5, bins=200)
        ax2.plot([trials.loc[(trials['laser_probability'] == 0.25) & (trials['laser_stimulation'] == 1), 'reaction_times'].median(),
                  trials.loc[(trials['laser_probability'] == 0.25) & (trials['laser_stimulation'] == 1), 'reaction_times'].median()],
                 ax2.get_ylim(), ls='--', color=colors['stim'])
        ax2.plot([trials.loc[(trials['laser_probability'] == 0.75) & (trials['laser_stimulation'] == 0), 'reaction_times'].median(),
                  trials.loc[(trials['laser_probability'] == 0.75) & (trials['laser_stimulation'] == 0), 'reaction_times'].median()],
                 ax2.get_ylim(), ls='--', color=colors['no-stim'])
        ax2.set(title='0% contrast (probe trials)', ylabel='Trial count', xlabel='Reaction times (s)', xlim=[-0.5, 1])

        ax3.hist(trials.loc[(trials['signed_contrast'].abs() >= 0.25) & (trials['laser_stimulation'] == 1),
                            'reaction_times'], label='Stim', color=colors['stim'], histtype='step', lw=1.5, bins=200)
        ax3.hist(trials.loc[(trials['signed_contrast'].abs() >= 0.25) & (trials['laser_stimulation'] == 0),
                            'reaction_times'], label='No stim', color=colors['no-stim'], histtype='step', lw=1.5, bins=200)
        ax3.plot([trials.loc[(trials['signed_contrast'].abs() >= 0.25) & (trials['laser_stimulation'] == 1), 'reaction_times'].median(),
                  trials.loc[(trials['signed_contrast'].abs() >= 0.25) & (trials['laser_stimulation'] == 1), 'reaction_times'].median()],
                 ax3.get_ylim(), ls='--', color=colors['stim'])
        ax3.plot([trials.loc[(trials['signed_contrast'].abs() >= 0.25) & (trials['laser_stimulation'] == 0), 'reaction_times'].median(),
                  trials.loc[(trials['signed_contrast'].abs() >= 0.25) & (trials['laser_stimulation'] == 0), 'reaction_times'].median()],
                 ax3.get_ylim(), ls='--', color=colors['no-stim'])
        ax3.set(title='High contrast stim', ylabel='Trial count', xlabel='Reaction times (s)', xlim=[-0.5, 1])

        ax4.hist(trials.loc[((trials['signed_contrast'].abs() < 0.25) & ((trials['signed_contrast'].abs() > 0)))
                            & (trials['laser_stimulation'] == 1),
                            'reaction_times'], label='Stim', color=colors['stim'], histtype='step', lw=1.5, bins=200)
        ax4.hist(trials.loc[((trials['signed_contrast'].abs() < 0.25) & ((trials['signed_contrast'].abs() > 0)))
                            & (trials['laser_stimulation'] == 0),
                            'reaction_times'], label='Stim', color=colors['no-stim'], histtype='step', lw=1.5, bins=200)
        ax4.plot([trials.loc[(trials['signed_contrast'].abs() < 0.25) & (trials['laser_stimulation'] == 1), 'reaction_times'].median(),
                  trials.loc[(trials['signed_contrast'].abs() < 0.25) & (trials['laser_stimulation'] == 1), 'reaction_times'].median()],
                 ax4.get_ylim(), ls='--', color=colors['stim'])
        ax4.plot([trials.loc[(trials['signed_contrast'].abs() < 0.25) & (trials['laser_stimulation'] == 0), 'reaction_times'].median(),
                  trials.loc[(trials['signed_contrast'].abs() < 0.25) & (trials['laser_stimulation'] == 0), 'reaction_times'].median()],
                 ax4.get_ylim(), ls='--', color=colors['no-stim'])
        ax4.set(title='Low contrast stim', ylabel='Trial count', xlabel='Reaction times (s)', xlim=[-0.5, 1])

        plt.tight_layout()
        sns.despine(trim=True)
        plt.savefig(join(fig_path, f'rt_distribution_{nickname}.png'))
        plt.savefig(join(fig_path, f'rt_distribution_{nickname}.pdf'))

        """
        results_df = results_df.append(pd.DataFrame(index=[results_df.shape[0] + 1], data={
            'bias': bias, 'stim': int('opto' in details['task_protocol']), 'date': details['date'],
            'sert-cre': subjects.loc[i, 'sert-cre'], 'subject': nickname}))
        """


