#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 11:42:01 2021

@author: guido
"""

import numpy as np
from os.path import join
import pandas as pd
from dlc_functions import get_dlc_XYs, get_raw_and_smooth_pupil_dia
import matplotlib.pyplot as plt
from scipy.stats import zscore
import seaborn as sns
from serotonin_functions import (load_trials, paths, query_opto_sessions, figure_style, load_subjects,
                                 behavioral_criterion)
from one.api import ONE
one = ONE()

# Settings
PLOT_EXAMPLE_TRACE = False
BEHAVIOR_CRIT = False
TIME_BINS = np.arange(-1, 3.2, 0.2)
BIN_SIZE = 0.2  # seconds
BASELINE = [1, 0]  # seconds
N_TRIALS = 20
TEST_TIME = [1.8, 2.2]
fig_path, _ = paths()
fig_path = join(fig_path, 'Pupil')

subjects = load_subjects()

# TESTING
#subjects = subjects[subjects['subject'] != 'ZFM-02181'].reset_index(drop=True)

results_df = pd.DataFrame()
results_df_baseline = pd.DataFrame()
for i, nickname in enumerate(subjects['subject']):
    print(f'Processing {nickname}..')

    # Query sessions
    eids = query_opto_sessions(nickname, include_ephys=True, one=one)

    if BEHAVIOR_CRIT:
        eids = behavioral_criterion(eids)

    # Loop over sessions
    pupil_size = pd.DataFrame()
    for j, eid in enumerate(eids):
        print(f'Processing session {j+1} of {len(eids)}')

        # Load in trials and video data
        try:
            trials = load_trials(eid, laser_stimulation=True, one=one)
        except:
            print('could not load trials')
            continue
        if trials is None:
            continue
        if 'laser_stimulation' not in trials.columns.values:
            continue

        details = one.get_details(eid)
        date = details['date']

        # Get stimulation block transitions
        no_probe_trials = trials.copy()
        no_probe_trials.loc[no_probe_trials['probe_trial'] == 1, 'laser_stimulation'] = (
            no_probe_trials.loc[no_probe_trials['probe_trial'] == 1, 'laser_probability'].round())
        stim_block_trans = np.append([0], np.array(np.where(np.diff(no_probe_trials['laser_stimulation']) != 0)) + 1)

        # Divide up stim blocks into early and late
        for k, trans in enumerate(stim_block_trans):
            trials.loc[trans:trans+N_TRIALS, 'block_progress'] = 1
            trials.loc[trans+N_TRIALS+1:trans+N_TRIALS*2+1, 'block_progress'] = 2
            if k != stim_block_trans.shape[0]-1:
                trials.loc[trans+N_TRIALS*2+1:stim_block_trans[k+1], 'block_progress'] = 3
            else:
                trials.loc[trans+N_TRIALS*2+1:-1, 'block_progress'] = 3

        # Load in camera timestamps and DLC output
        try:
            video_times, XYs = get_dlc_XYs(one, eid)
        except:
            print('Could not load video and/or DLC data')
            continue

        # If the difference between timestamps and video frames is too large, skip
        if np.abs(video_times.shape[0] - XYs['pupil_left_r'].shape[0]) > 10000:
            print('Timestamp mismatch, skipping..')
            continue

        # Get pupil diameter
        raw_diameter, diameter = get_raw_and_smooth_pupil_dia(eid, 'left', one)

        # Assume frames were dropped at the end
        if video_times.shape[0] > diameter.shape[0]:
            video_times = video_times[:diameter.shape[0]]
        elif diameter.shape[0] > video_times.shape[0]:
            diameter = diameter[:video_times.shape[0]]

        # Calculate percentage change
        diameter_perc = ((diameter - np.percentile(diameter[~np.isnan(diameter)], 2))
                         / np.percentile(diameter[~np.isnan(diameter)], 2)) * 100

        # Get trial triggered baseline subtracted pupil diameter
        for t, trial_start in enumerate(trials['stimOn_times']):
            this_diameter = np.array([np.nan] * TIME_BINS.shape[0])
            baseline_subtracted = np.array([np.nan] * TIME_BINS.shape[0])
            baseline = np.nanmedian(diameter_perc[(video_times > (trial_start - BASELINE[0]))
                                                  & (video_times < (trial_start - BASELINE[1]))])
            for b, time_bin in enumerate(TIME_BINS):
                this_diameter[b] = np.nanmedian(diameter_perc[
                    (video_times > (trial_start + time_bin) - (BIN_SIZE / 2))
                    & (video_times < (trial_start + time_bin) + (BIN_SIZE / 2))])
                baseline_subtracted[b] = np.nanmedian(diameter_perc[
                    (video_times > (trial_start + time_bin) - (BIN_SIZE / 2))
                    & (video_times < (trial_start + time_bin) + (BIN_SIZE / 2))]) - baseline
            pupil_size = pd.concat((pupil_size, pd.DataFrame(data={
                'diameter': this_diameter, 'baseline_subtracted': baseline_subtracted, 'eid': eid,
                'subject': nickname, 'trial': t, 'contrast': trials.loc[t, 'signed_contrast'],
                'sert': subjects.loc[i, 'sert-cre'], 'laser': trials.loc[t, 'laser_stimulation'],
                'feedback_type': trials.loc[t, 'feedbackType'],
                'laser_prob': trials.loc[t, 'laser_probability'], 'laser_block_progress': trials.loc[t, 'block_progress'],
                'time': TIME_BINS})))


        if PLOT_EXAMPLE_TRACE:
            colors, dpi = figure_style()
            f, ax1 = plt.subplots(1, 1, figsize=(3,2), dpi=dpi)
            ax1.plot(video_times[1000:3000], diameter_perc[1000:3000])
            ax1.set(xlabel='Time (s)', ylabel='Pupil size (%)', title=f'{nickname}, {date}')
            plt.tight_layout()
            sns.despine(trim=False)
            plt.savefig(join(fig_path, 'ExampleTraces', f'{nickname}_{date}.png'))
            plt.savefig(join(fig_path, 'ExampleTraces', f'{nickname}_{date}.pdf'))
            plt.close(f)

    if pupil_size.shape[0] < 100:
        continue

    # Add to overal dataframe
    pupil_size = pupil_size.reset_index(drop=True)
    pupil_size['abs_contrast'] = pupil_size['contrast'].abs()
    pupil_size['abs_log_contrast'] = pupil_size['abs_contrast'].copy()
    pupil_size.loc[pupil_size['abs_log_contrast'] == 0, 'abs_log_contrast'] = 0.01
    pupil_size['abs_log_contrast'] = np.log10(pupil_size['abs_log_contrast'])
    results_df = pd.concat((results_df, pd.DataFrame(data={
        'stim_all': pupil_size[pupil_size['laser'] == 1].groupby('time').median()['diameter'],
        'no_stim_all': pupil_size[pupil_size['laser'] == 0].groupby('time').median()['diameter'],
        'stim_blank': pupil_size[(pupil_size['contrast'] == 0) & (pupil_size['laser'] == 1)].groupby('time').median()['diameter'],
        'no_stim_blank': pupil_size[(pupil_size['contrast'] == 0) & (pupil_size['laser'] == 0)].groupby('time').median()['diameter'],
        'stim_blank_error': pupil_size[(pupil_size['contrast'] == 0) & (pupil_size['laser'] == 1) & (pupil_size['feedback_type'] == -1)].groupby('time').median()['diameter'],
        'no_stim_blank_error': pupil_size[(pupil_size['contrast'] == 0) & (pupil_size['laser'] == 0) & (pupil_size['feedback_type'] == -1)].groupby('time').median()['diameter'],
        'stim_block': pupil_size[(pupil_size['laser_prob'] == 0.75) & (pupil_size['laser'] == 1)].groupby('time').median()['diameter'],
        'no_stim_block': pupil_size[(pupil_size['laser_prob'] == 0.25) & (pupil_size['laser'] == 0)].groupby('time').median()['diameter'],
        'stim_probe': pupil_size[(pupil_size['laser_prob'] == 0.25) & (pupil_size['laser'] == 1)].groupby('time').median()['diameter'],
        'no_stim_probe': pupil_size[(pupil_size['laser_prob'] == 0.75) & (pupil_size['laser'] == 0)].groupby('time').median()['diameter'],
        'stim_early_block': pupil_size[(pupil_size['laser_block_progress'] == 1) & (pupil_size['laser'] == 1)].groupby('time').median()['diameter'],
        'no_stim_early_block': pupil_size[(pupil_size['laser_block_progress'] == 1) & (pupil_size['laser'] == 0)].groupby('time').median()['diameter'],
        'stim_middle_block': pupil_size[(pupil_size['laser_block_progress'] == 2) & (pupil_size['laser'] == 1)].groupby('time').median()['diameter'],
        'no_stim_middle_block': pupil_size[(pupil_size['laser_block_progress'] == 2) & (pupil_size['laser'] == 0)].groupby('time').median()['diameter'],
        'stim_late_block': pupil_size[(pupil_size['laser_block_progress'] == 3) & (pupil_size['laser'] == 1)].groupby('time').median()['diameter'],
        'no_stim_late_block': pupil_size[(pupil_size['laser_block_progress'] == 3) & (pupil_size['laser'] == 0)].groupby('time').median()['diameter'],
        'subject': nickname, 'sert-cre': subjects.loc[i, 'sert-cre']})))
    results_df_baseline = pd.concat((results_df_baseline, pd.DataFrame(data={
        'stim_all': pupil_size[pupil_size['laser'] == 1].groupby('time').median()['baseline_subtracted'],
        'no_stim_all': pupil_size[pupil_size['laser'] == 0].groupby('time').median()['baseline_subtracted'],
        'stim_blank': pupil_size[(pupil_size['contrast'] == 0) & (pupil_size['laser'] == 1)].groupby('time').median()['baseline_subtracted'],
        'no_stim_blank': pupil_size[(pupil_size['contrast'] == 0) & (pupil_size['laser'] == 0)].groupby('time').median()['baseline_subtracted'],
        'stim_blank_error': pupil_size[(pupil_size['contrast'] == 0) & (pupil_size['laser'] == 1) & (pupil_size['feedback_type'] == -1)].groupby('time').median()['baseline_subtracted'],
        'no_stim_blank_error': pupil_size[(pupil_size['contrast'] == 0) & (pupil_size['laser'] == 0) & (pupil_size['feedback_type'] == -1)].groupby('time').median()['baseline_subtracted'],
        'stim_block': pupil_size[(pupil_size['laser_prob'] == 0.75) & (pupil_size['laser'] == 1)].groupby('time').median()['baseline_subtracted'],
        'no_stim_block': pupil_size[(pupil_size['laser_prob'] == 0.25) & (pupil_size['laser'] == 0)].groupby('time').median()['baseline_subtracted'],
        'stim_probe': pupil_size[(pupil_size['laser_prob'] == 0.25) & (pupil_size['laser'] == 1)].groupby('time').median()['baseline_subtracted'],
        'no_stim_probe': pupil_size[(pupil_size['laser_prob'] == 0.75) & (pupil_size['laser'] == 0)].groupby('time').median()['baseline_subtracted'],
        'stim_early_block': pupil_size[(pupil_size['laser_block_progress'] == 1) & (pupil_size['laser'] == 1)].groupby('time').median()['baseline_subtracted'],
        'no_stim_early_block': pupil_size[(pupil_size['laser_block_progress'] == 1) & (pupil_size['laser'] == 0)].groupby('time').median()['baseline_subtracted'],
        'stim_middle_block': pupil_size[(pupil_size['laser_block_progress'] == 2) & (pupil_size['laser'] == 1)].groupby('time').median()['baseline_subtracted'],
        'no_stim_middle_block': pupil_size[(pupil_size['laser_block_progress'] == 2) & (pupil_size['laser'] == 0)].groupby('time').median()['baseline_subtracted'],
        'stim_late_block': pupil_size[(pupil_size['laser_block_progress'] == 3) & (pupil_size['laser'] == 1)].groupby('time').median()['baseline_subtracted'],
        'no_stim_late_block': pupil_size[(pupil_size['laser_block_progress'] == 3) & (pupil_size['laser'] == 0)].groupby('time').median()['baseline_subtracted'],
        'subject': nickname, 'sert-cre': subjects.loc[i, 'sert-cre']})))

    # Plot this animal
    colors, dpi = figure_style()
    f, ((ax1, ax2, ax5, ax8), (ax3, ax4, ax6, ax7)) = plt.subplots(2, 4, figsize=(8, 4), sharey=True, sharex=False, dpi=dpi)
    """
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='laser', estimator=np.median,
                           data=pupil_size[((pupil_size['laser_prob'] == 0.75) & (pupil_size['laser'] == 1))
                                           | ((pupil_size['laser_prob'] == 0.25) & (pupil_size['laser'] == 0))],
                           palette=[colors['no-stim'], colors['stim']], ci=68, ax=ax1, legend=None)
    """
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='laser', estimator=np.median,
                           data=pupil_size,
                           palette=[colors['no-stim'], colors['stim']], errorbar='se', ax=ax1, legend=None)

    ax1.set(title='%s, expression: %d' % (nickname, subjects.loc[i, 'expression']),
            ylabel='Pupil size (%)', xlabel='Time relative to trial start(s)',
            xticks=np.arange(-1, 3.1), ylim=[-20, 20])
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles=handles, labels=['No stim', 'Stim'], frameon=False)

    sns.lineplot(x='time', y='baseline_subtracted', hue='laser', estimator=np.median,
                 data=pupil_size[((pupil_size['laser_prob'] == 0.75) & (pupil_size['laser'] == 0))
                                           | ((pupil_size['laser_prob'] == 0.25) & (pupil_size['laser'] == 1))],
                 palette=[colors['no-stim'], colors['stim']], errorbar='se', legend=None, ax=ax2)
    ax2.set(xlabel='Time relative to trial start (s)', title='Probe trials', xticks=np.arange(-1, 3.1),
            ylabel='Pupil size change (%)')

    sns.lineplot(x='time', y='baseline_subtracted', data=pupil_size[pupil_size['laser'] == 1],
                 estimator=np.median, hue='abs_log_contrast', palette='viridis', errorbar=('ci', 0), legend=None, ax=ax3)
    ax3.set(xlabel='Time relative to trial start (s)', title='Stimulated trials', xticks=np.arange(-1, 3.1),
            ylabel='Pupil size change (%)')

    sns.lineplot(x='time', y='baseline_subtracted', data=pupil_size[pupil_size['laser'] == 0],
                 estimator=np.median, hue='abs_log_contrast', palette='viridis', errorbar=('ci', 0), ax=ax4)
    ax4.set(xlabel='Time relative to trial start (s)', title='Non-stimulated trials', xticks=np.arange(-1, 3.1),
            ylabel='Pupil size change (%)')
    ax4.legend(frameon=False, bbox_to_anchor=(1, 1), labels=[0, 6.25, 12.5, 25, 100])

    sns.lineplot(x='time', y='baseline_subtracted', hue='laser_prob',
                 data=pupil_size[((pupil_size['laser_prob'] == 0.75) & (pupil_size['laser'] == 1))
                                 | ((pupil_size['laser_prob'] == 0.25) & (pupil_size['laser'] == 1))],
                 estimator=np.median, palette='Paired', errorbar='se', ax=ax5)
    ax5.set(xlabel='Time relative to trial start (s)', title='Stimulated trials', xticks=np.arange(-1, 3.1))
    handles, labels = ax5.get_legend_handles_labels()
    ax5.legend(handles=handles, labels=['Probe', 'Block'], frameon=False)

    sns.lineplot(x='time', y='baseline_subtracted', data=pupil_size[pupil_size['laser'] == 0],
                 hue='laser_block_progress', palette='viridis',
                 estimator=np.median, errorbar='se', ax=ax6)
    ax6.set(xlabel='Time relative to trial start (s)', ylabel='Pupil size change (%)',
            xticks=np.arange(-1, 3.1), title='Non-stimulated trials')
    handles, labels = ax6.get_legend_handles_labels()
    ax6.legend(handles=handles, labels=['early', 'middle', 'late'], frameon=False)

    sns.lineplot(x='time', y='baseline_subtracted', data=pupil_size[pupil_size['laser'] == 1],
                 hue='laser_block_progress', palette='viridis',
                 estimator=np.median, errorbar='se', ax=ax7)
    ax7.set(xlabel='Time relative to trial start (s)', ylabel='Pupil size change (%)',
            xticks=np.arange(-1, 3.1), title='Stimulated trials')
    handles, labels = ax7.get_legend_handles_labels()
    ax7.legend(handles=handles, labels=['early', 'middle', 'late'], frameon=False)

    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='laser', estimator=np.median,
                           data=pupil_size[pupil_size['contrast'] == 0],
                           palette=[colors['no-stim'], colors['stim']], ci=68, ax=ax8, legend=None)

    ax8.set(title='0% contrast', ylabel='Pupil size (%)', xlabel='Time relative to trial start(s)',
            xticks=np.arange(-1, 3.1))

    plt.tight_layout()
    sns.despine(trim=True)

    plt.savefig(join(fig_path, f'{nickname}_pupil_opto_baseline.png'))
    plt.savefig(join(fig_path, f'{nickname}_pupil_opto_baseline.pdf'))

    # Plot this animal
    colors, dpi = figure_style()
    f, ((ax1, ax2, ax5, ax8), (ax3, ax4, ax6, ax7)) = plt.subplots(2, 4, figsize=(8, 4), sharey=True, sharex=False, dpi=dpi)
    """
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='laser', estimator=np.median,
                           data=pupil_size[((pupil_size['laser_prob'] == 0.75) & (pupil_size['laser'] == 1))
                                           | ((pupil_size['laser_prob'] == 0.25) & (pupil_size['laser'] == 0))],
                           palette=[colors['no-stim'], colors['stim']], ci=68, ax=ax1, legend=None)
    """
    lineplt = sns.lineplot(x='time', y='diameter', hue='laser', estimator=np.median,
                           data=pupil_size,
                           palette=[colors['no-stim'], colors['stim']], errorbar='se', ax=ax1, legend=None)

    ax1.set(title='%s, expression: %d' % (nickname, subjects.loc[i, 'expression']),
            ylabel='Pupil size (%)', xlabel='Time relative to trial start(s)',
            xticks=np.arange(-1, 3.1))
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles=handles, labels=['No stim', 'Stim'], frameon=False)

    sns.lineplot(x='time', y='diameter', hue='laser', estimator=np.median,
                 data=pupil_size[((pupil_size['laser_prob'] == 0.75) & (pupil_size['laser'] == 0))
                                           | ((pupil_size['laser_prob'] == 0.25) & (pupil_size['laser'] == 1))],
                 palette=[colors['no-stim'], colors['stim']], errorbar='se', legend=None, ax=ax2)
    ax2.set(xlabel='Time relative to trial start (s)', title='Probe trials', xticks=np.arange(-1, 3.1),
            ylabel='Pupil size change (%)')

    sns.lineplot(x='time', y='diameter', data=pupil_size[pupil_size['laser'] == 1],
                 estimator=np.median, hue='abs_log_contrast', palette='viridis', ci=0, legend=None, ax=ax3)
    ax3.set(xlabel='Time relative to trial start (s)', title='Stimulated trials', xticks=np.arange(-1, 3.1),
            ylabel='Pupil size change (%)')

    sns.lineplot(x='time', y='diameter', data=pupil_size[pupil_size['laser'] == 0],
                 estimator=np.median, hue='abs_log_contrast', palette='viridis', ci=0, ax=ax4)
    ax4.set(xlabel='Time relative to trial start (s)', title='Non-stimulated trials', xticks=np.arange(-1, 3.1),
            ylabel='Pupil size change (%)')
    ax4.legend(frameon=False, bbox_to_anchor=(1, 1), labels=[0, 6.25, 12.5, 25, 100])

    sns.lineplot(x='time', y='diameter', hue='laser_prob',
                 data=pupil_size[((pupil_size['laser_prob'] == 0.75) & (pupil_size['laser'] == 1))
                                 | ((pupil_size['laser_prob'] == 0.25) & (pupil_size['laser'] == 1))],
                 estimator=np.median, palette='Paired', errorbar='se', ax=ax5)
    ax5.set(xlabel='Time relative to trial start (s)', title='Stimulated trials', xticks=np.arange(-1, 3.1))
    handles, labels = ax5.get_legend_handles_labels()
    ax5.legend(handles=handles, labels=['Probe', 'Block'], frameon=False)

    plt.tight_layout()
    sns.despine(trim=True)

    sns.lineplot(x='time', y='diameter', data=pupil_size[pupil_size['laser'] == 0],
                 hue='laser_block_progress', palette='viridis',
                 estimator=np.median, errorbar='se', ax=ax6)
    ax6.set(xlabel='Time relative to trial start (s)', ylabel='Pupil size change (%)',
            xticks=np.arange(-1, 3.1), title='Non-stimulated trials')
    handles, labels = ax6.get_legend_handles_labels()
    ax6.legend(handles=handles, labels=['early', 'middle', 'late'], frameon=False)

    sns.lineplot(x='time', y='diameter', data=pupil_size[pupil_size['laser'] == 1],
                 hue='laser_block_progress', palette='viridis',
                 estimator=np.median, errorbar='se', ax=ax7)
    ax7.set(xlabel='Time relative to trial start (s)', ylabel='Pupil size change (%)',
            xticks=np.arange(-1, 3.1), title='Stimulated trials')
    handles, labels = ax7.get_legend_handles_labels()
    ax7.legend(handles=handles, labels=['early', 'middle', 'late'], frameon=False)

    lineplt = sns.lineplot(x='time', y='diameter', hue='laser', estimator=np.median,
                           data=pupil_size[pupil_size['feedback_type'] == -1],
                           palette=[colors['no-stim'], colors['stim']], errorbar='se', ax=ax8, legend=None)
    ax8.set(title='Unrewarded trials', ylabel='Pupil size (%)', xlabel='Time relative to trial start(s)',
            xticks=np.arange(-1, 3.1),)

    plt.savefig(join(fig_path, f'{nickname}_pupil_opto.pdf'), dpi=600)




# %%
results_df = results_df.reset_index()
results_df['diff_stim_all'] = results_df['stim_all'] - results_df['no_stim_all']
#results_df['diff_stim_all'] = ((results_df['stim_all'] - results_df['no_stim_all']) / results_df['no_stim_all']) * 100
results_df['diff_stim_blank'] = results_df['stim_blank'] - results_df['no_stim_blank']
results_df['diff_stim_block'] = results_df['stim_block'] - results_df['no_stim_block']
results_df['diff_stim_probe'] = results_df['stim_probe'] - results_df['no_stim_probe']
results_df['diff_stim_early'] = results_df['stim_early_block'] - results_df['no_stim_early_block']
results_df['diff_stim_middle'] = results_df['stim_middle_block'] - results_df['no_stim_middle_block']
results_df['diff_stim_late'] = results_df['stim_late_block'] - results_df['no_stim_late_block']
results_df['diff_block'] = results_df['stim_block'] - results_df['stim_probe']

results_df_baseline = results_df_baseline.reset_index()
results_df_baseline['diff_stim_all'] = results_df_baseline['stim_all'] - results_df_baseline['no_stim_all']
#results_df['diff_stim_all'] = ((results_df['stim_all'] - results_df['no_stim_all']) / results_df['no_stim_all']) * 100
results_df_baseline['diff_stim_blank'] = results_df_baseline['stim_blank'] - results_df_baseline['no_stim_blank']
results_df_baseline['diff_stim_block'] = results_df_baseline['stim_block'] - results_df_baseline['no_stim_block']
results_df_baseline['diff_stim_probe'] = results_df_baseline['stim_probe'] - results_df_baseline['no_stim_probe']
results_df_baseline['diff_stim_early'] = results_df_baseline['stim_early_block'] - results_df_baseline['no_stim_early_block']
results_df_baseline['diff_stim_middle'] = results_df_baseline['stim_middle_block'] - results_df_baseline['no_stim_middle_block']
results_df_baseline['diff_stim_late'] = results_df_baseline['stim_late_block'] - results_df_baseline['no_stim_late_block']
results_df_baseline['diff_block'] = results_df_baseline['stim_block'] - results_df_baseline['stim_probe']

# %% Plot over mice
colors, dpi = figure_style()
f, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, figsize=(8, 4), dpi=dpi)

sns.lineplot(x='time', y='diff_stim_all', data=results_df, hue='sert-cre',
             units='subject', estimator=None, legend=None, ax=ax1,
             palette=[colors['wt'], colors['sert']], lw=1.5)
ax1.set(xlabel='Time relative to trial start (s)', ylabel='5-HT stim. induced pupil size change (%)',
              xticks=np.arange(-1, 3.1), title='All trials')

sns.lineplot(x='time', y='diff_stim_blank', data=results_df, hue='sert-cre',
             units='subject', estimator=None, legend=None, ax=ax2,
             palette=[colors['wt'], colors['sert']], lw=1.5)
ax2.set(xlabel='Time relative to trial start (s)', ylabel='5-HT stim. induced pupil size change (%)',
              xticks=np.arange(-1, 3.1), title='0% contrast')

sns.lineplot(x='time', y='diff_stim_block', data=results_df, hue='sert-cre',
             units='subject', estimator=None, legend=None, ax=ax3,
             palette=[colors['wt'], colors['sert']], lw=1.5)
ax3.set(xlabel='Time relative to trial start (s)', ylabel='5-HT stim. induced pupil size change (%)',
              xticks=np.arange(-1, 3.1))

sns.lineplot(x='time', y='diff_stim_probe', data=results_df, hue='sert-cre',
             units='subject', estimator=None, legend=None, ax=ax4,
             palette=[colors['wt'], colors['sert']], lw=1.5)
ax4.set(xlabel='Time relative to trial start (s)', ylabel='5-HT stim. induced pupil size change (%)',
              xticks=np.arange(-1, 3.1))

sns.lineplot(x='time', y='diff_block', data=results_df, hue='sert-cre',
             units='subject', estimator=None, legend=None, ax=ax5)

sns.lineplot(x='time', y='diff_stim_early', data=results_df, hue='sert-cre',
             units='subject', estimator=None, legend=None, ax=ax6)
sns.lineplot(x='time', y='diff_stim_middle', data=results_df, hue='sert-cre',
             units='subject', estimator=None, legend=None, ax=ax6)
sns.lineplot(x='time', y='diff_stim_late', data=results_df, hue='sert-cre',
             units='subject', estimator=None, legend=None, ax=ax6)

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'OverAnimalsDiff.png'))
plt.savefig(join(fig_path, 'OverAnimalsDiff.pdf'))

f, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, figsize=(8, 4), sharey=True, dpi=dpi)

sns.lineplot(x='time', y='diff_stim_all', data=results_df_baseline, hue='sert-cre',
             units='subject', estimator=None, legend=None, ax=ax1,
             palette=[colors['wt'], colors['sert']], lw=1.5)
ax1.set(xlabel='Time relative to trial start (s)', ylabel='5-HT stim. induced\npupil size change (%)',
              xticks=np.arange(-1, 3.1), title='All trials', ylim=[-6, 6])

sns.lineplot(x='time', y='diff_stim_blank', data=results_df_baseline, hue='sert-cre',
             units='subject', estimator=None, legend=None, ax=ax2,
             palette=[colors['wt'], colors['sert']], lw=1.5)
ax2.set(xlabel='Time relative to trial start (s)', ylabel='5-HT stim. induced pupil size change (%)',
              xticks=np.arange(-1, 3.1), title='0% contrast')

sns.lineplot(x='time', y='diff_stim_block', data=results_df_baseline, hue='sert-cre',
             units='subject', estimator=None, legend=None, ax=ax3,
             palette=[colors['wt'], colors['sert']], lw=1.5)
ax3.set(xlabel='Time relative to trial start (s)', ylabel='5-HT stim. induced pupil size change (%)',
              xticks=np.arange(-1, 3.1))

sns.lineplot(x='time', y='diff_stim_probe', data=results_df_baseline, hue='sert-cre',
             units='subject', estimator=None, legend=None, ax=ax4,
             palette=[colors['wt'], colors['sert']], lw=1.5)
ax4.set(xlabel='Time relative to trial start (s)', ylabel='5-HT stim. induced pupil size change (%)',
              xticks=np.arange(-1, 3.1))

sns.lineplot(x='time', y='diff_block', data=results_df_baseline, hue='sert-cre',
             units='subject', estimator=None, legend=None, ax=ax5)

sns.lineplot(x='time', y='diff_stim_early', data=results_df_baseline, hue='sert-cre',
             units='subject', estimator=None, legend=None, ax=ax6)
sns.lineplot(x='time', y='diff_stim_middle', data=results_df_baseline, hue='sert-cre',
             units='subject', estimator=None, legend=None, ax=ax6)
sns.lineplot(x='time', y='diff_stim_late', data=results_df_baseline, hue='sert-cre',
             units='subject', estimator=None, legend=None, ax=ax6)

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'OverAnimalsDiffBaseline.png'))
plt.savefig(join(fig_path, 'OverAnimalsDiffBaseline.pdf'))
