#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 11:42:01 2021

@author: guido
"""

import numpy as np
from os.path import join
import pandas as pd
from dlc_functions import get_dlc_XYs, get_pupil_diameter
import matplotlib.pyplot as plt
from scipy.stats import zscore
import seaborn as sns
from serotonin_functions import load_trials, paths, query_opto_sessions, figure_style, load_subjects
from one.api import ONE
one = ONE()

# Settings
TIME_BINS = np.arange(-1, 3.2, 0.2)
BIN_SIZE = 0.2
BASELINE = 0.5
N_TRIALS = 20
_, fig_path, _ = paths()
fig_path = join(fig_path, 'Pupil')

subjects = load_subjects(behavior=True)
results_df = pd.DataFrame()
for i, nickname in enumerate(subjects['subject']):
    print(f'Processing {nickname}..')

    # Query sessions
    eids = query_opto_sessions(nickname, one=one)

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

        # Get stimulation block transitions
        no_probe_trials = trials.copy()
        no_probe_trials.loc[no_probe_trials['probe_trial'] == 1, 'laser_stimulation'] = (
            no_probe_trials.loc[no_probe_trials['probe_trial'] == 1, 'laser_probability'].round())
        stim_block_trans = np.append([0], np.array(np.where(np.diff(no_probe_trials['laser_stimulation']) != 0)) + 1)

        # Divide up stim blocks
        for k, trans in enumerate(stim_block_trans):
            trials.loc[trans:trans+N_TRIALS, 'block_progress'] = 1
            trials.loc[trans+N_TRIALS+1:trans+N_TRIALS*2+1, 'block_progress'] = 2
            if k != stim_block_trans.shape[0]-1:
                trials.loc[trans+N_TRIALS*2+1:stim_block_trans[k+1], 'block_progress'] = 3
            else:
                trials.loc[trans+N_TRIALS*2+1:-1, 'block_progress'] = 3

        # Load in camera timestamps and DLC output
        try:
            video_times, XYs = get_dlc_XYs(eid, one=one)
        except:
            print('Could not load video and/or DLC data')
            continue

        # If the difference between timestamps and video frames is too large, skip
        if np.abs(video_times.shape[0]
                  - XYs['pupil_left_r'].shape[len(XYs['pupil_left_r'].shape) - 1]) > 10000:
            print('Timestamp mismatch, skipping..')
            continue

        # Get pupil diameter
        diameter = get_pupil_diameter(XYs)

        # Assume frames were dropped at the end
        if video_times.shape[0] > diameter.shape[0]:
            video_times = video_times[:diameter.shape[0]]
        elif diameter.shape[0] > video_times.shape[0]:
            diameter = diameter[:video_times.shape[0]]

        # z-score pupil
        diameter_zscore = zscore(diameter)

        # Get trial triggered baseline subtracted pupil diameter
        for t, trial_start in enumerate(trials['stimOn_times']):
            this_diameter = np.array([np.nan] * TIME_BINS.shape[0])
            baseline_subtracted = np.array([np.nan] * TIME_BINS.shape[0])
            baseline = np.mean(diameter_zscore[(video_times > (trial_start - BASELINE))
                                               & (video_times < trial_start)])
            for b, time_bin in enumerate(TIME_BINS):
                this_diameter[b] = np.mean(diameter_zscore[
                    (video_times > (trial_start + time_bin) - (BIN_SIZE / 2))
                    & (video_times < (trial_start + time_bin) + (BIN_SIZE / 2))])
                baseline_subtracted[b] = np.mean(diameter_zscore[
                    (video_times > (trial_start + time_bin) - (BIN_SIZE / 2))
                    & (video_times < (trial_start + time_bin) + (BIN_SIZE / 2))]) - baseline
            pupil_size = pupil_size.append(pd.DataFrame(data={
                'diameter': this_diameter, 'baseline_subtracted': baseline_subtracted, 'eid': eid,
                'subject': nickname, 'trial': t, 'contrast': trials.loc[t, 'signed_contrast'],
                'sert': subjects.loc[i, 'sert-cre'], 'laser': trials.loc[t, 'laser_stimulation'],
                'laser_prob': trials.loc[t, 'laser_probability'], 'progress': trials.loc[t, 'block_progress'],
                'time': TIME_BINS}))
    if pupil_size.shape[0] < 100:
        continue

    # Add to overal dataframe
    pupil_size = pupil_size.reset_index(drop=True)
    pupil_size['abs_contrast'] = pupil_size['contrast'].abs()
    pupil_size['abs_log_contrast'] = pupil_size['abs_contrast'].copy()
    pupil_size.loc[pupil_size['abs_log_contrast'] == 0, 'abs_log_contrast'] = 0.01
    pupil_size['abs_log_contrast'] = np.log10(pupil_size['abs_log_contrast'])
    results_df = results_df.append(pd.DataFrame(data={
        'stim_block': pupil_size[(pupil_size['laser_prob'] == 0.75) & (pupil_size['laser'] == 1)].groupby('time').mean()['baseline_subtracted'],
        'no_stim_block': pupil_size[(pupil_size['laser_prob'] == 0.25) & (pupil_size['laser'] == 0)].groupby('time').mean()['baseline_subtracted'],
        'stim_probe': pupil_size[(pupil_size['laser_prob'] == 0.25) & (pupil_size['laser'] == 1)].groupby('time').mean()['baseline_subtracted'],
        'no_stim_probe': pupil_size[(pupil_size['laser_prob'] == 0.75) & (pupil_size['laser'] == 0)].groupby('time').mean()['baseline_subtracted'],
        'subject': nickname, 'sert-cre': subjects.loc[i, 'sert-cre']}))

    pupil_df = pd.concat([pupil_size[pupil_size['laser'] == 1].groupby(['time', 'progress']).mean()['baseline_subtracted'].rename('stim'),
                          pupil_size[pupil_size['laser'] == 0].groupby(['time', 'progress']).mean()['baseline_subtracted'].rename('no-stim')], axis=1)
    pupil_df['stim_diff'] = pupil_df['stim'] - pupil_df['no-stim']


    # Plot this animal
    colors, dpi = figure_style()
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 2.5), dpi=dpi)
    sns.lineplot(x='time', y='stim_diff', hue='progress', data=pupil_df, palette='flare', ci=68, ax=ax1)
    ax1.set(title='%s, expression: %d' % (nickname, subjects.loc[i, 'expression']), ylim=[-0.3, 0.3],
            yticks=np.arange(-0.3, 0.31, 0.1), ylabel='Stimulation induced change in pupil size',
            xlabel='Time relative to trial start(s)')
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles=handles, labels=[f'1-{N_TRIALS}', f'{N_TRIALS+1}-{N_TRIALS*2}', f'{N_TRIALS*2+1}+'],
               frameon=False, title='trials after switch')

    sns.lineplot(x='time', y='baseline_subtracted', data=pupil_size, hue='progress', palette='flare', ax=ax2)
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles=handles, labels=[f'1-{N_TRIALS}', f'{N_TRIALS+1}-{N_TRIALS*2}', f'{N_TRIALS*2+1}+'],
               frameon=False, title='trials after switch')
    ax2.set(ylabel='z-scored pupil size', xlabel='Time relative to trial start(s)')

    plt.tight_layout()
    sns.despine(trim=True)

    plt.savefig(join(fig_path, f'{nickname}_pupil_opto_uncertainty.png'))
    plt.savefig(join(fig_path, f'{nickname}_pupil_opto_uncertainty.pdf'))

results_df = results_df.reset_index()
results_df['diff_stim_block'] = results_df['stim_block'] - results_df['no_stim_block']
results_df['diff_stim_probe'] = results_df['stim_probe'] - results_df['no_stim_probe']
results_df['diff_block'] = results_df['stim_block'] - results_df['stim_probe']

# %% Plot over mice

ax = dict()
f, (ax['dsb'], ax['dsp'], ax['db']) = plt.subplots(1, 3, figsize=(6, 2), dpi=dpi)

sns.lineplot(x='time', y='diff_stim_block', data=results_df, hue='sert-cre',
             units='subject', estimator=None, legend=None, ax=ax['dsb'])

sns.lineplot(x='time', y='diff_stim_probe', data=results_df, hue='sert-cre',
             units='subject', estimator=None, legend=None, ax=ax['dsp'])

sns.lineplot(x='time', y='diff_block', data=results_df, hue='sert-cre',
             units='subject', estimator=None, legend=None, ax=ax['db'])



