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
import seaborn as sns
from serotonin_functions import paths, figure_style, load_subjects
from one.api import ONE
one = ONE()

# Settings
BASELINE = 5
OPTO_SESSIONS = 10
EPHYS_RIG = False
_, fig_path, _ = paths()
fig_path = join(fig_path, 'Pupil')

subjects = load_subjects()
subjects = subjects[subjects['subject'] != 'ZFM-02602']
subjects = subjects[subjects['subject'] != 'ZFM-02180']
subjects = subjects[subjects['subject'] != 'ZFM-01867']
subjects = subjects.reset_index(drop=True)
results_df = pd.DataFrame()
for i, nickname in enumerate(subjects['subject']):
    print(f'Processing {nickname}..')

    # Query all sessions
    eids, details = one.search(subject=nickname, task_protocol='biased', details=True)
    eids = np.array(eids)
    if len(eids) == 0:
        continue
    all_dates = [i['date'] for i in details]

    # Find first stimulated session
    _, stim_details = one.search(subject=nickname,
                                 task_protocol=['_iblrig_tasks_opto_biasedChoiceWorld'],
                                 details=True)
    stim_dates = [i['date'] for i in stim_details]
    if len(stim_dates) == 0:
        continue
    rel_ses = -(np.arange(len(all_dates)) - all_dates.index(np.min(stim_dates)))

    # Loop over sessions
    eids = eids[(rel_ses >= -BASELINE) & (rel_ses <= OPTO_SESSIONS)]
    rel_ses = rel_ses[(rel_ses >= -BASELINE) & (rel_ses <= OPTO_SESSIONS)]
    pupil_size = pd.DataFrame()
    for j, eid in enumerate(eids[(rel_ses >= -BASELINE) & (rel_ses <= OPTO_SESSIONS)]):
        print(f'Processing day {rel_ses[j]}')
        if EPHYS_RIG:
            rig = one.get_details(eid, full=True)['location']
            if 'ephys' not in rig:
                continue

        # Load in camera timestamps and DLC output
        try:
            video_times, XYs = get_dlc_XYs(eid, one=one)
        except:
            print('Could not load video and/or DLC data')
            continue
        if XYs['pupil_bottom_r'][0].shape[0] < 100:
            continue

        # Get pupil diameter
        diameter = get_pupil_diameter(XYs)

        # Add to dataframe
        results_df = results_df.append(pd.DataFrame(index=[results_df.shape[0] + 1], data={
            'subject': nickname, 'date': details[j]['date'], 'session': rel_ses[j],
            'opto': 'opto' in details[j]['task_protocol'], 'sert-cre': subjects.loc[i, 'sert-cre'],
            'diameter_mean': np.mean(diameter), 'diameter_std': np.std(diameter),
            'diameter_median': np.median(diameter)}))

    # Subtract baseline
    results_df.loc[results_df['subject'] == nickname, 'diameter_median_baseline'] = (
        (results_df.loc[results_df['subject'] == nickname, 'diameter_median']
        - (results_df.loc[(results_df['subject'] == nickname)
                         & (results_df['session'].between(-BASELINE, -1)), 'diameter_median'].median())))

# %% Plot
colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(2, 2), dpi=dpi)
"""
sns.lineplot(x='session', y='bias', hue='sert-cre', style='subject', estimator=None,
             data=results_df, dashes=False, markers=['o']*int(results_df.shape[0]/2),
             legend=False, lw=2, ms=8, palette=[colors['wt'], colors['sert']], ax=ax1)

sns.lineplot(x='session', y='bias', hue='sert-cre', style='subject', dashes=False,
             estimator=None,
             data=results_df, legend=False, lw=2, ms=8, palette=[colors['wt'], colors['sert']], ax=ax1)
"""
sns.lineplot(x='session', y='diameter_median_baseline', hue='sert-cre', ci=68,
             data=results_df[results_df['sert-cre'] == 1], legend=False,
             palette=[colors['sert']], ax=ax1)
ax1.set(ylabel='Pupil diameter', xlabel='Days relative to first opto day', xlim=[-5, 10])

