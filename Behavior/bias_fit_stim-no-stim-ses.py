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
                                 load_subjects, fit_psychfunc, query_opto_sessions, get_bias)
from one.api import ONE
one = ONE()

# Settings
_, fig_path, save_path = paths()
fig_path = join(fig_path, 'Behavior', 'Psychometrics')
N_SES = 5

subjects = load_subjects()
subjects = subjects[subjects['subject'] != 'ZFM-02602']
subjects = subjects[subjects['subject'] != 'ZFM-02180']
subjects = subjects[subjects['subject'] != 'ZFM-01867']
subjects = subjects.reset_index()

results_df = pd.DataFrame()
for i, nickname in enumerate(subjects['subject']):
    print(f'Processing {nickname}')
    sessions = one.alyx.rest('sessions', 'list', subject=nickname,
                             task_protocol='biased',
                             project='serotonin_inference')
    eids = [sess['url'][-36:] for sess in sessions][::-1]
    if len(eids) == 0:
        continue

    # Apply behavior criterion
    #eids = behavioral_criterion(eids)
    eids = np.array(eids)

    # Get first opto session array
    opto = np.empty(len(eids))
    for j, eid in enumerate(eids):
        details = one.get_details(eid)
        opto[j] = int('opto' in details['task_protocol'])
    first_opto = np.where(opto == 1)[0][0]
    rel_ses = np.append(np.arange(-first_opto, 0), np.arange(1, (opto.shape[0] - first_opto) + 1))

    # Get bias for non-stim sessions
    trials = pd.DataFrame()
    for j, eid in enumerate(eids[(rel_ses >= -3) & (rel_ses < 0)]):
        try:
            if subjects.loc[i, 'sert-cre'] == 1:
                these_trials = load_trials(eid, laser_stimulation=False, one=one)
            else:
                these_trials = load_trials(eid, laser_stimulation=False, patch_old_opto=False, one=one)
            """
            these_trials = load_trials(eid, laser_stimulation=True, one=one)
            """
            trials = trials.append(these_trials, ignore_index=True)
        except:
            pass
    bias_no_stim = get_bias(trials)

    # Get bias for stim sessions
    trials = pd.DataFrame()
    for j, eid in enumerate(eids[(rel_ses <= 3) & (rel_ses > 1) & (opto == 1)]):
        try:
            if subjects.loc[i, 'sert-cre'] == 1:
                these_trials = load_trials(eid, laser_stimulation=True, one=one)
            else:
                these_trials = load_trials(eid, laser_stimulation=True, patch_old_opto=False, one=one)
            """
            these_trials = load_trials(eid, laser_stimulation=True, one=one)
            """
            trials = trials.append(these_trials, ignore_index=True)
        except:
            pass
    bias_stim = get_bias(trials)

    results_df = results_df.append(pd.DataFrame(index=[results_df.shape[0]+1], data={
        'subject': nickname, 'sert-cre': subjects.loc[i, 'sert-cre'], 'bias_no_stim': bias_no_stim,
        'bias_stim': bias_stim}))

# %% Plot
colors, dpi = figure_style()
colors = [colors['wt'], colors['sert']]
f, ax1 = plt.subplots(1, 1, figsize=(2, 2), dpi=dpi)
for i, subject in enumerate(results_df['subject']):
    ax1.plot([1, 2], [results_df.loc[results_df['subject'] == subject, 'bias_no_stim'],
                      results_df.loc[results_df['subject'] == subject, 'bias_stim']],
             color = colors[results_df.loc[results_df['subject'] == subject, 'sert-cre'].values[0]], marker='o', ms=2)
ax1.set(xlabel='', xticks=[1, 2], xticklabels=['Non-stimulated\nsessions', 'Stimulated\nsessions'],
        ylabel='Bias', ylim=[0.1, 0.7], yticks=np.arange(0.1, 0.71, 0.1))
plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'bias_fit_stim_no_stim_ses.png'), dpi=300)
plt.savefig(join(fig_path, 'bias_fit_stim_no_stim_ses.pdf'))

