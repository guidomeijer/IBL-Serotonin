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
from datetime import datetime
from models.expSmoothing_prevAction import expSmoothing_prevAction as exp_prev_action
from serotonin_functions import paths, criteria_opto_eids, load_exp_smoothing_trials, figure_style
from oneibl.one import ONE
one = ONE()

# Settings
REMOVE_OLD_FIT = False
POSTERIOR = 'posterior_mean'
_, fig_path, save_path = paths()
fig_path = join(fig_path, '5HT', 'opto-behavior')

subjects = pd.read_csv(join('..', 'subjects.csv'))

results_df = pd.DataFrame()
accuracy_df = pd.DataFrame()
block_switches = pd.DataFrame()
for i, nickname in enumerate(subjects['subject']):

    # Query sessions
    if subjects.loc[i, 'date_range_blocks'] == 'all':
        eids, ses_details = one.search(subject=nickname,
                                       task_protocol='_iblrig_tasks_opto_biasedChoiceWorld',
                                       details=True)
    elif subjects.loc[i, 'date_range_blocks'] == 'none':
        continue
    else:
        eids, ses_details = one.search(subject=nickname, task_protocol='_iblrig_tasks_opto_biasedChoiceWorld',
                                       date_range=[subjects.loc[i, 'date_range_blocks'][:10],
                                                   subjects.loc[i, 'date_range_blocks'][11:]],
                                       details=True)

    #eids = criteria_opto_eids(eids, max_lapse=0.5, max_bias=0.5, min_trials=200, one=one)
    if len(eids) == 0:
        continue

    # Get list of session dates and order
    ses_dates = [datetime.strptime(i, '%Y-%m-%d') for i in [j['start_time'][:10] for j in ses_details]]
    ses_day = np.argsort(ses_dates)

    print(f'Processing {nickname} [{i + 1} of {subjects.shape[0]}]')
    for j, eid in enumerate(eids):
        print(f'Processing session {j + 1} of {len(eids)}')

        # Get trial data
        actions, stimuli, stim_side, prob_left, stimulated, session_uuids = load_exp_smoothing_trials(
            [eid], laser_stimulation=True, one=one)

        if len(session_uuids) > 0:
            # Fit models
            model = exp_prev_action('./model_fit_results/', session_uuids, f'{nickname}',
                                    actions, stimuli, stim_side)
            model.load_or_train(nb_steps=2000, remove_old=REMOVE_OLD_FIT)
            params = model.get_parameters(parameter_type=POSTERIOR)
            results_df = results_df.append(pd.DataFrame(index=[results_df.shape[0]+1], data={
                'tau': 1/params[0], 'sert-cre': subjects.loc[i, 'sert-cre'], 'subject': nickname,
                'date': ses_details[j]['start_time'][:10],
                'day': ses_day[j] + 1}))

# %% Plot

figure_style()
f, ax1 = plt.subplots()
sns.lineplot(x='day', y='tau', data=results_df, hue='sert-cre', style='subject', estimator=None, dashes=False)

