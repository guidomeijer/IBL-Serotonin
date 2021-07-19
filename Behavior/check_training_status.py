# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 11:38:26 2021

@author: Guido Meijer
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from serotonin_functions import load_trials, plot_psychometric, fit_psychfunc, figure_style
from one.api import ONE
one = ONE()

# Settings
N_SESSIONS = 3

# Query all current subjects
subj_info = one.alyx.rest('subjects', 'list', project='serotonin_inference', alive=True)
subjects = [i['nickname'] for i in subj_info]

# Set up figure
colors, dpi = figure_style()
f, axs = plt.subplots(int(np.floor(np.sqrt(len(subjects)))), int(np.ceil(np.sqrt(len(subjects)))),
                      figsize=(8, 4), dpi=dpi, sharex=True, sharey=True)
axs = np.concatenate(axs)

# Loop through subjects
for i, nickname in enumerate(subjects):
    eids, details = one.search(subject=nickname, details=True)
    if 'training' in details[0]['task_protocol']:
        print(f'\n-- {nickname} --')

        # Mouse is in training
        eids, details = one.search(subject=nickname, details=True)
        if len(eids) < N_SESSIONS:
            print(f'{nickname} does not have enough sessions to determine training status')
            continue
        trials = pd.DataFrame()
        ses_loaded = 0
        j = -1
        while ses_loaded < N_SESSIONS:
            j += 1
            ses_date = str(details[j]['date'])
            try:
                these_trials = load_trials(eids[j], one=one)
                these_trials['date'] = ses_date
                trials = trials.append(these_trials, ignore_index=True)
                ses_loaded += 1
            except:
                pass
        last_ses = trials['date'].max()
        print(f'Last session loaded for {nickname}: {last_ses}')

        # Fit psychometric curve
        stim_levels = np.sort(trials['signed_contrast'].unique())
        if stim_levels.shape[0] < 9:
            print(f'{nickname} is NOT TRAINED (not all contrasts are introduced)')
            continue
        pars = fit_psychfunc(stim_levels, trials.groupby('signed_contrast').size(),
                             trials.groupby('signed_contrast').mean()['right_choice'])
        bias = pars[0]
        threshold = pars[1]
        perc_correct = (trials.loc[np.abs(trials['signed_contrast']) >= 0.5, 'correct'].sum()
                        / trials.loc[np.abs(trials['signed_contrast']) >= 0.5, 'correct'].shape[0])
        if (np.abs(bias) < 15) & (threshold < 20) & (perc_correct > 0.8):
            print(f'{nickname} is TRAINED and ready to be moved to biased')
            print(f'bias: {bias:.1f}\nthreshold: {threshold:.1f}\nperf: {perc_correct*100:.1f}%')
            axs[i].text(-35, 0.85, f'{nickname}\nready for biased', fontsize=7)
        else:
            print(f'{nickname} is NOT TRAINED')
            print(f'bias: {bias:.1f} (<15)\nthreshold: {threshold:.1f} (<20)\nperf: {perc_correct*100:.1f}% (>80%)')
            axs[i].text(-35, 0.85, f'{nickname}\nin training', fontsize=7)

        # Plot psychometric curve
        plot_psychometric(trials, ax=axs[i], color='k')
        plt.tight_layout()
        sns.despine(trim=True)

    elif details[0]['task_protocol'][:31] == '_iblrig_tasks_biasedChoiceWorld':
        print(f'\n-- {nickname} --')

        # Mouse is in biased
        eids, details = one.search(subject=nickname,
                                   task_protocol='_iblrig_tasks_biasedChoiceWorld', details=True)
        if len(eids) < N_SESSIONS:
            this_n_ses = len(eids)
        else:
            this_n_ses = N_SESSIONS
        trials = pd.DataFrame()
        ses_loaded = 0
        j = -1
        while ses_loaded < N_SESSIONS:
            j += 1
            ses_date = str(details[j]['date'])
            try:
                these_trials = load_trials(eids[j], one=one)
                these_trials['date'] = ses_date
                trials = trials.append(these_trials, ignore_index=True)
                ses_loaded += 1
            except:
                pass
        last_ses = trials['date'].max()
        print(f'Last session loaded for {nickname}: {last_ses}')
        perc_correct = (trials.loc[np.abs(trials['signed_contrast']) >= 0.5, 'correct'].sum()
                        / trials.loc[np.abs(trials['signed_contrast']) >= 0.5, 'correct'].shape[0])
        if (len(eids) >= 5) and (perc_correct > 0.8):
            print(f'{nickname} is READY for opto\n{len(eids)} biased sessions'
                  f'\n% correct last session: {perc_correct*100:.1f}%')
            axs[i].text(-35, 0.85, f'{nickname}\nready for opto', fontsize=7)
        else:
            print(f'{nickname} is NOT READY for opto\n{len(eids)} biased sessions'
                  f'\n% correct last session: {perc_correct*100:.1f}%')
            axs[i].text(-35, 0.85, f'{nickname}\nin biased', fontsize=7)

        # Plot psychometric curve
        plot_psychometric(trials[trials['probabilityLeft'] == 0.8], ax=axs[i],
                          color=colors['left'])
        plot_psychometric(trials[trials['probabilityLeft'] == 0.2], ax=axs[i],
                          color=colors['right'])
        plt.tight_layout()
        sns.despine(trim=True)


    elif details[0]['task_protocol'][:36] == '_iblrig_tasks_opto_biasedChoiceWorld':
        print(f'\n-- {nickname} --')

        # Mouse is in opto
        eids, details = one.search(subject=nickname,
                                   task_protocol='_iblrig_tasks_opto_biasedChoiceWorld',
                                   details=True)
        if len(eids) < N_SESSIONS:
            this_n_ses = len(eids)
        else:
            this_n_ses = N_SESSIONS
        trials = pd.DataFrame()
        ses_loaded = 0
        j = -1
        while ses_loaded < N_SESSIONS:
            j += 1
            ses_date = str(details[j]['date'])
            try:
                these_trials = load_trials(eids[j], one=one)
                these_trials['date'] = ses_date
                trials = trials.append(these_trials, ignore_index=True)
                ses_loaded += 1
            except:
                pass
        last_ses = trials['date'].max()
        print(f'Last session loaded for {nickname}: {last_ses}')
        perc_correct = (trials.loc[np.abs(trials['signed_contrast']) >= 0.5, 'correct'].sum()
                        / trials.loc[np.abs(trials['signed_contrast']) >= 0.5, 'correct'].shape[0])
        if (len(eids) >= 5) and (perc_correct > 0.8):
            print(f'{nickname} is READY for ephys\n{len(eids)} opto sessions'
                  f'\n% correct last session: {perc_correct*100:.1f}%')
            axs[i].text(-35, 0.85, f'{nickname}\nready for ephys', fontsize=7)
        else:
            print(f'{nickname} is NOT READY for ephys\n{len(eids)} opto sessions'
                  f'\n% correct last session: {perc_correct*100:.1f}%')
            axs[i].text(-35, 0.85, f'{nickname}\nin opto', fontsize=7)

        # Plot psychometric curve
        plot_psychometric(trials[trials['probabilityLeft'] == 0.8], ax=axs[i],
                          color=colors['left'])
        plot_psychometric(trials[trials['probabilityLeft'] == 0.2], ax=axs[i],
                          color=colors['right'])
        plt.tight_layout()
        sns.despine(trim=True)
