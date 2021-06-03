# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 12:10:23 2019

@author: guido
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from functions_pharmacology import paths, load_session_one

# Settings
FIG_PATH = paths()[1]
SAVE_PLOT = True
PREVIOUS_TRIALS = 10

# Load in session dates
sessions = pd.read_csv('pharmacology_sessions.csv', header=1)

# Load data
results = pd.DataFrame()
slopes = pd.DataFrame()
for i, nickname in enumerate(sessions['Nickname']):
    for j, condition in enumerate(['Pre-vehicle', 'Drug']):

        # Load in data
        print('Processing subject %s, session %s' % (nickname, sessions.loc[i, condition]))
        contrast_l, contrast_r, prob_l, correct, choice = load_session_one(
                                                            nickname, sessions.loc[i, condition])

        # Transform contrast and make signed
        p = 3.5
        contrast_l_tr = np.tanh(contrast_l * p) / np.tanh(p)
        contrast_r_tr = np.tanh(contrast_r * p) / np.tanh(p)
        signed_contrast = np.copy(contrast_l_tr)
        signed_contrast[np.isnan(signed_contrast)] = -contrast_r_tr[~np.isnan(contrast_r_tr)]

        # Split out contrasts that were succesfully detected from detection failures
        success_contrast = np.zeros(signed_contrast.shape)
        success_contrast[correct == 1] = signed_contrast[correct == 1]
        failure_contrast = np.zeros(signed_contrast.shape)
        failure_contrast[correct == -1] = signed_contrast[correct == -1]

        # Create dataframe
        data = pd.DataFrame(data={'signed_contrast': signed_contrast,
                                  'success_contrast': success_contrast,
                                  'failure_contrast': failure_contrast})

        # Shift successfull and failure contrasts
        for t in range(PREVIOUS_TRIALS):
            data.loc[:, 'success-%s' % str(t + 1)] = data['success_contrast'].shift(
                                                        periods=t+1, fill_value=0).to_numpy()
            data.loc[:, 'failure-%s' % str(t + 1)] = data['failure_contrast'].shift(
                                                        periods=t+1, fill_value=0).to_numpy()

        # Only predict the choice during 0% contrast trials
        data = data[data['signed_contrast'] == 0]
        data.drop(columns=['signed_contrast', 'success_contrast', 'failure_contrast'],
                  inplace=True)

        # Fit model
        choice[choice == -1] = 0
        choice = choice[signed_contrast == 0]
        clf = LogisticRegression()
        clf.fit(data.values, choice)
        coef = clf.coef_[0]

        # Get coefficients for success and failure contrasts
        success = []
        failure = []
        for k in range(PREVIOUS_TRIALS):
            success.append(coef[data.columns.values == 'success-%s' % str(k + 1)][0])
            failure.append(coef[data.columns.values == 'failure-%s' % str(k + 1)][0])
        results = results.append(pd.DataFrame(data={'trial': np.arange(1, PREVIOUS_TRIALS + 1),
                                                    'success': success, 'failure': failure,
                                                    'subject': nickname, 'condition': condition}))

        # Fit lines
        poly = np.polyfit(np.arange(1, PREVIOUS_TRIALS + 1), success, 1)
        success_slope = np.rad2deg(np.arctan(poly[0]))
        poly = np.polyfit(np.arange(1, PREVIOUS_TRIALS + 1), failure, 1)
        failure_slope = np.rad2deg(np.arctan(poly[0]))
        slopes = slopes.append(pd.Series(data={'success': success_slope, 'failure': failure_slope,
                                               'subject': nickname, 'condition': condition}),
                               ignore_index=True)

# %% Plot results
n_subjects = results['subject'].unique().shape[0]
f, ax = plt.subplots(2, n_subjects + 1, figsize=(18, 8))
sns.set(context='paper', font_scale=1.5, style='ticks')

for i, subject in enumerate(results['subject'].unique()):
    sns.lineplot(x='trial', y='success', data=results[results['subject'] == subject],
                 hue='condition', ci=68, palette='Dark2', ax=ax[0, i])
    ax[0, i].set(ylabel='Weight of success term', title=subject,
                 xlabel='Trials in the past')

sns.lineplot(x='trial', y='success', data=results, hue='condition', ci=68, palette='Dark2',
             ax=ax[0, -1])
ax[0, -1].set(ylabel='Weight of success term',
              title='All mice', xlabel='Trials in the past')

for i, subject in enumerate(results['subject'].unique()):
    sns.lineplot(x='trial', y='failure', data=results[results['subject'] == subject],
                 hue='condition', ci=68, palette='Dark2', ax=ax[1, i])
    ax[1, i].set(ylabel='Weight of failure term',
                 title=subject, xlabel='Trials in the past')

sns.lineplot(x='trial', y='failure', data=results, hue='condition', ci=68, palette='Dark2',
             ax=ax[1, -1])
ax[1, -1].set(ylabel='Weight of failure term',
              title='All mice', xlabel='Trials in the past')

plt.tight_layout()
sns.despine(trim=True)
if SAVE_PLOT:
    plt.savefig(join(FIG_PATH, 'logistic_model_coefficients'))

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
colors = sns.color_palette(n_colors=3)

for i, subject in enumerate(slopes['subject'].unique()):
    ax1.plot([slopes.loc[(slopes['condition'] == 'Pre-vehicle')
                         & (slopes['subject'] == subject), 'success'],
              slopes.loc[(slopes['condition'] == 'Drug')
                         & (slopes['subject'] == subject), 'success']],
             'o-', lw=2, color=colors[i], label=subject)
ax1.set(ylabel='Slope', xticks=[0, 1], xticklabels=['Pre-vehicle', 'Drug'], title='Successes')

for i, subject in enumerate(slopes['subject'].unique()):
    ax2.plot([slopes.loc[(slopes['condition'] == 'Pre-vehicle')
                         & (slopes['subject'] == subject), 'failure'],
              slopes.loc[(slopes['condition'] == 'Drug')
                         & (slopes['subject'] == subject), 'failure']],
             'o-', lw=2, color=colors[i], label=subject)
ax2.set(ylabel='Slope', xticks=[0, 1], xticklabels=['Pre-vehicle', 'Drug'], title='Failures')

if SAVE_PLOT:
    plt.savefig(join(FIG_PATH, 'logistic_model_slopes'))