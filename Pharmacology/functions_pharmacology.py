#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 10:22:07 2020

@author: guido
"""

from oneibl.one import ONE
import numpy as np
import pandas as pd
from scipy.stats import sem
from psytrack.hyperOpt import hyperOpt
import matplotlib.pyplot as plt
import seaborn as sns
from my_functions import load_opto_trials
import statsmodels.api as sm
from ibl_pipeline.utils import psychofit as psy
from models.expSmoothing_prevAction import expSmoothing_prevAction as exp_prev_action
from models.expSmoothing_stimside import expSmoothing_stimside as exp_stimside


def paths():
    data_path = '/home/guido/Data/5HT'
    fig_path = '/home/guido/Figures/5HT/Pharmacology'
    save_path = '/home/guido/Data/5HT'
    return data_path, fig_path, save_path


def load_session_one(nickname, date):
    # Find session in ONE
    one = ONE()
    d_types = ['_iblrig_taskSettings.raw',
               'trials.probabilityLeft',
               'trials.contrastLeft',
               'trials.contrastRight',
               'trials.choice',
               'trials.feedbackType']
    eid = one.search(subject=nickname, date_range=date, task_protocol='biased')
    if len(eid) != 1:
        raise Exception('Error loading session')
    d, prob_l, contrast_l, contrast_r, choice, correct = one.load(eid[0], d_types,
                                                                  dclass_output=False)

    # Exclude ommisions
    contrast_l = contrast_l[choice != 0]
    contrast_r = contrast_r[choice != 0]
    prob_l = prob_l[choice != 0]
    correct = correct[choice != 0]
    choice = choice[choice != 0]

    return contrast_l, contrast_r, prob_l, correct, choice


def fit_running_avg_model(nickname, date, remove_old_fit=False, one=None):
    if one is None:
        one = ONE()
    eid = one.search(subject=nickname, date_range=date, task_protocol='biased')
    if len(eid) != 1:
        raise Exception('Error loading session')
    trials = load_opto_trials(eid[0], download=True, invert_choice=True)
    model = exp_stimside('./model_fit_results/', eid, '%s_%s' % (nickname, date),
                         np.array(trials['choice'].values),
                         np.array(trials['signed_contrast'].values),
                         np.array(trials['stim_side'].values))
    model.load_or_train(nb_steps=2000, remove_old=remove_old_fit)
    params = model.get_parameters(parameter_type='posterior_mean')
    return 1 / params[0]  # tau parameter


def fit_psychfunc(stim_levels, n_trials, proportion):
    # Fit a psychometric function with two lapse rates
    #
    # Returns vector pars with [threshold, bias, lapselow, lapsehigh]

    assert(stim_levels.shape == n_trials.shape == proportion.shape)

    pars, _ = psy.mle_fit_psycho(np.vstack((stim_levels, n_trials, proportion)),
                                 P_model='erf_psycho_2gammas',
                                 parstart=np.array([0, 20, 0.05, 0.05]),
                                 parmin=np.array([-100, 2, 0, 0]),
                                 parmax=np.array([100, 100., 1, 1]))
    return pars


def plot_psychometric(stim_levels, n_trials, proportion, ax, **kwargs):
    assert stim_levels.ndim == 1
    assert(n_trials.shape == proportion.shape)

    if proportion.ndim > 1:
        pars = fit_psychfunc(stim_levels, np.mean(n_trials, axis=0), np.mean(proportion, axis=0))
    else:
        pars = fit_psychfunc(stim_levels, n_trials, proportion)

    # plot psychfunc
    g = sns.lineplot(np.arange(-29, 29),
                     psy.erf_psycho_2gammas(pars, np.arange(-29, 29)),
                     ax=ax, **kwargs)

    # plot psychfunc: -100, +100
    sns.lineplot(np.arange(-37, -32),
                 psy.erf_psycho_2gammas(pars, np.arange(-103, -98)),
                 ax=ax, **kwargs)
    sns.lineplot(np.arange(32, 37),
                 psy.erf_psycho_2gammas(pars, np.arange(98, 103)),
                 ax=ax, **kwargs)

    # now break the x-axis
    stim_levels[stim_levels == -100] = -35
    stim_levels[stim_levels == 100] = 35

    # plot datapoints with errorbars on top
    if proportion.ndim > 1:
        ax.errorbar(stim_levels, np.mean(proportion, axis=0), yerr=sem(proportion),
                    fmt='o', **kwargs)

    g.set_xticks([-35, -25, -12.5, 0, 12.5, 25, 35])
    g.set_xticklabels(['-100', '-25', '-12.5', '0', '12.5', '25', '100'],
                      size='small', rotation=45)
    g.set_xlim([-40, 40])
    g.set_ylim([0, 1])
    g.set_yticks([0, 0.25, 0.5, 0.75, 1])
    g.set_yticklabels(['0', '25', '50', '75', '100'])


def fit_psytrack(nickname, date, previous_trials=0):

    # Load data
    contrast_l, contrast_r, prob_l, correct, choice = load_session_one(nickname, date)

    # Change values to what the model input
    choice[choice == 1] = 2
    choice[choice == -1] = 1
    correct[correct == -1] = 0
    contrast_l[np.isnan(contrast_l)] = 0
    contrast_r[np.isnan(contrast_r)] = 0

    # Transform visual contrast
    p = 3.5
    contrast_l_transform = np.tanh(contrast_l * p) / np.tanh(p)
    contrast_r_transform = np.tanh(contrast_r * p) / np.tanh(p)

    # Reformat the stimulus vectors to matrices which include previous trials
    s1_trans = contrast_l_transform
    s2_trans = contrast_r_transform
    for i in range(1, 10):
        s1_trans = np.column_stack((s1_trans, np.append([contrast_l_transform[0]]*(i+i),
                                                        contrast_l_transform[i:-i])))
        s2_trans = np.column_stack((s2_trans, np.append([contrast_r_transform[0]]*(i+i),
                                                        contrast_r_transform[i:-i])))

    # Create input dict
    D = {'name': nickname,
         'y': choice,
         'correct': correct,
         'dayLength': choice.shape[0],
         'inputs': {'s1': s1_trans, 's2': s2_trans}
         }

    # Model parameters
    weights = {'bias': 1,
               's1': previous_trials+1,
               's2': previous_trials+1}
    K = np.sum([weights[i] for i in weights.keys()])
    hyper = {'sigInit': 2**4.,
             'sigma': [2**-4.]*K,
             'sigDay': None}
    optList = ['sigInit', 'sigma']

    # Fit model
    print('Fitting model..')
    hyp, evd, wMode, hess = hyperOpt(D, hyper, weights, optList)

    return wMode, prob_l, hyp


def fit_psytrack_multiple_days(nickname, dates, previous_trials=0):

    # Load data
    for i, date in enumerate(dates):
        if i == 0:
            contrast_l, contrast_r, prob_l, correct, choice = load_session_one(nickname, date)
            day_length = contrast_l.shape[0]
        else:
            c_l, c_r, p_l, cr, ch = load_session_one(nickname, date)
            contrast_l = np.append(contrast_l, c_l)
            contrast_r = np.append(contrast_r, c_r)
            prob_l = np.append(prob_l, p_l)
            correct = np.append(correct, cr)
            choice = np.append(choice, ch)
            day_length = np.append(day_length, contrast_l.shape[0])

    # Change values to what the model input
    choice[choice == 1] = 2
    choice[choice == -1] = 1
    correct[correct == -1] = 0
    contrast_l[np.isnan(contrast_l)] = 0
    contrast_r[np.isnan(contrast_r)] = 0

    # Transform visual contrast
    p = 3.5
    contrast_l_transform = np.tanh(contrast_l * p) / np.tanh(p)
    contrast_r_transform = np.tanh(contrast_r * p) / np.tanh(p)

    # Reformat the stimulus vectors to matrices which include previous trials
    s1_trans = contrast_l_transform
    s2_trans = contrast_r_transform
    for i in range(1, 10):
        s1_trans = np.column_stack((s1_trans, np.append([contrast_l_transform[0]]*(i+i),
                                                        contrast_l_transform[i:-i])))
        s2_trans = np.column_stack((s2_trans, np.append([contrast_r_transform[0]]*(i+i),
                                                        contrast_r_transform[i:-i])))

    # Create input dict
    D = {'name': nickname,
         'y': choice,
         'correct': correct,
         'dayLength': day_length,
         'inputs': {'s1': s1_trans, 's2': s2_trans}
         }

    # Model parameters
    weights = {'bias': 1,
               's1': previous_trials+1,
               's2': previous_trials+1}
    K = np.sum([weights[i] for i in weights.keys()])
    hyper = {'sigInit': 2**4.,
             'sigma': [2**-4.]*K,
             'sigDay': [2**-4.]*K}
    optList = ['sigInit', 'sigma', 'sigDay']

    # Fit model
    print('Fitting model..')
    hyp, evd, wMode, hess = hyperOpt(D, hyper, weights, optList)

    return wMode, prob_l, hyp


def plot_psytrack(wMode, prob_l, plot_stim=True):

    f, ax1 = plt.subplots(1, 1, figsize=(10, 5))
    block_switch = np.where(np.abs(np.diff(prob_l)) > 0.1)[0]
    block_switch = np.concatenate(([0], block_switch+1, [np.size(prob_l)]), axis=0)
    for i, ind in enumerate(block_switch[:-1]):
        if prob_l[block_switch[i]] == 0.5:
            ax1.fill([block_switch[i], block_switch[i], block_switch[i+1], block_switch[i+1]],
                     [-4, 4, 4, -4], color=[0.7, 0.7, 0.7])
        if prob_l[block_switch[i]] == 0.2:
            ax1.fill([block_switch[i], block_switch[i], block_switch[i+1], block_switch[i+1]],
                     [-4, 4, 4, -4], color=[0.6, 0.6, 1])
        if prob_l[block_switch[i]] == 0.8:
            ax1.fill([block_switch[i], block_switch[i], block_switch[i+1], block_switch[i+1]],
                     [-4, 4, 4, -4], color=[1, 0.6, 0.6])
    ax1.plot(wMode[0], color='k', lw=3)
    if plot_stim is True:
        ax1.plot(wMode[1], color='r', lw=3)
        ax1.plot(wMode[2], color='b', lw=3)
    ax1.legend(['Bias', 'Left stimulus', 'Right stimulus'], fontsize=12)
    ax1.set(ylabel='Weight', xlabel='Trials')
    sns.set(context='paper', font_scale=1.5, style='ticks')
    sns.despine(trim=True)
    plt.tight_layout(pad=2)
    return ax1


def fit_probabilistic_choice_model(nickname, date, previous_trials=6):

    # Load data
    contrast_l, contrast_r, prob_l, correct, choice = load_session_one(nickname, date)

    # Make dataframe
    signed_contrast = np.copy(contrast_l)
    signed_contrast[np.isnan(signed_contrast)] = -contrast_r[~np.isnan(contrast_r)]
    data = pd.DataFrame(data={'choice': choice, 'correct': correct,
                              'signed_contrast': signed_contrast, 'prop_left': prob_l})

    # Rewardeded choices:
    data.loc[(data['choice'] == 0) & (data['correct'].isnull()), 'rchoice'] = 0  # NoGo trials
    data.loc[(data['choice'] == -1) & (data['correct'] == -1), 'rchoice'] = 0
    data.loc[(data['choice'] == -1) & (data['correct'] == 1), 'rchoice'] = -1
    data.loc[(data['choice'] == 1) & (data['correct'] == 1), 'rchoice'] = 1
    data.loc[(data['choice'] == 0) & (data['correct'].isnull()), 'rchoice'] = 0  # NoGo trials
    data.loc[(data['choice'] == 1) & (data['correct'] == -1), 'rchoice'] = 0

    # Unrewarded choices:
    data.loc[(data['choice'] == 0) & (data['correct'].isnull()), 'uchoice'] = 0  # NoGo trials
    data.loc[(data['choice'] == -1) & (data['correct'] == -1), 'uchoice'] = -1
    data.loc[(data['choice'] == -1) & (data['correct'] == 1), 'uchoice'] = 0
    data.loc[(data['choice'] == 1) & (data['correct'] == 1), 'uchoice'] = 0
    data.loc[(data['choice'] == 0) & (data['correct'].isnull()), 'uchoice'] = 0  # NoGo trials
    data.loc[(data['choice'] == 1) & (data['correct'] == -1), 'uchoice'] = 1

    # Shift rewarded and unrewarded predictors by one
    for i in range(previous_trials):
        data.loc[:, 'rchoice-%s' % str(i+1)] = data['rchoice'].shift(
                                                    periods=i+1, fill_value=0).to_numpy()
        data.loc[:, 'uchoice-%s' % str(i+1)] = data['uchoice'].shift(
                                                    periods=i+1, fill_value=0).to_numpy()

    # Drop any nan trials
    data.dropna(inplace=True)

    # Make sensory predictors (no 0 predictor)
    contrasts = [.25, 1, .125, .0625]
    for i in contrasts:
        data.loc[(data['signed_contrast'].abs() == i), i] = np.sign(
                    data.loc[(data['signed_contrast'].abs() == i), 'signed_contrast'].to_numpy())
        data[i].fillna(0,  inplace=True)

    # Add block probability predictor
    data.loc[(data['prop_left'] == 0.5), 'block'] = 0
    data.loc[(data['prop_left'] == 0.2), 'block'] = -1
    data.loc[(data['prop_left'] == 0.8), 'block'] = 1

    # Make choice in between 0 and 1 -> 1 for right and 0 for left
    data.loc[data['choice'] == -1, 'choice'] = 0

    # Create predictor matrix
    exog = data.copy()
    exog.drop(columns=['correct', 'signed_contrast', 'choice', 'prop_left', 'rchoice', 'uchoice'],
              inplace=True)
    exog = sm.add_constant(exog)

    # Fit model
    logit_model = sm.Logit(data['choice'], exog)
    result = logit_model.fit()
    weights = result.params.rename(index=str)
    return weights
