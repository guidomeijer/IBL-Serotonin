#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 09:03:38 2021

@author: guido
"""
from serotonin_functions import load_exp_smoothing_trials
import pandas as pd
import numpy as np
from oneibl.one import ONE
from models.expSmoothing_prevAction import expSmoothing_prevAction as exp_prev_action
one = ONE()

sessions = pd.read_csv('pharmacology_sessions.csv', header=1)

results_df = pd.DataFrame()
for i, nickname in enumerate(sessions['Nickname'].unique()):
    # Pre-vehicle
    eids = []
    for s, date in enumerate(sessions.loc[sessions['Nickname'] == nickname, 'Pre-vehicle']):
        eid = one.search(subject=nickname, date_range=date, task_protocol='biased')
        eids.append(eid[0])

    # Load trials
    actions, stimuli, stim_side, eids = load_exp_smoothing_trials(eids, one)

    # Fit model
    model = exp_prev_action('./model_fit_results/', eids, nickname, actions, stimuli, stim_side)
    model.load_or_train(nb_steps=2000, remove_old=False)
    params = model.get_parameters(parameter_type='posterior_mean')
    tau_pre = 1/params[0]

    # Drug
    eids = []
    for s, date in enumerate(sessions.loc[sessions['Nickname'] == nickname, 'Drug']):
        eid = one.search(subject=nickname, date_range=date, task_protocol='biased')
        eids.append(eid[0])

    # Load trials
    actions, stimuli, stim_side, eids = load_exp_smoothing_trials(eids, one)

    # Fit model
    model = exp_prev_action('./model_fit_results/', eids, nickname, actions, stimuli, stim_side)
    model.load_or_train(nb_steps=2000, remove_old=False)
    params = model.get_parameters(parameter_type='posterior_mean')
    tau_drug = 1/params[0]

    # Post-vehicle
    eids = []
    for s, date in enumerate(sessions.loc[sessions['Nickname'] == nickname, 'Post-vehicle']):
        eid = one.search(subject=nickname, date_range=date, task_protocol='biased')
        eids.append(eid[0])

    # Load trials
    actions, stimuli, stim_side, eids = load_exp_smoothing_trials(eids, one)

    # Fit model
    model = exp_prev_action('./model_fit_results/', eids, nickname, actions, stimuli, stim_side)
    model.load_or_train(nb_steps=2000, remove_old=False)
    params = model.get_parameters(parameter_type='posterior_mean')
    tau_post = 1/params[0]

    # Add to dataframe
    results_df = results_df.append(pd.DataFrame(data={'tau': [tau_pre, tau_drug, tau_post],
                                                      'Subject': nickname,
                                                      'condition': ['Pre-vehicle', 'Drug', 'Post-vehicle']}))