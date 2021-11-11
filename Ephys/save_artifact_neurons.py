#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 11:33:15 2021
By: Guido Meijer
"""

from os.path import join
import pandas as pd

REPO_PATH = '/home/guido/Repositories/IBL-Serotonin'
artifact_neurons = pd.DataFrame()

# ZFM-02180, 2021-05-21
artifact_neurons = artifact_neurons.append(pd.DataFrame(data={
    'subject': 'ZFM-02180', 'date': '2021-05-21', 'eid': 'f56ef419-f7ef-4ecc-98ec-1257457c57e7', 'probe': 'probe00',
    'neuron_id': [65, 269, 389, 388, 283, 284, 354, 356, 383, 436, 459, 161, 430]}))

# ZFM-02600, 2021-08-26
artifact_neurons = artifact_neurons.append(pd.DataFrame(data={
    'subject': 'ZFM-02600', 'date': '2021-05-26', 'eid': 'c8f68575-e5a8-4960-a690-0149c5c4683f', 'probe': 'probe00',
    'neuron_id': [333, 365, 364]}))
artifact_neurons = artifact_neurons.append(pd.DataFrame(data={
    'subject': 'ZFM-02600', 'date': '2021-05-26', 'eid': 'c8f68575-e5a8-4960-a690-0149c5c4683f', 'probe': 'probe01',
    'neuron_id': [557]}))

# ZFM-02600, 2021-08-27
artifact_neurons = artifact_neurons.append(pd.DataFrame(data={
    'subject': 'ZFM-02600', 'date': '2021-05-27', 'eid': '0d24afce-9d3c-449e-ac9f-577eefefbd7e', 'probe': 'probe00',
    'neuron_id': [489]}))
artifact_neurons = artifact_neurons.append(pd.DataFrame(data={
    'subject': 'ZFM-02600', 'date': '2021-05-27', 'eid': '0d24afce-9d3c-449e-ac9f-577eefefbd7e', 'probe': 'probe01',
    'neuron_id': [290, 291]}))

artifact_neurons.to_csv(join(REPO_PATH, 'artifact_neurons.csv'), index=False)

