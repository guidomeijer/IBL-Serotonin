#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 11:33:15 2021
By: Guido Meijer
"""

from os.path import join
import pandas as pd

REPO_PATH = '/home/guido/Repositories/IBL-Serotonin'
#REPO_PATH = 'C:\\Users\\guido\\Repositories\\IBL-Serotonin'
artifact_neurons = pd.DataFrame()

# ZFM-02180, 2021-05-21
artifact_neurons = artifact_neurons.append(pd.DataFrame(data={
    'subject': 'ZFM-02180', 'date': '2021-05-18', 'pid': '4ad893e9-e699-417e-9559-b0014a8cedb9', 'probe': 'probe00',
    'neuron_id': [841]}))

# ZFM-02180, 2021-05-21
artifact_neurons = artifact_neurons.append(pd.DataFrame(data={
    'subject': 'ZFM-02180', 'date': '2021-05-21', 'pid': '9c2db5c8-8f90-406c-a42b-4f2bbcb7ce21', 'probe': 'probe00',
    'neuron_id': [65, 269, 389, 388, 283, 284, 354, 356, 383, 436, 459, 161, 430]}))

# ZFM-02600, 2021-08-26
artifact_neurons = artifact_neurons.append(pd.DataFrame(data={
    'subject': 'ZFM-02600', 'date': '2021-08-26', 'pid': '382684fb-7088-4d4c-83ca-486e93c342f0', 'probe': 'probe00',
    'neuron_id': [19, 20, 30, 31, 68, 69, 94, 104, 105, 255, 256, 287]}))
artifact_neurons = artifact_neurons.append(pd.DataFrame(data={
    'subject': 'ZFM-02600', 'date': '2021-08-26', 'pid': 'eead949d-45e5-46ae-9b1d-646b280b4ecf', 'probe': 'probe01',
    'neuron_id': [0, 24, 364, 557, 625, 627, 628]}))

# ZFM-02600, 2021-08-27
artifact_neurons = artifact_neurons.append(pd.DataFrame(data={
    'subject': 'ZFM-02600', 'date': '2021-08-27', 'pid': 'ff02e36b-95e9-4985-a6e3-ba2977063496', 'probe': 'probe00',
    'neuron_id': [382, 406]}))
artifact_neurons = artifact_neurons.append(pd.DataFrame(data={
    'subject': 'ZFM-02600', 'date': '2021-08-27', 'pid': 'cc8dd669-7938-4594-868d-b8ca0663b69a', 'probe': 'probe01',
    'neuron_id': [3, 4, 260, 261]}))

# ZFM-02601, 2021-11-19
artifact_neurons = artifact_neurons.append(pd.DataFrame(data={
    'subject': 'ZFM-02601', 'date': '2021-11-19', 'pid': 'dd5af9d9-16b1-477a-885d-713dc5066b8c', 'probe': 'probe00',
    'neuron_id': [23, 51, 109, 120, 201, 202, 203, 205, 206, 207, 208, 209, 210, 220, 232, 234, 242, 244, 248, 253, 263]}))

# ZFM-03330, 2022-02-16
artifact_neurons = artifact_neurons.append(pd.DataFrame(data={
    'subject': 'ZFM-03330', 'date': '2022-02-16', 'pid': '455d9684-12b3-4c40-97a3-5a47b9c31589', 'probe': 'probe01',
    'neuron_id': [185]}))

# ZFM-03330, 2022-02-17
artifact_neurons = artifact_neurons.append(pd.DataFrame(data={
    'subject': 'ZFM-03330', 'date': '2022-02-17', 'pid': '7a82c06b-0e33-454b-a98f-786a4024c1d0', 'probe': 'probe00',
    'neuron_id': [170, 236, 241, 242]}))
artifact_neurons = artifact_neurons.append(pd.DataFrame(data={
    'subject': 'ZFM-03330', 'date': '2022-02-17', 'pid': 'f07d1d05-72a9-4599-8970-bce441c9d21c', 'probe': 'probe01',
    'neuron_id': [1, 106, 131, 132]}))

artifact_neurons.to_csv(join(REPO_PATH, 'artifact_neurons.csv'), index=False)

