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

# ZFM-03331, 2022-03-11
artifact_neurons = artifact_neurons.append(pd.DataFrame(data={
    'subject': 'ZFM-03331', 'date': '2022-03-11', 'pid': '1b1c4658-050c-447f-a08b-28f668bec84f', 'probe': 'probe00',
    'neuron_id': [31, 32, 73, 79, 192, 217, 221, 276, 371, 318, 369, 370 ,378, 379, 380]}))
artifact_neurons = artifact_neurons.append(pd.DataFrame(data={
    'subject': 'ZFM-03331', 'date': '2022-03-11', 'pid': '465fbc08-3a9f-4714-a0a0-79d76b39ccbc', 'probe': 'probe01',
    'neuron_id': [85, 131, 391, 512, 513, 572, 601, 618]}))

# ZFM-04122, 2022-05-12
artifact_neurons = artifact_neurons.append(pd.DataFrame(data={
    'subject': 'ZFM-04122', 'date': '2022-05-12', 'pid': '09249aa4-006c-4aa9-b86a-215da86e1778', 'probe': 'probe00',
    'neuron_id': [288, 290, 362, 423]}))

# ZFM-03329, 2022-03-02
artifact_neurons = artifact_neurons.append(pd.DataFrame(data={
    'subject': 'ZFM-03329', 'date': '2022-03-02', 'pid': 'd8866f7d-7c34-4592-a042-316b17bfbb19', 'probe': 'probe00',
    'neuron_id': [44, 138, 204]}))

# ZFM-03323, 2022-04-07
artifact_neurons = artifact_neurons.append(pd.DataFrame(data={
    'subject': 'ZFM-03323', 'date': '2022-04-07', 'pid': '56c34de2-8fc4-4f37-979b-094951ab79ae', 'probe': 'probe00',
    'neuron_id': [0, 1, 30, 72, 94, 95, 116, 141, 287, 500, 571]}))
artifact_neurons = artifact_neurons.append(pd.DataFrame(data={
    'subject': 'ZFM-03323', 'date': '2022-04-07', 'pid': '77cb0293-f054-4488-b34e-65869853e20e', 'probe': 'probe01',
    'neuron_id': [253, 380, 534, 535, 625, 667, 739, 848]}))

# ZFM-03323, 2022-04-07
artifact_neurons = artifact_neurons.append(pd.DataFrame(data={
    'subject': 'ZFM-04820', 'date': '2022-09-13', 'pid': 'cf08dda8-478f-4292-a06f-4c4dae9f8755', 'probe': 'probe00',
    'neuron_id': [597]}))

artifact_neurons.to_csv(join(REPO_PATH, 'artifact_neurons.csv'), index=False)

