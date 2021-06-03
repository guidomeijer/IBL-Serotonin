#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 13:41:04 2021

@author: guido
"""

import numpy as np
from os.path import join
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from my_functions import paths

# Settings
HIST_PATH = '/home/guido/Histology'
REGIONS = ['Periaqueductal gray', 'Midbrain reticular nucleus', 'Dorsal nucleus raphe']
_, fig_path, _ = paths()
fig_path = join(fig_path, '5HT', 'opto-behavior')

# Load in subjects
subjects = pd.read_csv('subjects.csv')

results = pd.DataFrame()
for i, subject in enumerate(subjects['subject']):
    try:
        this_data = pd.read_csv(join(HIST_PATH, subject, 'cellfinder', 'analysis', 'summary.csv'))
        this_data['subject'] = subject
        this_data['sert_cre'] = subjects.loc[subjects['subject'] == subject, 'sert-cre'].values[0]
        results = results.append(this_data.loc[this_data['structure_name'].isin(REGIONS)])
    except:
        print(f'Could not load histology for {subject}')


