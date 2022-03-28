#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 10:12:54 2022
By: Guido Meijer
"""

import pandas as pd
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import networkx as nx
from serotonin_functions import paths, load_subjects

fig_path, save_path = paths()

# Load in data
cca_df = pd.read_csv(join(save_path, 'cca_results.csv'))

# Add expression
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    cca_df.loc[cca_df['subject'] == nickname, 'expression'] = subjects.loc[subjects['subject'] == nickname, 'expression'].values[0]
cca_df = cca_df[cca_df['expression'] == 1]

G = nx.Graph()
G.add_nodes_from(cca_df['region_1'].unique())
for i, region_1 in cca_df['region_1'].unique():
    for j, region_2 in cca_df['region_2'].unique():

        cca_df.loc[(cca_df['region_1'] == region_1)]

