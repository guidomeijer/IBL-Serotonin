#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 10:51:33 2022
By: Guido Meijer
"""

from os.path import join
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from serotonin_functions import paths, load_subjects

# Load in data
save_path, fig_path = paths()
lda_opto_df = pd.read_csv(join(save_path, 'lda_task_events_opto.csv'))

# Add expression
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    lda_opto_df.loc[lda_opto_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]


