#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 13:42:14 2022
By: Guido Meijer
"""

import numpy as np
from sklearn import preprocessing
import numpy.random as npr
import os
import json
from collections import defaultdict
from os.path import join, isdir
from sklearn import preprocessing
from serotonin_functions import (paths, behavioral_criterion, query_opto_sessions, load_subjects)
from glm_hmm_utils import get_all_unnormalized_data_this_session, fit_glm, create_train_test_sessions
from one.api import ONE
one = ONE()

"""
Adapted from on Zoe Ashwood's code (https://github.com/zashwood/glm-hmm)
"""

# Settings
N_FOLDS = 5
MIN_SESSIONS = 2

npr.seed(42)

# Paths
_, save_path = paths()
save_path = join(save_path, 'GLM-HMM')

# Create folders
if not isdir(save_path):
    os.mkdir(save_path)
if not isdir(join(save_path, 'data_by_animal')):
    os.mkdir(join(save_path, 'data_by_animal'))

# Query which subjects to use and create eid list per subject
subjects = load_subjects()
animal_list = subjects['subject'].values
animal_eid_dict = dict()
for i, nickname in enumerate(subjects['subject']):

    # Query sessions
    eids = query_opto_sessions(nickname, include_ephys=True, one=one)
    eids = behavioral_criterion(eids, max_lapse=1.1, max_bias=1.1, min_trials=100, one=one)

    # animal_eid_dict is a dict with subjects as keys and a list of eids per subject
    animal_eid_dict[nickname] = eids

# Require that each animal has enough sessions
for animal in animal_list:
    num_sessions = len(animal_eid_dict[animal])
    if num_sessions < MIN_SESSIONS:
        animal_list = np.delete(animal_list,
                                np.where(animal_list == animal))

# %%
# Identify idx in master array where each animal's data starts and ends:
animal_start_idx = {}
animal_end_idx = {}

final_animal_eid_dict = defaultdict(list)
# WORKHORSE: iterate through each animal and each animal's set of eids;
# obtain unnormalized data.  Write out each animal's data and then also
# write to master array
for z, animal in enumerate(animal_list):
    sess_counter = 0
    for eid in animal_eid_dict[animal]:
        try:
            animal, unnormalized_inpt, y, session, num_viols_50, rewarded = \
                get_all_unnormalized_data_this_session(
                    eid, one)
            if num_viols_50 < 10:  # only include session if number of viols is less than 10
                if sess_counter == 0:
                    animal_unnormalized_inpt = np.copy(unnormalized_inpt)
                    animal_y = np.copy(y)
                    animal_session = session
                    animal_rewarded = np.copy(rewarded)
                else:
                    animal_unnormalized_inpt = np.vstack(
                        (animal_unnormalized_inpt, unnormalized_inpt))
                    animal_y = np.vstack((animal_y, y))
                    animal_session = np.concatenate((animal_session, session))
                    animal_rewarded = np.vstack((animal_rewarded, rewarded))
                sess_counter += 1
                final_animal_eid_dict[animal].append(eid)
        except Exception as err:
            print(err)
    # Write out animal's unnormalized data matrix:
    np.savez(join(save_path, 'data_by_animal', animal + '_unnormalized.npz'),
             animal_unnormalized_inpt, animal_y, animal_session)
    animal_session_fold_lookup = create_train_test_sessions(animal_session, N_FOLDS)
    np.savez(join(save_path, 'data_by_animal', animal + "_session_fold_lookup.npz"),
             animal_session_fold_lookup)
    np.savez(join(save_path, 'data_by_animal', animal + '_rewarded.npz'),
             animal_rewarded)
    assert animal_rewarded.shape[0] == animal_y.shape[0]
    # Now create or append data to master array across all animals:
    if z == 0:
        master_inpt = np.copy(animal_unnormalized_inpt)
        animal_start_idx[animal] = 0
        animal_end_idx[animal] = master_inpt.shape[0] - 1
        master_y = np.copy(animal_y)
        master_session = animal_session
        master_session_fold_lookup_table = animal_session_fold_lookup
        master_rewarded = np.copy(animal_rewarded)
    else:
        animal_start_idx[animal] = master_inpt.shape[0]
        master_inpt = np.vstack((master_inpt, animal_unnormalized_inpt))
        animal_end_idx[animal] = master_inpt.shape[0] - 1
        master_y = np.vstack((master_y, animal_y))
        master_session = np.concatenate((master_session, animal_session))
        master_session_fold_lookup_table = np.vstack(
            (master_session_fold_lookup_table, animal_session_fold_lookup))
        master_rewarded = np.vstack((master_rewarded, animal_rewarded))
# Write out data from across animals
assert np.shape(master_inpt)[0] == np.shape(master_y)[
    0], "inpt and y not same length"
assert np.shape(master_rewarded)[0] == np.shape(master_y)[
    0], "rewarded and y not same length"
assert len(np.unique(master_session)) == \
       np.shape(master_session_fold_lookup_table)[
           0], "number of unique sessions and session fold lookup don't " \
               "match"
normalized_inpt = np.copy(master_inpt)
normalized_inpt[:, 0] = preprocessing.scale(normalized_inpt[:, 0])
np.savez(join(save_path, 'all_animals_concat.npz'),
         normalized_inpt, master_y, master_session)
np.savez(join(save_path, 'all_animals_concat_unnormalized.npz'),
              master_inpt, master_y, master_session)
np.savez(join(save_path, 'all_animals_concat_session_fold_lookup.npz'),
         master_session_fold_lookup_table)
np.savez(join(save_path, 'all_animals_concat_rewarded.npz'),
         master_rewarded)
np.savez(join(save_path, 'data_by_animal', 'animal_list.npz'),
         animal_list)

json_dump = json.dumps(final_animal_eid_dict)
f = open(join(save_path, "final_animal_eid_dict.json"), "w")
f.write(json_dump)
f.close()

# Now write out normalized data (when normalized across all animals) for
# each animal:
counter = 0
for animal in animal_start_idx.keys():
    start_idx = animal_start_idx[animal]
    end_idx = animal_end_idx[animal]
    inpt = normalized_inpt[range(start_idx, end_idx + 1)]
    y = master_y[range(start_idx, end_idx + 1)]
    session = master_session[range(start_idx, end_idx + 1)]
    counter += inpt.shape[0]
    np.savez(join(save_path, 'data_by_animal', animal + '_processed.npz'),
             inpt, y, session)

assert counter == master_inpt.shape[0]