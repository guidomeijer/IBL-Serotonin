#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 13:17:10 2022
By: Guido Meijer
"""

from one.api import ONE

subjects = ['ZFM-04820']
project_name = 'serotonin_inference'
one = ONE(base_url='https://alyx.internationalbrainlab.org')

# queries for the subjects sessins above that don't have the project labeled
sessions = one.alyx.rest('sessions', 'list', django=f'subject__nickname__in,{subjects},~projects__name,{project_name}', use_cache=False)
for ses in sessions:
    print(ses['subject'], ses['start_time'][:10], ses['number'], ses['projects'])
    one.alyx.rest('sessions', 'partial_update', id=ses['id'], data={'projects': [project_name]})

# after this operation, the query returns empty results
sessions = one.alyx.rest('sessions', 'list', django=f'subject__nickname__in,{subjects},~projects__name,{project_name}', use_cache=False)
assert(len(sessions) == 0)