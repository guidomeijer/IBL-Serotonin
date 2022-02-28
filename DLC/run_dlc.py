#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 10:00:29 2021

@author: guido
"""

from iblvideo import run_session
from one.api import ONE
one = ONE()

"""
sessions = one.alyx.rest('sessions', 'list', django='project__name__icontains,ibl_fiberfluo_pilot_01,'
                                                   '~subject__nickname__icontains,ZFM-03450')
eids = [i['url'][-36:] for i in sessions]
"""

#sessions = one.search(task_protocol='_iblrig_NPH_tasks_', subject='!ZFM-03450')
#sessions = one.search(task_protocol='_iblrig_NPH_tasks_')
#sessions = one.search(project='serotonin_inference', task_protocol='biased')
eids = one.search(project='serotonin_inference', task_protocol='_iblrig_tasks_opto_ephysChoiceWorld')

for i, eid in enumerate(eids):
    print(f'\n\nProcessing session {eid} [{i + 1} of {len(eids)}]\n\n')
    status = run_session(eid, machine='guido', cams=['left', 'right', 'body'], one=one, overwrite=False)

"""
status = run_session('315349d8-eb22-4c9b-871d-833f875e6641', machine='guido', cams=['left'],
                     one=one, frames=10000, clobber=True, overwrite=True)
"""
