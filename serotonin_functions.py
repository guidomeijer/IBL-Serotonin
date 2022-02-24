1# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:22:01 2020

@author: guido
"""

import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import pandas as pd
import tkinter as tk
import pathlib
import patsy
import statsmodels.api as sm
from sklearn.model_selection import KFold
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from os.path import join, realpath, dirname
from glob import glob
from datetime import datetime
from brainbox.io.spikeglx import spikeglx
from iblutil.numerical import ismember
from ibllib.atlas import BrainRegions
from ibllib.atlas import AllenAtlas
from one.api import ONE

# This is the date at which I fixed a crucial bug in the laser driver, all optogenetics data
# from before this date is not to be trusted
DATE_GOOD_OPTO = '2021-07-01'

# This is the date at which light shielding was added so that the mouse couldn't see the opto stim
DATE_LIGHT_SHIELD = '2021-06-08'


def load_subjects(behavior=None):
    subjects = pd.read_csv(join(pathlib.Path(__file__).parent.resolve(), 'subjects.csv'))
    subjects = subjects[~((subjects['expression'] == 0) & (subjects['sert-cre'] == 1))]
    if behavior:
        subjects = subjects[subjects['include_behavior'] == 1]
    subjects = subjects.reset_index(drop=True)
    return subjects


def paths():
    """
    Make a file in the root of the repository called 'serotonin_paths.py' with in it:

    DATA_PATH = '/path/to/Flatiron/data'
    FIG_PATH = '/path/to/save/figures'
    SAVE_PATH = '/path/to/save/data'

    """
    from serotonin_paths import FIG_PATH
    save_path = join(dirname(realpath(__file__)), 'Data')
    return FIG_PATH, save_path


def figure_style():
    """
    Set style for plotting figures
    """
    sns.set(style="ticks", context="paper",
            font="Arial",
            rc={"font.size": 7,
                 "axes.titlesize": 8,
                 "axes.labelsize": 8,
                 "axes.linewidth": 0.5,
                 "lines.linewidth": 1,
                 "lines.markersize": 3,
                 "xtick.labelsize": 7,
                 "ytick.labelsize": 7,
                 "savefig.transparent": True,
                 "xtick.major.size": 2.5,
                 "ytick.major.size": 2.5,
                 "xtick.major.width": 0.5,
                 "ytick.major.width": 0.5,
                 "xtick.minor.size": 2,
                 "ytick.minor.size": 2,
                 "xtick.minor.width": 0.5,
                 "ytick.minor.width": 0.5,
                 'legend.fontsize': 7,
                 'legend.title_fontsize': 7
                 })
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    colors = {'general': 'orange',
              'sert': sns.color_palette('Dark2')[0],
              'wt': [0.75, 0.75, 0.75],
              'left': sns.color_palette('colorblind')[1],
              'right': sns.color_palette('colorblind')[0],
              'enhanced': sns.color_palette('colorblind')[3],
              'suppressed': sns.color_palette('colorblind')[0],
              'no-modulation': sns.color_palette('colorblind')[7],
              'stim': sns.color_palette('colorblind')[9],
              'no-stim': sns.color_palette('colorblind')[7],
              'probe': sns.color_palette('colorblind')[4],
              'block': sns.color_palette('colorblind')[6],
              'RS': sns.color_palette('Set2')[0],
              'FS': sns.color_palette('Set2')[1],
              'early': [0.7, 0.7, 0.7], 'late': 'k'}
    screen_width = tk.Tk().winfo_screenwidth()
    dpi = screen_width / 12
    return colors, dpi


def get_artifact_neurons():
    artifact_neurons = pd.read_csv(join(pathlib.Path(__file__).parent.resolve(), 'artifact_neurons.csv'))
    return artifact_neurons


def remove_artifact_neurons(df):
    artifact_neurons = pd.read_csv(join(pathlib.Path(__file__).parent.resolve(), 'artifact_neurons.csv'))
    for i, column in enumerate(df.columns):
        if df[column].dtype == bool:
            df[column] = df[column].astype('boolean')
    df = pd.merge(df, artifact_neurons, indicator=True, how='outer',
                  on=['pid', 'neuron_id', 'probe', 'subject', 'date']).query('_merge=="left_only"').drop('_merge', axis=1)
    return df


def query_opto_sessions(subject, one=None):
    one = one or ONE()
    sessions = one.alyx.rest('sessions', 'list', subject=subject,
                             task_protocol='_iblrig_tasks_opto_biasedChoiceWorld',
                             project='serotonin_inference')
    return [sess['url'][-36:] for sess in sessions]


def query_ephys_sessions(selection='aligned', one=None):
    if one is None:
        one = ONE()
    if selection == 'all':
        # Query all opto_ephysChoiceWorld sessions
        ins = one.alyx.rest('insertions', 'list',
                        django='session__project__name__icontains,serotonin_inference,'
                               'session__qc__lt,50')
    elif selection == 'aligned':
        # Query all ephys-histology aligned sessions
        ins = one.alyx.rest('insertions', 'list',
                        django='session__project__name__icontains,serotonin_inference,'
                               'session__qc__lt,50,'
                               'json__extended_qc__alignment_count__gt,0')
    elif selection == 'aligned-behavior':
        # Query sessions with an alignment and that meet behavior criterion
        ins = one.alyx.rest('insertions', 'list',
                        django='session__project__name__icontains,serotonin_inference,'
                               'session__qc__lt,50,'
                               'json__extended_qc__alignment_count__gt,0,'
                               'session__extended_qc__behavior,1')
    else:
        ins = []

    # Only include subjects from subjects.csv
    incl_subjects = load_subjects()
    ins = [i for i in ins if i['session_info']['subject'] in incl_subjects['subject'].values]

    # Get list of eids and probes
    rec = pd.DataFrame()
    rec['pid'] = np.array([i['id'] for i in ins])
    rec['eid'] = np.array([i['session'] for i in ins])
    rec['probe'] = np.array([i['name'] for i in ins])
    rec['subject'] = np.array([i['session_info']['subject'] for i in ins])
    rec['date'] = np.array([i['session_info']['start_time'][:10] for i in ins])
    return rec


def load_trials(eid, laser_stimulation=False, invert_choice=False, invert_stimside=False,
                patch_old_opto=True, one=None):
    one = one or ONE()
    data, _ = one.load_datasets(eid, datasets=[
        '_ibl_trials.stimOn_times.npy', '_ibl_trials.feedback_times.npy',
        '_ibl_trials.goCue_times.npy', '_ibl_trials.probabilityLeft.npy',
        '_ibl_trials.contrastLeft.npy', '_ibl_trials.contrastRight.npy',
        '_ibl_trials.feedbackType.npy', '_ibl_trials.choice.npy',
        '_ibl_trials.firstMovement_times.npy'])
    trials = pd.DataFrame(data=np.vstack(data).T, columns=[
        'stimOn_times', 'feedback_times', 'goCue_times', 'probabilityLeft', 'contrastLeft',
        'contrastRight', 'feedbackType', 'choice', 'firstMovement_times'])
    if trials.shape[0] == 0:
        return
    trials['signed_contrast'] = trials['contrastRight']
    trials.loc[trials['signed_contrast'].isnull(), 'signed_contrast'] = -trials['contrastLeft']
    if laser_stimulation:
        trials['laser_stimulation'] = one.load_dataset(eid, dataset='_ibl_trials.laserStimulation.npy')
        try:
            trials['laser_probability'] = one.load_dataset(eid, dataset='_ibl_trials.laserProbability.npy')
            trials['probe_trial'] = ((trials['laser_stimulation'] == 0) & (trials['laser_probability'] == 0.75)
                                     | (trials['laser_stimulation'] == 1) & (trials['laser_probability'] == 0.25)).astype(int)
        except:
            trials['laser_probability'] = trials['laser_stimulation'].copy()
            trials.loc[(trials['signed_contrast'] == 0)
                       & (trials['laser_stimulation'] == 0), 'laser_probability'] = 0.25
            trials.loc[(trials['signed_contrast'] == 0)
                       & (trials['laser_stimulation'] == 1), 'laser_probability'] = 0.75

    trials['correct'] = trials['feedbackType']
    trials.loc[trials['correct'] == -1, 'correct'] = 0
    trials['right_choice'] = -trials['choice']
    trials.loc[trials['right_choice'] == -1, 'right_choice'] = 0
    trials['stim_side'] = (trials['signed_contrast'] > 0).astype(int)
    trials.loc[trials['stim_side'] == 0, 'stim_side'] = -1
    trials.loc[(trials['signed_contrast'] == 0) & (trials['contrastLeft'].isnull()),
               'stim_side'] = 1
    trials.loc[(trials['signed_contrast'] == 0) & (trials['contrastRight'].isnull()),
               'stim_side'] = -1
    if 'firstMovement_times' in trials.columns.values:
        trials['reaction_times'] = trials['firstMovement_times'] - trials['goCue_times']
    if invert_choice:
        trials['choice'] = -trials['choice']
    if invert_stimside:
        trials['stim_side'] = -trials['stim_side']
        trials['signed_contrast'] = -trials['signed_contrast']

    # Patch datasets that contained a bug in the laser driver code
    ses_date = datetime.strptime(one.get_details(eid)['start_time'][:10], '%Y-%m-%d')
    if ((ses_date < datetime.strptime(DATE_GOOD_OPTO, '%Y-%m-%d'))
            and patch_old_opto and laser_stimulation):

        # The bug flipped laser on and off pulses after a long reaction time trial
        bug_trial = ((trials.loc[trials['laser_stimulation'] == 1, 'feedback_times']
                      - trials.loc[trials['laser_stimulation'] == 1, 'stimOn_times']) > 10).idxmax()
        print(f'Patching buggy opto data, excluding {trials.shape[0] - bug_trial} trials')
        trials = trials[:bug_trial]

    return trials


def combine_regions(acronyms, split_thalamus=True, abbreviate=False):
    regions = np.array(['root'] * len(acronyms), dtype=object)
    if abbreviate:
        regions[np.in1d(acronyms, ['ILA', 'PL', 'MOs', 'ACAd', 'ACAv'])] = 'mPFC'
        regions[np.in1d(acronyms, ['ORBl', 'ORBm'])] = 'ORB'
        if split_thalamus:
            regions[np.in1d(acronyms, ['PO'])] = 'PO'
            regions[np.in1d(acronyms, ['LP'])] = 'LP'
            regions[np.in1d(acronyms, ['LD'])] = 'LD'
            regions[np.in1d(acronyms, ['RT'])] = 'RT'
            regions[np.in1d(acronyms, ['VAL'])] = 'VAL'
        else:
            regions[np.in1d(acronyms, ['PO', 'LP', 'LD', 'RT', 'VAL'])] = 'Thal'
        regions[np.in1d(acronyms, ['SCm', 'SCs', 'SCig', 'SCsg', 'SCdg'])] = 'SC'
        regions[np.in1d(acronyms, ['RSPv', 'RSPd'])] = 'RSP'
        regions[np.in1d(acronyms, ['GPi', 'GPe'])] = 'GP'
        regions[np.in1d(acronyms, ['MRN'])] = 'MRN'
        regions[np.in1d(acronyms, ['ZI'])] = 'ZI'
        regions[np.in1d(acronyms, ['PAG'])] = 'PAG'
        regions[np.in1d(acronyms, ['LGv', 'LGd'])] = 'LG'
        regions[np.in1d(acronyms, ['PIR'])] = 'Pir'
        regions[np.in1d(acronyms, ['SNr', 'SNc', 'SNl'])] = 'SN'
        regions[np.in1d(acronyms, ['VISa', 'VISam'])] = 'PPC'
        regions[np.in1d(acronyms, ['MEA', 'CEA', 'BLA', 'COAa'])] = 'Amyg'
        regions[np.in1d(acronyms, ['AON'])] = 'AON'
        regions[np.in1d(acronyms, ['CP', 'STR', 'STRd', 'STRv'])] = 'Str'
        regions[np.in1d(acronyms, ['CA1', 'CA3', 'DG'])] = 'Hipp'
    else:
        regions[np.in1d(acronyms, ['ILA', 'PL', 'MOs', 'ACAd', 'ACAv'])] = 'Medial prefrontal cortex'
        regions[np.in1d(acronyms, ['ORBl', 'ORBm'])] = 'Orbitofrontal cortex'
        if split_thalamus:
            regions[np.in1d(acronyms, ['PO'])] = 'Thalamus (PO)'
            regions[np.in1d(acronyms, ['LP'])] = 'Thalamus (LP)'
            regions[np.in1d(acronyms, ['LD'])] = 'Thalamus (LD)'
            regions[np.in1d(acronyms, ['RT'])] = 'Thalamus (RT)'
            regions[np.in1d(acronyms, ['VAL'])] = 'Thalamus (VAL)'
        else:
            regions[np.in1d(acronyms, ['PO', 'LP', 'LD', 'RT', 'VAL'])] = 'Thalamus'
        regions[np.in1d(acronyms, ['SCm', 'SCs', 'SCig', 'SCsg', 'SCdg'])] = 'Superior colliculus'
        regions[np.in1d(acronyms, ['RSPv', 'RSPd'])] = 'Retrosplenial'
        regions[np.in1d(acronyms, ['GPi', 'GPe'])] = 'Globus pallidus'
        regions[np.in1d(acronyms, ['MRN'])] = 'Midbrain reticular nucleus'
        regions[np.in1d(acronyms, ['AON'])] = 'Anterior olfactory nucleus'
        regions[np.in1d(acronyms, ['ZI'])] = 'Zona incerta'
        regions[np.in1d(acronyms, ['PAG'])] = 'Periaqueductal gray'
        regions[np.in1d(acronyms, ['LGv', 'LGd'])] = 'Lateral geniculate'
        regions[np.in1d(acronyms, ['PIR'])] = 'Piriform'
        regions[np.in1d(acronyms, ['SNr', 'SNc', 'SNl'])] = 'Substantia nigra'
        regions[np.in1d(acronyms, ['VISa', 'VISam'])] = 'Posterior parietal cortex'
        regions[np.in1d(acronyms, ['MEA', 'CEA', 'BLA', 'COAa'])] = 'Amygdala'
        regions[np.in1d(acronyms, ['CP', 'STR', 'STRd', 'STRv'])] = 'Tail of the striatum'
        regions[np.in1d(acronyms, ['CA1', 'CA3', 'DG'])] = 'Hippocampus'
    return regions


def remap(ids, source='Allen', dest='Beryl', combine=False, split_thalamus=False, abbreviate=True,
          brainregions=None):
    br = brainregions or BrainRegions()
    _, inds = ismember(ids, br.id[br.mappings[source]])
    ids = br.id[br.mappings[dest][inds]]
    acronyms = br.get(br.id[br.mappings[dest][inds]])['acronym']
    if combine:
        return combine_regions(acronyms, split_thalamus=split_thalamus, abbreviate=abbreviate)
    else:
        return acronyms


def get_full_region_name(acronyms):
    brainregions = BrainRegions()
    full_region_names = []
    for i, acronym in enumerate(acronyms):
        try:
            regname = brainregions.name[np.argwhere(brainregions.acronym == acronym).flatten()][0]
            full_region_names.append(regname)
        except IndexError:
            full_region_names.append(acronym)
    if len(full_region_names) == 1:
        return full_region_names[0]
    else:
        return full_region_names


def behavioral_criterion(eids, max_lapse=0.3, max_bias=0.4, min_trials=1, one=None):
    if one is None:
        one = ONE()
    use_eids = []
    for j, eid in enumerate(eids):
        try:
            trials = load_trials(eid, one=one)
            lapse_l = 1 - (np.sum(trials.loc[trials['signed_contrast'] == -1, 'choice'] == 1)
                           / trials.loc[trials['signed_contrast'] == -1, 'choice'].shape[0])
            lapse_r = 1 - (np.sum(trials.loc[trials['signed_contrast'] == 1, 'choice'] == -1)
                           / trials.loc[trials['signed_contrast'] == 1, 'choice'].shape[0])
            bias = np.abs(0.5 - (np.sum(trials.loc[trials['signed_contrast'] == 0, 'choice'] == 1)
                                 / np.shape(trials.loc[trials['signed_contrast'] == 0, 'choice'] == 1)[0]))
            details = one.get_details(eid)
            if ((lapse_l < max_lapse) & (lapse_r < max_lapse) & (trials.shape[0] > min_trials)
                    & (bias < max_bias)):
                use_eids.append(eid)
            else:
                print('%s %s excluded (n_trials: %d, lapse_l: %.2f, lapse_r: %.2f, bias: %.2f)'
                      % (details['subject'], details['start_time'][:10], trials.shape[0], lapse_l, lapse_r, bias))
        except Exception:
            print('Could not load session %s' % eid)
    return use_eids


def load_exp_smoothing_trials(eids, stimulated=None, rt_cutoff=0.2, after_probe_trials=0, stim_trial_shift=0,
                              pseudo=False, patch_old_opto=True, min_trials=100, one=None):
    """
    Parameters
    ----------
    eids : list
        List of eids
    stimulated : None or string
        If None, do not return the stimulated array, if a string these are the options:
            all: all laser stimulated trials
            probe: only laser probe trials
            block: only laser block trials (no probes)
            rt: this is a weird one - return a reaction time cut off as stimulated trials
    rt_cutoff : float
        Only used if stimulated = 'rt'. Reaction time cutoff in seconds above which stimulated is 1
    after_probe_trials : int
        Only used if stimulated = 'probe'. How many trials after a probe trial are still counted.
    pseudo : bool
        Whether to use pseudo stimulated blocks or shuffled probes as control
    """

    if isinstance(stimulated, str):
        assert stimulated in ['all', 'probe', 'block', 'rt']

    if one is None:
        one=ONE()
    stimuli_arr, actions_arr, stim_sides_arr, prob_left_arr, stimulated_arr, session_uuids = [], [], [], [], [], []
    for j, eid in enumerate(eids):
        try:
            # Load in trials vectors
            trials = load_trials(eid, invert_stimside=True, laser_stimulation=True,
                                 patch_old_opto=patch_old_opto, one=one)
            if trials.shape[0] < min_trials:
                continue
            if stimulated == 'all':
                stim_trials = trials['laser_stimulation'].values
            elif stimulated == 'probe':
                stim_trials = ((trials['laser_stimulation'] == 1) & (trials['laser_probability'] <= .5)).values
                if pseudo:
                    stim_trials = shuffle(stim_trials)
                for k, ind in enumerate(np.where(stim_trials == 1)[0]):
                    stim_trials[ind:ind + (after_probe_trials + 1)] = 1
            elif stimulated == 'block':
                stim_trials = trials['laser_stimulation'].values
                if 'laser_probability' in trials.columns:
                    stim_trials[(trials['laser_stimulation'] == 0) & (trials['laser_probability'] == .75)] = 1
                    stim_trials[(trials['laser_stimulation'] == 1) & (trials['laser_probability'] == .25)] = 0
            elif stimulated == 'rt':
                stim_trials = (trials['reaction_times'] > rt_cutoff).values
            if stim_trial_shift > 0:
                stim_trials = np.append(np.zeros(stim_trial_shift), stim_trials)[:-stim_trial_shift]
            if stimulated is not None:
                stimulated_arr.append(stim_trials)
            stimuli_arr.append(trials['signed_contrast'].values)
            actions_arr.append(trials['choice'].values)
            stim_sides_arr.append(trials['stim_side'].values)
            prob_left_arr.append(trials['probabilityLeft'].values)
            session_uuids.append(eid)
        except:
            print(f'Could not load trials for {eid}')

    if (len(session_uuids) == 0) and (stimulated is not None):
        return [], [], [], [], [], []
    elif len(session_uuids) == 0:
        return [], [], [], [], []

    # Get maximum number of trials across sessions
    max_len = np.array([len(stimuli_arr[k]) for k in range(len(stimuli_arr))]).max()

    # Pad with 0 such that we obtain nd arrays of size nb_sessions x nb_trials
    stimuli = np.array([np.concatenate((stimuli_arr[k], np.zeros(max_len-len(stimuli_arr[k]))))
                        for k in range(len(stimuli_arr))])
    actions = np.array([np.concatenate((actions_arr[k], np.zeros(max_len-len(actions_arr[k]))))
                        for k in range(len(actions_arr))])
    prob_left = np.array([np.concatenate((prob_left_arr[k], np.zeros(max_len-len(prob_left_arr[k]))))
                        for k in range(len(prob_left_arr))])
    stim_side = np.array([np.concatenate((stim_sides_arr[k],
                                          np.zeros(max_len-len(stim_sides_arr[k]))))
                          for k in range(len(stim_sides_arr))])
    if stimulated is not None:
        opto_stim = np.array([np.concatenate((stimulated_arr[k], np.zeros(max_len-len(stimulated_arr[k]))))
                              for k in range(len(stimulated_arr))])
    session_uuids = np.array(session_uuids)

    if session_uuids.shape[0] == 1:
        stimuli = np.array([np.squeeze(stimuli)])
        actions = np.array([np.squeeze(actions)])
        prob_left = np.array([np.squeeze(prob_left)])
        stim_side = np.array([np.squeeze(stim_side)])
        if stimulated is not None:
            opto_stim = np.array([np.squeeze(opto_stim)])

    if stimulated is not None:
        return actions, stimuli, stim_side, prob_left, opto_stim, session_uuids
    else:
        return actions, stimuli, stim_side, prob_left, session_uuids


def load_passive_opto_times(eid, return_off_times=False, one=None):
    """
    Load in the time stamps of the optogenetic stimulation at the end of the recording, after the
    taks and the spontaneous activity.

    Returns
    opto_train_times : 1D array
        Timestamps of the start of each pulse train
    opto_pulse_times : 1D array
        Timestamps of all individual pulses
    opto_off_times : 1D array
        Timestamps of when the light pulse goes off
    """

    if one is None:
        one = ONE()

    # Load in laser pulses
    try:
        one.load_datasets(eid, datasets=[
            '_spikeglx_ephysData_g*_t0.nidq.cbin', '_spikeglx_ephysData_g*_t0.nidq.meta',
            '_spikeglx_ephysData_g*_t0.nidq.ch'], download_only=True)
    except:
        print('Error loading opto trace')
        return [], []
    session_path = one.eid2path(eid)
    nidq_file = glob(str(session_path.joinpath('raw_ephys_data/_spikeglx_ephysData_g*_t0.nidq.cbin')))[0]
    sr = spikeglx.Reader(nidq_file)
    offset = int((sr.shape[0] / sr.fs - 720) * sr.fs)
    opto_trace = sr.read_sync_analog(slice(offset, sr.shape[0]))[:, 1]
    opto_times = np.arange(offset, sr.shape[0]) / sr.fs

    # Get start times of pulse trains
    opto_on_times = opto_times[np.concatenate((np.diff(opto_trace), [0])) > 1]
    opto_off_times = opto_times[np.concatenate((np.diff(opto_trace), [0])) < -1]

    if len(opto_on_times) == 0:
        print(f'No pulses found for {eid}')
        return [], []
    else:
        # Find the opto pulses after the spontaneous activity (after a long break, here 100s)
        opto_train_times = opto_on_times[np.concatenate(([True], np.diff(opto_on_times) > 1))]

        if np.sum(np.diff(opto_train_times) > 100) > 0:
            first_pulse = np.where(np.diff(opto_train_times) > 100)[0][0]+1
        elif opto_train_times[0] - opto_times[0] > 100:
            first_pulse = 0
        else:
            print('Could not find passive laser pulses')
            return [], []
        opto_train_times = opto_train_times[first_pulse:]
        opto_on_times = opto_on_times[first_pulse:]
        opto_off_times = opto_off_times[first_pulse:]
        if return_off_times:
            return opto_train_times, opto_on_times, opto_off_times
        else:
            return opto_train_times, opto_on_times


def load_opto_pulse_times(eid, part='begin', time_slice=400, one=None):
        if one is None:
            one = ONE()

        # Load in laser pulses
        one.load_datasets(eid, datasets=[
            '_spikeglx_ephysData_g*_t0.nidq.cbin', '_spikeglx_ephysData_g*_t0.nidq.meta',
            '_spikeglx_ephysData_g*_t0.nidq.ch'], download_only=True)
        session_path = one.eid2path(eid)
        nidq_file = glob(str(session_path.joinpath('raw_ephys_data/_spikeglx_ephysData_g*_t0.nidq.cbin')))[0]
        sr = spikeglx.Reader(nidq_file)
        if part == 'begin':
            opto_trace = sr.read_sync_analog(slice(0, int(time_slice * sr.fs)))[:, 1]
            opto_times = np.arange(0, int(time_slice * sr.fs)) / sr.fs
        elif part == 'end':
            offset = int((sr.shape[0] / sr.fs - time_slice) * sr.fs)
            opto_trace = sr.read_sync_analog(slice(offset, offset + int(time_slice * sr.fs)))[:, 1]
            opto_times = np.arange(offset, offset + len(opto_trace)) / sr.fs

        # Get all pulse times
        opto_high_times = opto_times[opto_trace > 1]
        if len(opto_high_times) == 0:
            print(f'No pulses found for {eid}')
            return []
        else:
            opto_pulses = opto_high_times[np.concatenate(([True], np.diff(opto_high_times) > 0.01))]
            return opto_pulses


def load_lfp(eid, probe, time_start, time_end, relative_to='begin', destriped=False, one=None):
    one = one or ONE()
    destriped_lfp_path = join(paths()[1], 'LFP')

    # Download LFP data
    if destriped:
        ses_details = one.get_details(eid)
        subject = ses_details['subject']
        date = ses_details['start_time'][:10]
        lfp_path = join(destriped_lfp_path, f'{subject}_{date}_{probe}_destriped_lfp.cbin')
    else:
        lfp_paths, _ = one.load_datasets(eid, download_only=True, datasets=[
            '_spikeglx_ephysData_g*_t0.imec*.lf.cbin', '_spikeglx_ephysData_g*_t0.imec*.lf.meta',
            '_spikeglx_ephysData_g*_t0.imec*.lf.ch'], collections=[f'raw_ephys_data/{probe}'] * 3)
        lfp_path = lfp_paths[0]
    sr = spikeglx.Reader(lfp_path)

    # Convert time to samples
    if relative_to == 'begin':
        samples_start = int(time_start * sr.fs)
        samples_end = int(time_end * sr.fs)
    elif relative_to == 'end':
        samples_start = sr.shape[0] - int(time_start * sr.fs)
        samples_end = sr.shape[0] - int(time_end * sr.fs)

    # Load in lfp slice
    signal = sr.read(nsel=slice(samples_start, samples_end, None), csel=slice(None, None, None))[0]
    signal = signal.T
    time = np.arange(samples_start, samples_end) / sr.fs

    return signal, time


def plot_scalar_on_slice(
        regions, values, coord=-1000, slice='coronal', mapping='Beryl', hemisphere='left',
        cmap='viridis', background='boundary', clevels=None, brain_atlas=None, colorbar=False, ax=None):
    """
    Function to plot scalar value per allen region on histology slice
    :param regions: array of acronyms of Allen regions
    :param values: array of scalar value per acronym. If hemisphere is 'both' and different values want to be shown on each
    hemispheres, values should contain 2 columns, 1st column for LH values, 2nd column for RH values
    :param coord: coordinate of slice in um (not needed when slice='top')
    :param slice: orientation of slice, options are 'coronal', 'sagittal', 'horizontal', 'top' (top view of brain)
    :param mapping: atlas mapping to use, options are 'Allen', 'Beryl' or 'Cosmos'
    :param hemisphere: hemisphere to display, options are 'left', 'right', 'both'
    :param background: background slice to overlay onto, options are 'image' or 'boundary'
    :param cmap: colormap to use
    :param clevels: min max color levels [cim, cmax]
    :param brain_atlas: AllenAtlas object
    :param colorbar: whether to plot a colorbar
    :param ax: optional axis object to plot on
    :return:
    """

    if clevels is None:
        clevels = (np.min(values), np.max(values))

    ba = brain_atlas or AllenAtlas()
    br = ba.regions

    # Find the mapping to use
    map_ext = '-lr'
    map = mapping + map_ext

    region_values = np.zeros_like(br.id) * np.nan

    if len(values.shape) == 2:
        for r, vL, vR in zip(regions, values[:, 0], values[:, 1]):
            region_values[np.where(br.acronym[br.mappings[map]] == r)[0][0]] = vR
            region_values[np.where(br.acronym[br.mappings[map]] == r)[0][1]] = vL
    else:
        for r, v in zip(regions, values):
            region_values[np.where(br.acronym[br.mappings[map]] == r)[0]] = v

        lr_divide = int((br.id.shape[0] - 1) / 2)
        if hemisphere == 'left':
            region_values[0:lr_divide] = np.nan
        elif hemisphere == 'right':
            region_values[lr_divide:] = np.nan
            region_values[0] = np.nan

    if ax:
        fig = ax.get_figure()
    else:
        fig, ax = plt.subplots()

    if background == 'boundary':
        cmap_bound = matplotlib.cm.get_cmap("bone_r").copy()
        cmap_bound.set_under([1, 1, 1], 0)

    if slice == 'coronal':

        if background == 'image':
            ba.plot_cslice(coord / 1e6, volume='image', mapping=map, ax=ax)
            ba.plot_cslice(
                coord / 1e6, volume='value', region_values=region_values, mapping=map, cmap=cmap,
                vmin=clevels[0], vmax=clevels[1], ax=ax)
        else:
            ba.plot_cslice(
                coord / 1e6, volume='value', region_values=region_values, mapping=map, cmap=cmap,
                vmin=clevels[0], vmax=clevels[1], ax=ax)
            ba.plot_cslice(
                coord / 1e6, volume='boundary', mapping=map, ax=ax, cmap=cmap_bound, vmin=0.01,
                vmax=0.8)

    elif slice == 'sagittal':
        if background == 'image':
            ba.plot_sslice(coord / 1e6, volume='image', mapping=map, ax=ax)
            ba.plot_sslice(
                coord / 1e6, volume='value', region_values=region_values, mapping=map, cmap=cmap,
                vmin=clevels[0], vmax=clevels[1], ax=ax)
        else:
            ba.plot_sslice(
                coord / 1e6, volume='value', region_values=region_values, mapping=map, cmap=cmap,
                vmin=clevels[0], vmax=clevels[1], ax=ax)
            ba.plot_sslice(
                coord / 1e6, volume='boundary', mapping=map, ax=ax, cmap=cmap_bound, vmin=0.01,
                vmax=0.8)

    elif slice == 'horizontal':
        if background == 'image':
            ba.plot_hslice(coord / 1e6, volume='image', mapping=map, ax=ax)
            ba.plot_hslice(
                coord / 1e6, volume='value', region_values=region_values, mapping=map, cmap=cmap,
                vmin=clevels[0], vmax=clevels[1], ax=ax)
        else:
            ba.plot_hslice(
                coord / 1e6, volume='value', region_values=region_values, mapping=map, cmap=cmap,
                vmin=clevels[0], vmax=clevels[1], ax=ax)
            ba.plot_hslice(
                coord / 1e6, volume='boundary', mapping=map, ax=ax, cmap=cmap_bound, vmin=0.01,
                vmax=0.8)

    elif slice == 'top':
        if background == 'image':
            ba.plot_top(volume='image', mapping=map, ax=ax)
            ba.plot_top(
                volume='value', region_values=region_values, mapping=map, cmap=cmap,
                vmin=clevels[0], vmax=clevels[1], ax=ax)
        else:
            ba.plot_top(
                volume='value', region_values=region_values, mapping=map, cmap=cmap,
                vmin=clevels[0], vmax=clevels[1], ax=ax)
            ba.plot_top(
                volume='boundary', mapping=map, ax=ax, cmap=cmap_bound, vmin=0.01, vmax=0.8)

    return fig, ax


def fit_psychfunc(stim_levels, n_trials, proportion):
    # Fit a psychometric function with two lapse rates
    #
    # Returns vector pars with [bias, threshold, lapselow, lapsehigh]
    import psychofit as psy
    assert(stim_levels.shape == n_trials.shape == proportion.shape)
    if stim_levels.max() <= 1:
        stim_levels = stim_levels * 100

    pars, _ = psy.mle_fit_psycho(np.vstack((stim_levels, n_trials, proportion)),
                                 P_model='erf_psycho_2gammas',
                                 parstart=np.array([0, 20, 0.05, 0.05]),
                                 parmin=np.array([-100, 5, 0, 0]),
                                 parmax=np.array([100, 100, 1, 1]))
    return pars


def plot_psychometric(trials, ax, **kwargs):
    import psychofit as psy
    if trials['signed_contrast'].max() <= 1:
        trials['signed_contrast'] = trials['signed_contrast'] * 100

    stim_levels = np.sort(trials['signed_contrast'].unique())
    pars = fit_psychfunc(stim_levels, trials.groupby('signed_contrast').size(),
                         trials.groupby('signed_contrast').mean()['right_choice'])

    # plot psychfunc
    sns.lineplot(x=np.arange(-27, 27), y=psy.erf_psycho_2gammas(pars, np.arange(-27, 27)),
                 ax=ax, **kwargs)

    # plot psychfunc: -100, +100
    sns.lineplot(x=np.arange(-36, -31), y=psy.erf_psycho_2gammas(pars, np.arange(-103, -98)),
                 ax=ax, **kwargs)
    sns.lineplot(x=np.arange(31, 36), y=psy.erf_psycho_2gammas(pars, np.arange(98, 103)),
                 ax=ax, **kwargs)

    # now break the x-axis
    trials['signed_contrast'].replace(-100, -35)
    trials['signed_contrast'].replace(100, 35)

    # plot datapoints with errorbars on top
    sns.lineplot(x=trials['signed_contrast'], y=trials['right_choice'], ax=ax,
                     **{**{'err_style':"bars",
                     'linewidth':0, 'linestyle':'None', 'mew':0.5,
                     'marker':'o', 'ci':68}, **kwargs})

    ax.set(xticks=[-35, -25, -12.5, 0, 12.5, 25, 35], xlim=[-40, 40], ylim=[0, 1.02],
           yticks=[0, 0.25, 0.5, 0.75, 1], yticklabels=['0', '25', '50', '75', '100'],
           ylabel='Right choices', xlabel='Contrast (%)')
    ax.set_xticklabels(['-100', '-25', '-12.5', '0', '12.5', '25', '100'])
    #break_xaxis()


def break_xaxis(y=-0.004, **kwargs):

    # axisgate: show axis discontinuities with a quick hack
    # https://twitter.com/StevenDakin/status/1313744930246811653?s=19
    # first, white square for discontinuous axis
    plt.text(-30, y, '-', fontsize=14, fontweight='bold',
             horizontalalignment='center', verticalalignment='center',
             color='w')
    plt.text(30, y, '-', fontsize=14, fontweight='bold',
             horizontalalignment='center', verticalalignment='center',
             color='w')

    # put little dashes to cut axes
    plt.text(-30, y, '/ /', horizontalalignment='center',
             verticalalignment='center', fontsize=12, fontweight='bold')
    plt.text(30, y, '/ /', horizontalalignment='center',
             verticalalignment='center', fontsize=12, fontweight='bold')


def get_bias(trials):
    import psychofit as psy
    """
    Calculate bias by fitting psychometric curves to the 80/20 and 20/80 blocks, finding the
    point on the y-axis when contrast = 0% and getting the difference.
    """
    if len(trials) == 0:
        return np.nan

    # 20/80 blocks
    these_trials = trials[trials['probabilityLeft'] == 0.2]
    stim_levels = np.sort(these_trials['signed_contrast'].unique())
    pars_right = fit_psychfunc(stim_levels, these_trials.groupby('signed_contrast').size(),
                               these_trials.groupby('signed_contrast').mean()['right_choice'])
    bias_right = psy.erf_psycho_2gammas(pars_right, 0)

    # 80/20 blocks
    these_trials = trials[trials['probabilityLeft'] == 0.8]
    stim_levels = np.sort(these_trials['signed_contrast'].unique())
    pars_left = fit_psychfunc(stim_levels, these_trials.groupby('signed_contrast').size(),
                              these_trials.groupby('signed_contrast').mean()['right_choice'])
    bias_left = psy.erf_psycho_2gammas(pars_left, 0)

    return bias_right - bias_left


def fit_glm(behav, prior_blocks=True, opto_stim=False, rt_cutoff=None, folds=3):

    # drop trials with contrast-level 50, only rarely present (should not be its own regressor)
    behav = behav[np.abs(behav.signed_contrast) != 50]

    # add extra parameters to GLM
    model_str = 'choice ~ 1 + stimulus_side:C(contrast, Treatment) + previous_choice:C(previous_outcome)'
    if opto_stim:
        model_str = model_str + ' + laser_stimulation'
    if prior_blocks:
        model_str = model_str + ' + block_id'

    # drop NaNs
    behav = behav.dropna(subset=['trial_feedback_type', 'choice', 'previous_choice',
                                 'previous_outcome', 'reaction_times']).reset_index(drop=True)

    # use patsy to easily build design matrix
    endog, exog = patsy.dmatrices(model_str, data=behav, return_type='dataframe')

    # remove the one column (with 0 contrast) that has no variance
    if 'stimulus_side:C(contrast, Treatment)[0.0]' in exog.columns:
        exog.drop(columns=['stimulus_side:C(contrast, Treatment)[0.0]'], inplace=True)

    # recode choices for logistic regression
    endog['choice'] = endog['choice'].map({-1:0, 1:1})

    # rename columns
    exog.rename(columns={'Intercept': 'bias',
             'stimulus_side:C(contrast, Treatment)[6.25]': '6.25',
             'stimulus_side:C(contrast, Treatment)[12.5]': '12.5',
             'stimulus_side:C(contrast, Treatment)[25.0]': '25',
             'stimulus_side:C(contrast, Treatment)[50.0]': '50',
             'stimulus_side:C(contrast, Treatment)[100.0]': '100',
             'previous_choice:C(previous_outcome)[-1.0]': 'unrewarded',
             'previous_choice:C(previous_outcome)[1.0]': 'rewarded'},
             inplace=True)

    # NOW FIT THIS WITH STATSMODELS - ignore NaN choices
    logit_model = sm.Logit(endog, exog)
    res = logit_model.fit_regularized(disp=False) # run silently

    # what do we want to keep?
    params = pd.DataFrame(res.params).T
    params['pseudo_rsq'] = res.prsquared # https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.LogitResults.prsquared.html?highlight=pseudo
    params['condition_number'] = np.linalg.cond(exog)

    # ===================================== #
    # ADD MODEL ACCURACY - cross-validate

    kf = KFold(n_splits=folds, shuffle=True)

    if rt_cutoff is None:
        acc = np.array([])
        for train, test in kf.split(endog):
            X_train, X_test, y_train, y_test = exog.loc[train], exog.loc[test], \
                                               endog.loc[train], endog.loc[test]
            # fit again
            logit_model = sm.Logit(y_train, X_train)
            res = logit_model.fit_regularized(disp=False)  # run silently

            # compute the accuracy on held-out data [from Luigi]:
            # suppose you are predicting Pr(Left), let's call it p,
            # the % match is p if the actual choice is left, or 1-p if the actual choice is right
            # if you were to simulate it, in the end you would get these numbers
            y_test['pred'] = res.predict(X_test)
            y_test.loc[y_test['choice'] == 0, 'pred'] = 1 - y_test.loc[y_test['choice'] == 0, 'pred']
            acc = np.append(acc, y_test['pred'].mean())

        # average prediction accuracy over the K folds
        params['accuracy'] = np.mean(acc)
    else:
        acc_rt_short , acc_rt_long = np.array([]), np.array([])
        exog_short = exog[behav['reaction_times'] < rt_cutoff].copy().reset_index(drop=True)
        exog_long = exog[behav['reaction_times'] > rt_cutoff].copy().reset_index(drop=True)
        endog_short = endog[behav['reaction_times'] < rt_cutoff].copy().reset_index(drop=True)
        endog_long = endog[behav['reaction_times'] > rt_cutoff].copy().reset_index(drop=True)
        for train, test in kf.split(endog_short):
            X_train, X_test, y_train, y_test = exog_short.loc[train], exog_short.loc[test], \
                                               endog_short.loc[train], endog_short.loc[test]
            # fit again
            logit_model = sm.Logit(y_train, X_train)
            res = logit_model.fit_regularized(disp=False)  # run silently

            # compute the accuracy on held-out data [from Luigi]:
            # suppose you are predicting Pr(Left), let's call it p,
            # the % match is p if the actual choice is left, or 1-p if the actual choice is right
            # if you were to simulate it, in the end you would get these numbers
            y_test['pred'] = res.predict(X_test)
            y_test.loc[y_test['choice'] == 0, 'pred'] = 1 - y_test.loc[y_test['choice'] == 0, 'pred']
            acc_rt_short = np.append(acc_rt_short, y_test['pred'].mean())
        for train, test in kf.split(endog_long):
            X_train, X_test, y_train, y_test = exog_long.loc[train], exog_long.loc[test], \
                                               endog_long.loc[train], endog_long.loc[test]
            # fit again
            logit_model = sm.Logit(y_train, X_train)
            res = logit_model.fit_regularized(disp=False)  # run silently

            # compute the accuracy on held-out data [from Luigi]:
            # suppose you are predicting Pr(Left), let's call it p,
            # the % match is p if the actual choice is left, or 1-p if the actual choice is right
            # if you were to simulate it, in the end you would get these numbers
            y_test['pred'] = res.predict(X_test)
            y_test.loc[y_test['choice'] == 0, 'pred'] = 1 - y_test.loc[y_test['choice'] == 0, 'pred']
            acc_rt_long = np.append(acc_rt_long, y_test['pred'].mean())

    # average prediction accuracy over the K folds
    params['accuracy_rt_short'] = np.mean(acc_rt_short)
    params['accuracy_rt_long'] = np.mean(acc_rt_long)

    return params  # wide df