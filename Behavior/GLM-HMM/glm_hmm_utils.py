import numpy as np
from scipy.stats import bernoulli
from GLM import glm
from sklearn import preprocessing
from os.path import join
import sys
import ssm
import autograd.numpy as np
import autograd.numpy.random as npr

"""
Adapted from on Zoe Ashwood's code (https://github.com/zashwood/glm-hmm)
"""


def get_raw_data(eid, one):
    # Get animal:
    animal = one.get_details(eid)['subject']

    # Get choice data, stim data and rewarded/not rewarded:
    trials = one.load_object(eid, 'trials')

    return (animal, trials.contrastLeft, trials.contrastRight, trials.feedbackType, trials.choice,
            trials.probabilityLeft)


def create_stim_vector(stim_left, stim_right):
    # want stim_right - stim_left
    # Replace NaNs with 0:
    stim_left = np.nan_to_num(stim_left, nan=0)
    stim_right = np.nan_to_num(stim_right, nan=0)
    # now get 1D stim
    signed_contrast = stim_right - stim_left
    return signed_contrast


def create_previous_choice_vector(choice):
    ''' choice: choice vector of size T
        previous_choice : vector of size T with previous choice made by
        animal - output is in {0, 1}, where 0 corresponds to a previous left
        choice; 1 corresponds to right.
        If the previous choice was a violation, replace this with the choice
        on the previous trial that was not a violation.
        locs_mapping: array of size (~num_viols)x2, where the entry in
        column 1 is the location in the previous choice vector that was a
        remapping due to a violation and the
        entry in column 2 is the location in the previous choice vector that
        this location was remapped to
    '''
    previous_choice = np.hstack([np.array(choice[0]), choice])[:-1]
    locs_to_update = np.where(previous_choice == -1)[0]
    locs_with_choice = np.where(previous_choice != -1)[0]
    loc_first_choice = locs_with_choice[0]
    locs_mapping = np.zeros((len(locs_to_update) - loc_first_choice, 2),
                            dtype='int')

    for i, loc in enumerate(locs_to_update):
        if loc < loc_first_choice:
            # since no previous choice, bernoulli sample: (not output of
            # bernoulli rvs is in {1, 2})
            previous_choice[loc] = bernoulli.rvs(0.5, 1) - 1
        else:
            # find nearest loc that has a previous choice value that is not
            # -1, and that is earlier than current trial
            potential_matches = locs_with_choice[
                np.where(locs_with_choice < loc)]
            absolute_val_diffs = np.abs(loc - potential_matches)
            absolute_val_diffs_ind = absolute_val_diffs.argmin()
            nearest_loc = potential_matches[absolute_val_diffs_ind]
            locs_mapping[i - loc_first_choice, 0] = int(loc)
            locs_mapping[i - loc_first_choice, 1] = int(nearest_loc)
            previous_choice[loc] = previous_choice[nearest_loc]
    assert len(np.unique(
        previous_choice)) <= 2, "previous choice should be in {0, 1}; " + str(
        np.unique(previous_choice))
    return previous_choice, locs_mapping


def create_wsls_covariate(previous_choice, success, locs_mapping):
    '''
    inputs:
    success: vector of size T, entries are in {-1, 1} and 0 corresponds to
    failure, 1 corresponds to success
    previous_choice: vector of size T, entries are in {0, 1} and 0
    corresponds to left choice, 1 corresponds to right choice
    locs_mapping: location remapping dictionary due to violations
    output:
    wsls: vector of size T, entries are in {-1, 1}.  1 corresponds to
    previous choice = right and success OR previous choice = left and
    failure; -1 corresponds to
    previous choice = left and success OR previous choice = right and failure
    '''
    # remap previous choice vals to {-1, 1}
    remapped_previous_choice = 2 * previous_choice - 1
    previous_reward = np.hstack([np.array(success[0]), success])[:-1]
    # Now need to go through and update previous reward to correspond to
    # same trial as previous choice:
    for i, loc in enumerate(locs_mapping[:, 0]):
        nearest_loc = locs_mapping[i, 1]
        previous_reward[loc] = previous_reward[nearest_loc]
    wsls = previous_reward * remapped_previous_choice
    assert len(np.unique(wsls)) == 2, "wsls should be in {-1, 1}"
    return wsls


def remap_choice_vals(choice):
    # raw choice vector has CW = 1 (correct response for stim on left),
    # CCW = -1 (correct response for stim on right) and viol = 0.  Let's
    # remap so that CW = 0, CCw = 1, and viol = -1
    choice_mapping = {1: 0, -1: 1, 0: -1}
    new_choice_vector = [choice_mapping[old_choice] for old_choice in choice]
    return new_choice_vector


def create_design_mat(choice, stim_left, stim_right, rewarded):
    # Create unnormalized_inpt: with first column = stim_right - stim_left,
    # second column as past choice, third column as WSLS
    stim = create_stim_vector(stim_left, stim_right)
    T = len(stim)
    design_mat = np.zeros((T, 3))
    design_mat[:, 0] = stim
    # make choice vector so that correct response for stim>0 is choice =1
    # and is 0 for stim <0 (viol is mapped to -1)
    choice = remap_choice_vals(choice)
    # create past choice vector:
    previous_choice, locs_mapping = create_previous_choice_vector(choice)
    # create wsls vector:
    wsls = create_wsls_covariate(previous_choice, rewarded, locs_mapping)
    # map previous choice to {-1,1}
    design_mat[:, 1] = 2 * previous_choice - 1
    design_mat[:, 2] = wsls
    return design_mat


def get_all_unnormalized_data_this_session(eid, one):
    # Load raw data
    animal, stim_left, stim_right, rewarded, choice, bias_probs \
        = get_raw_data(eid, one)
    # Exclude 50-50 trials
    trials_to_study = np.where(bias_probs != 0.5)[0]
    num_viols_50 = len(np.where(choice[trials_to_study] == 0)[0])
    if num_viols_50 < 10:
        # Create design mat = matrix of size T x 3, with entries for
        # stim/past choice/wsls
        unnormalized_inpt = create_design_mat(choice[trials_to_study],
                                              stim_left[trials_to_study],
                                              stim_right[trials_to_study],
                                              rewarded[trials_to_study])
        y = np.expand_dims(remap_choice_vals(choice[trials_to_study]), axis=1)
        session = [eid for i in range(y.shape[0])]
        rewarded = np.expand_dims(rewarded[trials_to_study], axis=1)
    else:
        unnormalized_inpt = np.zeros((90, 3))
        y = np.zeros((90, 1))
        session = []
        rewarded = np.zeros((90, 1))
    return animal, unnormalized_inpt, y, session, num_viols_50, rewarded


def create_train_test_sessions(session, num_folds=3):
    # create a session-fold lookup table
    num_sessions = len(np.unique(session))
    # Map sessions to folds:
    unshuffled_folds = np.repeat(np.arange(num_folds),
                                 np.ceil(num_sessions / num_folds))
    shuffled_folds = npr.permutation(unshuffled_folds)[:num_sessions]
    #assert len(np.unique(shuffled_folds)) == num_folds, "require at least one session per fold for " \
    #                                                                "each animal!"
    # Look up table of shuffle-folds:
    sess_id = np.array(np.unique(session), dtype='str')
    shuffled_folds = np.array(shuffled_folds, dtype='O')
    session_fold_lookup_table = np.transpose(
        np.vstack([sess_id, shuffled_folds]))
    return session_fold_lookup_table


def load_session_fold_lookup(file_path):
    container = np.load(file_path, allow_pickle=True)
    data = [container[key] for key in container]
    session_fold_lookup_table = data[0]
    return session_fold_lookup_table


def load_data(animal_file):
    container = np.load(animal_file, allow_pickle=True)
    data = [container[key] for key in container]
    inpt = data[0]
    y = data[1]
    session = data[2]
    return inpt, y, session


def load_global_params(global_params_file):
    container = np.load(global_params_file, allow_pickle=True)
    data = [container[key] for key in container]
    global_params = data[0]
    return global_params


def load_animal_list(list_file):
    container = np.load(list_file, allow_pickle=True)
    data = [container[key] for key in container]
    animal_list = data[0]
    return animal_list


def load_glm_vectors(glm_vectors_file):
    container = np.load(glm_vectors_file, allow_pickle=True)
    data = [container[key] for key in container][0]
    loglikelihood_train = data[0][0]
    recovered_weights = data[1][0]
    return loglikelihood_train, recovered_weights


# Append column of zeros to weights matrix in appropriate location
def append_zeros(weights):
    weights_tranpose = np.transpose(weights, (1, 0, 2))
    weights = np.transpose(
        np.vstack([
            weights_tranpose,
            np.zeros((1, weights_tranpose.shape[1], weights_tranpose.shape[2]))
        ]), (1, 0, 2))
    return weights


def load_cluster_arr(cluster_arr_file):
    container = np.load(cluster_arr_file, allow_pickle=True)
    data = [container[key] for key in container]
    cluster_arr = data[0]
    return cluster_arr


def partition_data_by_session(inpt, y, mask, session):
    '''
    Partition inpt, y, mask by session
    :param inpt: arr of size TxM
    :param y:  arr of size T x D
    :param mask: Boolean arr of size T indicating if element is violation or
    not
    :param session: list of size T containing session ids
    :return: list of inpt arrays, data arrays and mask arrays, where the
    number of elements in list = number of sessions and each array size is
    number of trials in session
    '''
    inputs = []
    datas = []
    indexes = np.unique(session, return_index=True)[1]
    unique_sessions = [session[index] for index in sorted(indexes)]
    counter = 0
    masks = []
    for sess in unique_sessions:
        idx = np.where(session == sess)[0]
        counter += len(idx)
        inputs.append(inpt[idx, :])
        datas.append(y[idx, :])
        masks.append(mask[idx, :])
    assert counter == inpt.shape[0], "not all trials assigned to session!"
    return inputs, datas, masks


def fit_glm(inputs, datas, M, C):
    new_glm = glm(M, C)
    new_glm.fit_glm(datas, inputs, masks=None, tags=None)
    # Get loglikelihood of training data:
    loglikelihood_train = new_glm.log_marginal(datas, inputs, None, None)
    recovered_weights = new_glm.Wk
    return loglikelihood_train, recovered_weights


def fit_glm_hmm(datas, inputs, masks, K, D, M, C, N_em_iters,
                transition_alpha, prior_sigma, global_fit,
                params_for_initialization, save_title):
    '''
    Instantiate and fit GLM-HMM model
    :param datas:
    :param inputs:
    :param masks:
    :param K:
    :param D:
    :param M:
    :param C:
    :param N_em_iters:
    :param global_fit:
    :param glm_vectors:
    :param save_title:
    :return:
    '''
    if global_fit == True:
        # Prior variables
        # Choice of prior
        this_hmm = ssm.HMM(K,
                           D,
                           M,
                           observations="input_driven_obs",
                           observation_kwargs=dict(C=C,
                                                   prior_sigma=prior_sigma),
                           transitions="sticky",
                           transition_kwargs=dict(alpha=transition_alpha,
                                                  kappa=0))
        # Initialize observation weights as GLM weights with some noise:
        glm_vectors_repeated = np.tile(params_for_initialization, (K, 1, 1))
        glm_vectors_with_noise = glm_vectors_repeated + np.random.normal(
            0, 0.2, glm_vectors_repeated.shape)
        this_hmm.observations.params = glm_vectors_with_noise
    else:
        # Choice of prior
        this_hmm = ssm.HMM(K,
                           D,
                           M,
                           observations="input_driven_obs",
                           observation_kwargs=dict(C=C,
                                                   prior_sigma=prior_sigma),
                           transitions="sticky",
                           transition_kwargs=dict(alpha=transition_alpha,
                                                  kappa=0))
        # Initialize HMM-GLM with global parameters:
        this_hmm.params = params_for_initialization
        # Get log_prior of transitions:
    print("=== fitting GLM-HMM ========")
    # Fit this HMM and calculate marginal likelihood
    lls = this_hmm.fit(datas,
                       inputs=inputs,
                       masks=masks,
                       method="em",
                       num_iters=N_em_iters,
                       initialize=False,
                       tolerance=10 ** -4)
    # Save raw parameters of HMM, as well as loglikelihood during training
    np.savez(save_title, this_hmm.params, lls)
    return None


def launch_glm_hmm_job(inpt, y, session, mask, session_fold_lookup_table, K, D,
                       C, N_em_iters, transition_alpha, prior_sigma, fold,
                       iter, global_fit, init_param_file, save_directory):
    print("Starting inference with K = " + str(K) + "; Fold = " + str(fold) +
          "; Iter = " + str(iter))
    sys.stdout.flush()
    sessions_to_keep = session_fold_lookup_table[np.where(
        session_fold_lookup_table[:, 1] != fold), 0]
    idx_this_fold = [str(sess) in sessions_to_keep for sess in session]
    this_inpt, this_y, this_session, this_mask = inpt[idx_this_fold, :], \
                                                 y[idx_this_fold, :], \
                                                 session[idx_this_fold], \
                                                 mask[idx_this_fold]
    # Only do this so that errors are avoided - these y values will not
    # actually be used for anything (due to violation mask)
    this_y[np.where(this_y == -1), :] = 1
    inputs, datas, masks = partition_data_by_session(
        this_inpt, this_y, this_mask, this_session)
    # Read in GLM fit if global_fit = True:
    if global_fit == True:
        _, params_for_initialization = load_glm_vectors(init_param_file)
    else:
        params_for_initialization = load_global_params(init_param_file)
    M = this_inpt.shape[1]
    npr.seed(iter)
    fit_glm_hmm(datas,
                inputs,
                masks,
                K,
                D,
                M,
                C,
                N_em_iters,
                transition_alpha,
                prior_sigma,
                global_fit,
                params_for_initialization,
                save_title=join(save_directory, 'glm_hmm_raw_parameters_itr_' + str(iter) + '.npz'))


def create_violation_mask(violation_idx, T):
    """
    Return indices of nonviolations and also a Boolean mask for inclusion (1
    = nonviolation; 0 = violation)
    :param test_idx:
    :param T:
    :return:
    """
    mask = np.array([i not in violation_idx for i in range(T)])
    nonviolation_idx = np.arange(T)[mask]
    mask = mask + 0
    assert len(nonviolation_idx) + len(
        violation_idx
    ) == T, "violation and non-violation idx do not include all dta!"
    return nonviolation_idx, np.expand_dims(mask, axis=1)

