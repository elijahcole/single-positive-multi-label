import numpy as np
import os
import argparse

pp = argparse.ArgumentParser(description='')
pp.add_argument('--dataset', type=str, choices=['pascal', 'coco', 'nuswide', 'cub'], required=True)
pp.add_argument('--num-pos', type=int, default=1, required=False, help='number of positive labels per image')
pp.add_argument('--num-neg', type=int, default=0, required=False, help='number of negative labels per image')
pp.add_argument('--seed', type=int, default=1200, required=False, help='random seed')
args = pp.parse_args()

def get_random_label_indices(label_vec, label_value, num_sel, rng):
    '''
    Given a 1D numpy array label_vec, return num_sel indices chosen at random
    from all indices where label_vec equals label_value.
    Note that all relevant indices are returned if fewer than num_sel are found.
    '''
    
    # find all indices where label_vec is equal to label_value:
    idx_all = np.nonzero(label_vec == label_value)[0]
    
    # shuffle those indices:
    rng.shuffle(idx_all)
    
    # return (up to) the first num_sel indices:
    if num_sel == -1:
        idx_sel = idx_all
    else:
        idx_sel = idx_all[:num_sel]
    
    return idx_sel

def observe_uniform(label_matrix, num_pos, num_neg, rng):
    '''
    label_matrix: binary (-1/+1) label matrix with shape num_items x num_classes
    num_pos: number of positive labels to observe for each item
    num_neg: number of negative labes to observe for each item
    rng: random number generator to use
    '''
    
    # check the observation parameters:
    assert (num_pos == -1) or (num_pos >= 0)
    assert (num_neg == -1) or (num_neg >= 0)
    
    # check that label_matrix is a binary numpy array:
    assert type(label_matrix) is np.ndarray
    label_values = np.unique(label_matrix)
    assert len(label_values) == 2
    assert -1 in label_values
    assert 1 in label_values
    assert len(np.unique(label_matrix)) == 2
    
    # apply uniform observation process:
    num_items, num_classes = np.shape(label_matrix)
    label_matrix_obs = np.zeros_like(label_matrix)
    for i in range(num_items):    
        idx_pos = get_random_label_indices(label_matrix[i,:], 1.0, num_pos, rng)
        label_matrix_obs[i, idx_pos] = 1.0
        idx_neg = get_random_label_indices(label_matrix[i,:], -1.0, num_neg, rng)
        label_matrix_obs[i, idx_neg] = -1.0
    
    return label_matrix_obs

base_path = os.path.join('../data/{}'.format(args.dataset))

for phase in ['train', 'val']:
    # load ground truth binary label matrix:
    label_matrix = np.load(os.path.join(base_path, 'formatted_{}_labels.npy'.format(phase)))
    assert np.max(label_matrix) == 1
    assert np.min(label_matrix) == 0

    # convert label matrix to -1 / +1 format:
    label_matrix[label_matrix == 0] = -1

    # choose observed labels, resulting in -1 / 0 / +1 format:
    rng = np.random.RandomState(args.seed)
    label_matrix_obs = observe_uniform(label_matrix, args.num_pos, args.num_neg, rng)

    # save observed labels:
    np.save(os.path.join(base_path, 'formatted_{}_labels_obs.npy'.format(phase)), label_matrix_obs)
