import numpy as np
import random
import json
import math
import os
from semantic_distribution import cat_name_to_weights


DATA_PATH = '../data/coco'
f_train = open(DATA_PATH + '/annotations/instances_train2014.json')
f_val = open(DATA_PATH + '/annotations/instances_val2014.json')
files = {'train' : json.load(f_train), 'val' : json.load(f_val)}


def observe_bias(bias_type: str, phase: str, seed: int) -> np.ndarray:
    """
    Returns a single-positive label matrix sampled according to the
    probability distribution induced by the bias_type.

    Arguments:
        - bias_type: one of size, loc, or semantic
        - phase: train or val
        - seed: To fix randomness
    """
    imID_to_anns = get_anns_by_imID()

    if bias_type == 'size' or bias_type == 'loc':
        (category_list, cat_id_to_index, cat_name_to_cat_id) = parse_categories(files['train']['categories'])
        return make_biased_matrix(imID_to_anns, bias_type,
                                  len(category_list), cat_id_to_index,
                                  seed, phase)
    elif bias_type == 'semantic':
        (category_list, cat_id_to_index, cat_name_to_cat_id) = parse_categories(files['val']['categories'])
        count_zero_weights = 0
        ordered_weights = get_ordered_weights(cat_name_to_weights, cat_id_to_index, cat_name_to_cat_id)
        for weight in ordered_weights:
            if weight == 0:
                count_zero_weights += 1
        
        assert len(ordered_weights) == len(cat_name_to_cat_id)
        return make_semantic_bias_matrix(ordered_weights, seed, phase)
    else:
        raise NotImplementedError(f"Unrecognized bias type: '{bias_type}'. Use one of `uniform`, `size`, `loc`, `semantic`")

def parse_categories(categories):
    category_list = []
    id_to_index = {}
    cat_name_to_cat_id = {}
    for i in range(len(categories)):
        category_list.append(categories[i]['name'])
        id_to_index[categories[i]['id']] = i
        cat_name_to_cat_id[categories[i]['name']] = categories[i]['id']
    return (category_list, id_to_index, cat_name_to_cat_id)

def get_anns_by_imID():
    imID_to_anns = {}
    for phase in ['train', 'val']:
        D = files[phase]
        for i in range(len(D['annotations'])):
            if not D['annotations'][i]['image_id'] in imID_to_anns:
                imID_to_anns[D['annotations'][i]['image_id']] = []
            imID_to_anns[D['annotations'][i]['image_id']].append(D['annotations'][i])
    return imID_to_anns


def get_sum_areas(annotations):
    sum_area = 0
    for ann in annotations:
        sum_area += ann['area']
    return sum_area


def get_imID_to_dims():
    imID_to_dims = {}
    for phase in ['train', 'val']:
        D = files[phase]
        for i in range(len(D['images'])):
            imID_to_dims[D['images'][i]['id']] = [D['images'][i]['width'], D['images'][i]['height']]
    return imID_to_dims


def get_loc_bias_norm_constant(annotations, imID_to_dims):
    sum_inv_dist = 0
    for ann in annotations:
        center = imID_to_dims[ann['image_id']]
        x_i = ann['bbox'][0] + ann['bbox'][2]/2  # x-coord. of center of annotation
        y_i = ann['bbox'][1] + ann['bbox'][3]/2  # y-coord. of center of annotation
        d_i = math.sqrt((x_i - center[0])**2 + (y_i - center[1])**2)
        if d_i == 0:
            d_i = 1
        sum_inv_dist += 1/d_i
    return 1/sum_inv_dist


def make_biased_matrix(imID_to_anns, bias_type, num_categories,
                       cat_id_to_index, rand_seed, phase):    
    random.seed(rand_seed)
    split = phase
    D = files[split]
    image_id_list = sorted(np.unique([str(D['annotations'][i]['image_id']) for i in range(len(D['annotations']))]))
    image_id_list = np.array(image_id_list, dtype=int)
    image_id_to_index = {image_id_list[i]: i for i in range(len(image_id_list))}

    image_id_and_weights = []
    count = 0
    if bias_type == 'loc':
        imID_to_dims = get_imID_to_dims()
    for im_id in image_id_list:
        weights_i = [0 for _ in range(num_categories)]
        if bias_type == 'size':
            sum_areas = get_sum_areas(imID_to_anns[im_id])
            assert sum_areas > 0
            annotations = imID_to_anns[im_id]
            for j in range(len(annotations)):
                weight = annotations[j]['area'] / sum_areas
                cat_id = annotations[j]['category_id']
                cat_idx = cat_id_to_index[cat_id]
                weights_i[cat_idx] += weight
        elif bias_type == 'loc':
            norm_const = get_loc_bias_norm_constant(imID_to_anns[im_id], imID_to_dims)
            assert norm_const > 0
            for j in range(len(imID_to_anns[im_id])):
                img_ctr = (0.5*imID_to_dims[im_id][0], 0.5*imID_to_dims[im_id][1])
                x_i = imID_to_anns[im_id][j]['bbox'][0] + imID_to_anns[im_id][j]['bbox'][2]/2
                y_i = imID_to_anns[im_id][j]['bbox'][1] + imID_to_anns[im_id][j]['bbox'][3]/2
                d_i = math.sqrt((x_i - img_ctr[0])**2 + (y_i - img_ctr[1])**2)
                if d_i == 0:
                    d_i = 0.0005
                cat_idx = cat_id_to_index[imID_to_anns[im_id][j]['category_id']]
                weights_i[cat_idx] += norm_const/d_i
        image_id_and_weights.append((im_id, weights_i))
        count += 1
    
    label_matrix = np.load(os.path.join(DATA_PATH, 'formatted_{}_labels.npy'.format(split)))
    biased_matrix = np.zeros_like(label_matrix)

    for k in range(len(image_id_and_weights)):
        im_id = int(image_id_and_weights[k][0])
        weights = image_id_and_weights[k][1]
        row_idx = image_id_to_index[im_id]
        labels = label_matrix[row_idx][:]
        assert len(labels) == len(weights)
        col_idx = random.choices(range(len(labels)), weights=weights)[0]
        assert label_matrix[row_idx][col_idx] == 1
        biased_matrix[row_idx][col_idx] = 1
    
    return biased_matrix


def get_ordered_weights(cat_name_to_weights, cat_id_to_index, cat_name_to_cat_id):
    ordered_weights = [0 for _ in range(len(cat_name_to_cat_id))]
    for cat_name in cat_name_to_cat_id:
        cat_id = cat_name_to_cat_id[cat_name]
        col_idx = cat_id_to_index[cat_id]
        ordered_weights[col_idx] = cat_name_to_weights[cat_name]
    return ordered_weights


def make_semantic_bias_matrix(weights, rand_seed: int, split: str):
    random.seed(rand_seed)
    D = files[split]
    image_id_list = sorted(np.unique([str(D['annotations'][i]['image_id']) for i in range(len(D['annotations']))]))
    image_id_list = np.array(image_id_list, dtype=int)
    image_id_to_index = {image_id_list[i]: i for i in range(len(image_id_list))}

    label_matrix = np.load(os.path.join(DATA_PATH, 'formatted_{}_labels.npy'.format(split)))
    biased_matrix = np.zeros_like(label_matrix)
    totalx = 0
    counter = 0
    for k in range(len(image_id_list)):
        totalx += 1
        im_id = image_id_list[k]
        row_idx = image_id_to_index[im_id]
        labels = label_matrix[row_idx][:]
        assert len(labels) == len(weights), f'len(labels) = {len(labels)}, len(weights) = {len(weights)}'
        curr_weights = [0 for _ in range(len(weights))]
        valid_indices = []
        for i in range(len(labels)):
            if not labels[i] == 0:
                curr_weights[i] = weights[i]
                valid_indices.append(i)
        test_w = np.unique(curr_weights)
        if len(test_w) == 1:  # if all weights are zero, choose any
            counter += 1
            assert test_w[0] == 0
            col_idx = random.choice(valid_indices)
        else:
            col_idx = random.choices(range(len(labels)), curr_weights)[0]
        assert label_matrix[row_idx][col_idx] == 1
        biased_matrix[row_idx][col_idx] = 1

    return biased_matrix
