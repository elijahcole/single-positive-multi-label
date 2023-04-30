from format_coco import parse_categories
import numpy as np
import random
import json
import math
import os
#  from distributions import cat_name_to_weights, cat_id_to_cat_name


DATA_PATH = '../data/coco'
f_train = open(DATA_PATH + '/annotations/instances_train2014.json')
f_val = open(DATA_PATH + 'instances_val2014.json')
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

    if bias_type == 'size' or bias_type == 'location':
        # TODO: parse_categories returns only two
        (category_list, cat_id_to_index, cat_name_to_cat_id) = parse_categories(files['train']['categories'])
        return make_biased_matrix(imID_to_anns, bias_type,
                                  len(category_list), cat_id_to_index,
                                  seed, files, phase)
    elif bias_type == 'empirical':
        # print(f'THIS SHOULD NOT BE NONE: {cat_name_to_weights}')  # TODO: MAKE SURE IT IS IMPORTED CORRECTLY
        (category_list, cat_id_to_index, cat_name_to_cat_id) = parse_categories(files['val']['categories'])
        # cat_name_to_cat_id = {cat_id_to_cat_name[cat_id]: cat_id for cat_id in cat_id_to_cat_name}
        set1 = set(cat_name_to_weights.keys())
        set2 = set(cat_name_to_cat_id.keys())
        print(f'CAT NAMES IN WEIGHTS BUT NOT IN THE ID MAP{set1-set2}')
        print(f'CAT NAMES IN ID MAP BUT NOT IN THE WEIGHTS: {set2 - set1}')
        print(len(cat_name_to_weights))
        print(len(cat_name_to_cat_id))
        count_zero_weights = 0
        ordered_weights = get_ordered_weights(cat_name_to_weights, cat_id_to_index, cat_name_to_cat_id)
        for weight in ordered_weights:
            if weight == 0:
                count_zero_weights += 1
        
        assert len(ordered_weights) == len(cat_name_to_cat_id)
        make_biased_matrix_emp(ordered_weights, bias_type)


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
        x_i = ann['bbox'][0] + ann['bbox'][2]/2 # x-coord. of center of annotation
        y_i = ann['bbox'][1] + ann['bbox'][3]/2 # y-coord. of center of annotation
        d_i = math.sqrt((x_i - center[0])**2 + (y_i - center[1])**2)
        if d_i == 0:
            d_i = 1
        sum_inv_dist += 1/d_i
    return 1/sum_inv_dist


def make_biased_matrix(imID_to_anns, bias_type, num_categories,
                       cat_id_to_index, rand_seed, files, phase):    
    random.seed(rand_seed)
    split = phase
    D = files[split]
    image_id_list = sorted(np.unique([str(D['annotations'][i]['image_id']) for i in range(len(D['annotations']))]))
    image_id_list = np.array(image_id_list, dtype=int)
    image_id_to_index = {image_id_list[i]: i for i in range(len(image_id_list))}

    image_id_and_weights = []
    total_im_ids = len(image_id_list)
    count = 0
    if bias_type == 'loc':
        imID_to_dims = get_imID_to_dims()
    for im_id in image_id_list:
        if count % int(total_im_ids*0.05) == 0:
            print(count/total_im_ids)
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


def make_biased_matrix_emp(weights, bias_type):
    N = 5
    SEED = 10
    for n in range(1, N+1):
        random.seed(SEED)
        SEED += 1234
        for split in ['train', 'val']:
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
                        # if not weights[i] == 0:
                        #     flag = True
                        # curr_weights[i] = 0
                        # if flag:
                        #     assert not weights[i] == 0
                test_w = np.unique(curr_weights)
                if len(test_w) == 1:  # if all weights are zero, choose any
                    counter += 1
                    assert test_w[0] == 0
                    col_idx = random.choice(valid_indices)
                else:
                    col_idx = random.choices(range(len(labels)), curr_weights)[0]
                assert label_matrix[row_idx][col_idx] == 1
                biased_matrix[row_idx][col_idx] = 1
            np.save(os.path.join(labels_path,
                                'coco_formatted_{}_{}_{}_labels_obs.npy'.format(split, bias_type, n)),
                                biased_matrix)
            print(counter/totalx)


# def get_imID_to_dims():
#     imID_to_dims = {}
#     for phase in ['train', 'phase']:
#         f = open(data_path + coco_path + 'instances_{}2014.json'.format(phase))
#         D = json.load(f)
#         for i in range(len(D['images'])):
#             imID_to_dims[D['images'][i]['id']] = [D['images'][i]['width'], D['images'][i]['height']]
#         f.close()
#     return imID_to_dims


# def get_center_bias_norm_const(annotations, imID_to_dims):
#     sum_inv_dist = 0
#     for ann in annotations:
#         center = imID_to_dims[ann['image_id']]
#         x_i = ann['bbox'][0] + ann['bbox'][2]/2 # x-coord. of center of annotation
#         y_i = ann['bbox'][1] + ann['bbox'][3]/2 # y-coord. of center of annotation
#         d_i = math.sqrt((x_i - center[0])**2 + (y_i - center[1])**2)
#         if d_i == 0:
#             d_i = 1
#         sum_inv_dist += 1/d_i
#     return 1/sum_inv_dist


# def get_weights_per_im(imID_to_anns, bias_type):
#     imID_to_catWeights = {}
#     imID_to_annWeights = {}
#     for imID in imID_to_anns:
#         # imID_to_catWeights[imID][i] will be the weight of the (i + 1)-th category as given by D['categories']
#         # reason i + 1 is that pascal_ann.json 1-indexed the categories
#         imID_to_catWeights[imID] = {meta['category_id_to_index'][cat_id] : 0 for cat_id in meta['category_id_to_index']}
#         if bias_type == 'size':
#             sum_areas = get_sum_areas(imID_to_anns[imID])
#             assert sum_areas > 0
#             for i in range(len(imID_to_anns[imID])):
#                 weight = imID_to_anns[imID][i]['area'] / sum_areas
#                 cat_col_idx = meta['category_id_to_index'][imID_to_anns[imID][i]['category_id']]
#                 imID_to_catWeights[imID][cat_col_idx] += weight
#         elif bias_type == 'location':
#             assert False  # TODO: Make location bias work too
#             imID_to_dims = get_imID_to_dims()
#             norm_const = get_center_bias_norm_const(imID_to_anns[imID], imID_to_dims)
#             assert norm_const > 0
#             for j in range(len(imID_to_anns[imID])):
#                 img_ctr = imID_to_dims[imID]
#                 x_i = imID_to_anns[imID][j]['bbox'][0] + imID_to_anns[imID][j]['bbox'][2]/2
#                 y_i = imID_to_anns[imID][j]['bbox'][1] + imID_to_anns[imID][j]['bbox'][3]/2
#                 d_i = math.sqrt((x_i - img_ctr[0])**2 + (y_i - img_ctr[1])**2)
#                 imID_to_annWeights[imID][j] = norm_const/d_i
#                 imID_to_catWeights[imID][imID_to_anns[imID][j]['category_id'] - 1] += norm_const/d_i
#     return (imID_to_catWeights, imID_to_annWeights)


# # def get_weights_per_im(imID_to_anns):

# #     imID_to_catWeights = {}
# #     imID_to_annWeights = {}
# #     for imID in imID_to_anns:
# #         # imID_to_catWeights[imID][i] will be the weight of the (i + 1)-th category as given by D['categories']
# #         # reason i + 1 is that pascal_ann.json 1-indexed the categories
# #         imID_to_catWeights[imID] = [0 for _ in range(20)]
# #         imID_to_annWeights[imID] = [0 for _ in range(len(imID_to_anns[imID]))]
# #         sum_areas = get_sum_areas(imID_to_anns[imID])
# #         assert sum_areas > 0

# #         # START new code
# #         # if bias == 'size':
# #         #     sum_areas = get_sum_areas(imID_to_anns[imID])
# #         #     assert sum_areas > 0
# #         # elif bias == 'location':
# #         #     continue
# #         # END NEW CODE
# #         for i in range(len(imID_to_anns[imID])):
# #             imID_to_catWeights[imID][imID_to_anns[imID][i]['category_id'] - 1] += imID_to_anns[imID][i]['area'] / sum_areas
# #             imID_to_annWeights[imID][i] = imID_to_anns[imID][i]['area'] / sum_areas
# #     return (imID_to_catWeights, imID_to_annWeights)


# def create_image_list(ann_dict, image_list, phase):
#     # TODO: no reason to repeat this code from format_pascal.py
#     # image_list = []
#     for cat_name in catName_to_catID:
#         with open(os.path.join(data_path + '/pascal', 'VOCdevkit', 'VOC2012', 'ImageSets', 'Main', cat_name + '_' + phase + '.txt'), 'r') as f:
#             for line in f:
#                 cur_line = line.rstrip().split(' ')
#                 image_id = cur_line[0]
#                 label = cur_line[-1]
#                 image_fname = image_id + '.jpg'
#                 # if cat occurs, save it in ann_dict, which maps from image to catID that occur in the image
#                 # image_list holds all images with annotated objects in them, classified by train or val
#                 if int(label) == 1:
#                     if image_fname not in ann_dict:
#                         ann_dict[image_fname] = []
#                         image_list[phase].append(image_id)
#                     ann_dict[image_fname].append(catName_to_catID[cat_name])
#     # return image_list


# def observe_bias(label_matrix, imID_to_catWeights, phase, num_observations=1):
#     '''
#     Issue: I want to visualize biased labels. Eli's approach is to have a matrix of ocurrences.
#     So I can get the most likely observation and make a label_matrix. But I won't be able to visualize
#     from that, I need to know which specific annotation is most likely.
#     '''
#     label_matrix_biased = np.zeros_like(label_matrix)
#     (num_images, _) = np.shape(label_matrix_biased)
#     for im_id in imID_to_catWeights:
#         meta[]
#     # for row_idx in range(num_images):
#     #     im_id = meta["row_idx_to_im_id"][phase][row_idx]
#     #     cat_idx_to_weight = imID_to_catWeights[im_id]
#     #     cat_weights = np.zeros_like(label_matrix[row_idx])
#     #     for cat_idx in range(len(cat_weights)):
#     #         cat_weights[cat_idx] = cat_idx_to_weight[cat_idx]
#     #     idx_pos = int(random.choices(label_matrix[row_idx], cat_weights, k=num_observations)[0])
#     #     label_matrix_biased[row_idx][idx_pos] = 1.0
#     return label_matrix_biased


# def get_matrixIdx_to_imID(image_list_ph):
#     matrix_idx_2_im_id = {}
#     for i in range(len(image_list_ph)):
#         matrix_idx_2_im_id[i] = image_list_ph[i]
#     return matrix_idx_2_im_id


# def get_biased_annotations(imID_to_anns, imID_to_annWeights):
#     biased_annotations = {}
#     for curr_image_id in imID_to_anns:
#         curr_weights = imID_to_annWeights[curr_image_id]
#         chosen_annotation = random.choices(imID_to_anns[curr_image_id], curr_weights)
#         biased_annotations[curr_image_id] = chosen_annotation[0]
#     return biased_annotations


# # def visualize_bias(imID_to_anns, biased_annotations):
# #     font = cv2.FONT_HERSHEY_SIMPLEX
# #     font_scale = 0.85
# #     color = (255, 0, 127)
# #     thickness = 2

# #     sample_size = 5
# #     indices = [random.randint(0, len(D['annotations']) - 1) for _ in range(sample_size)]
# #