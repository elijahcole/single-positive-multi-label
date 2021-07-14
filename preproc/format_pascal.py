import os
import json
import numpy as np
import argparse

pp = argparse.ArgumentParser(description='Format PASCAL 2012 metadata.')
pp.add_argument('--load-path', type=str, default='../data/pascal', help='Path to a directory containing a copy of the PASCAL dataset.')
pp.add_argument('--save-path', type=str, default='../data/pascal', help='Path to output directory.')
args = pp.parse_args()

catName_to_catID = {
    'aeroplane': 0,
    'bicycle': 1,
    'bird': 2,
    'boat': 3,
    'bottle': 4,
    'bus': 5,
    'car': 6,
    'cat': 7,
    'chair': 8,
    'cow': 9,
    'diningtable': 10,
    'dog': 11,
    'horse': 12,
    'motorbike': 13,
    'person': 14,
    'pottedplant': 15,
    'sheep': 16,
    'sofa': 17,
    'train': 18,
    'tvmonitor': 19
}

catID_to_catName = {catName_to_catID[k]: k for k in catName_to_catID}

ann_dict = {}
image_list = {'train': [], 'val': []}

for phase in ['train', 'val']:
    for cat in catName_to_catID:
        with open(os.path.join(args.load_path, 'VOCdevkit', 'VOC2012', 'ImageSets', 'Main', cat + '_' + phase + '.txt'), 'r') as f:
            for line in f:
                cur_line = line.rstrip().split(' ')
                image_id = cur_line[0]
                label = cur_line[-1]
                image_fname = image_id + '.jpg'
                if int(label) == 1:
                    if image_fname not in ann_dict:
                        ann_dict[image_fname] = []
                        image_list[phase].append(image_fname)
                    ann_dict[image_fname].append(catName_to_catID[cat])
    # create label matrix: 
    image_list[phase].sort()
    num_images = len(image_list[phase])
    label_matrix = np.zeros((num_images, len(catName_to_catID)))
    for i in range(num_images):
        cur_image = image_list[phase][i]
        label_indices = np.array(ann_dict[cur_image])
        label_matrix[i, label_indices] = 1.0
    np.save(os.path.join(args.save_path, 'formatted_' + phase + '_labels.npy'), label_matrix)
    np.save(os.path.join(args.save_path, 'formatted_' + phase + '_images.npy'), np.array(image_list[phase]))
