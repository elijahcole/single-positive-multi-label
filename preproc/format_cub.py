import os
import json
import numpy as np
import argparse
import pandas as pd

pp = argparse.ArgumentParser(description='Format CUB metadata.')
pp.add_argument('--load-path', type=str, default='../data/cub', help='Path to a directory containing a copy of the CUB dataset.')
pp.add_argument('--save-path', type=str, default='../data/cub', help='Path to output directory.')
args = pp.parse_args()

NUM_ATTRIBUTES = 312
NUM_INSTANCES = 11788

args.load_path = os.path.join(args.load_path, 'CUB_200_2011')

# get list of images: 
images_df = pd.read_csv(
    os.path.join(args.load_path, 'CUB_200_2011', 'images.txt'),
    delimiter = ' ',
    header = None,
    names = ['index', 'filename'],
    usecols = ['filename']
    )
images_np = images_df.to_numpy()
assert len(images_np) == NUM_INSTANCES

# get splits:
splits_df = pd.read_csv(
    os.path.join(args.load_path, 'CUB_200_2011', 'train_test_split.txt'),
    delimiter = ' ',
    header = None,
    names = ['index', 'is_train'],
    usecols = ['is_train']
    )
splits_np = splits_df.to_numpy()

# get classes:
classes_df = pd.read_csv(
    os.path.join(args.load_path, 'attributes.txt'),
    delimiter = ' ',
    header = None,
    names = ['index', 'attribute_name'],
    usecols = ['attribute_name']
    )
classes_np = classes_df.to_numpy()
assert len(classes_np) == 312

# get labels:
attributes_df = pd.read_csv(
    os.path.join(args.load_path, 'CUB_200_2011', 'attributes', 'image_attribute_labels.txt'),
    delimiter = ' ',
    header = None,
    names = ['image_id', 'attribute_id', 'is_present', 'certainty_id', 'time'],
    usecols = ['attribute_id', 'is_present']
    )
attributes_np = attributes_df.to_numpy()
assert len(attributes_np) == NUM_ATTRIBUTES * NUM_INSTANCES

labels_train = []
images_train = []
labels_test = []
images_test = []
k = 0
for i in range(NUM_INSTANCES):
    label_vector = []
    for j in range(NUM_ATTRIBUTES):
        label_vector.append(int(attributes_np[k, 1]))
        k += 1
    if splits_np[i] == 1:
        labels_train.append(label_vector)
        images_train.append(str(images_np[i][0]))
    else:
        labels_test.append(label_vector)
        images_test.append(str(images_np[i][0]))

np.save(os.path.join(args.save_path, 'formatted_train_labels.npy'), np.array(labels_train))
np.save(os.path.join(args.save_path, 'formatted_train_images.npy'), np.array(images_train))
np.save(os.path.join(args.save_path, 'formatted_val_labels.npy'), np.array(labels_test))
np.save(os.path.join(args.save_path, 'formatted_val_images.npy'), np.array(images_test))
