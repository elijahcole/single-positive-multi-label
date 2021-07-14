import os
import json
import numpy as np
from PIL import Image
import torch
import copy
from torch.utils.data import Dataset
from torchvision import transforms

def get_metadata(dataset_name):
    if dataset_name == 'pascal':
        meta = {
            'num_classes': 20,
            'path_to_dataset': 'data/pascal',
            'path_to_images': 'data/pascal/VOCdevkit/VOC2012/JPEGImages'
        }
    elif dataset_name == 'coco':
        meta = {
            'num_classes': 80,
            'path_to_dataset': 'data/coco',
            'path_to_images': 'data/coco'
        }
    elif dataset_name == 'nuswide':
        meta = {
            'num_classes': 81,
            'path_to_dataset': 'data/nuswide',
            'path_to_images': ''
        }
    elif dataset_name == 'cub':
        meta = {
            'num_classes': 312,
            'path_to_dataset': 'data/cub',
            'path_to_images': ''
        }
    else:
        raise NotImplementedError('Metadata dictionary not implemented.')
    return meta

def get_imagenet_stats():
    '''
    Returns standard ImageNet statistics. 
    '''
    
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    return (imagenet_mean, imagenet_std)

def get_transforms():
    '''
    Returns image transforms.
    '''
    
    (imagenet_mean, imagenet_std) = get_imagenet_stats()
    tx = {}
    tx['train'] = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    tx['val'] = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    tx['test'] = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    
    return tx

def generate_split(num_ex, frac, rng):
    '''
    Computes indices for a randomized split of num_ex objects into two parts,
    so we return two index vectors: idx_1 and idx_2. Note that idx_1 has length
    (1.0 - frac)*num_ex and idx_2 has length frac*num_ex. Sorted index sets are 
    returned because this function is for splitting, not shuffling. 
    '''
    
    # compute size of each split:
    n_2 = int(np.round(frac * num_ex))
    n_1 = num_ex - n_2
    
    # assign indices to splits:
    idx_rand = rng.permutation(num_ex)
    idx_1 = np.sort(idx_rand[:n_1])
    idx_2 = np.sort(idx_rand[-n_2:])
    
    return (idx_1, idx_2)

def get_data(P):
    '''
    Given a parameter dictionary P, initialize and return the specified dataset. 
    '''
    
    # define transforms:
    tx = get_transforms()
    
    # select and return the right dataset:
    if P['dataset'] == 'coco':
        ds = multilabel(P, tx).get_datasets()
    elif P['dataset'] == 'pascal':
        ds = multilabel(P, tx).get_datasets()
    elif P['dataset'] == 'nuswide':
        raise NotImplementedError('Coming soon!')
    elif P['dataset'] == 'cub':
        raise NotImplementedError('Coming soon!')
    else:
        raise ValueError('Unknown dataset.')
    
    # Optionally overwrite the observed training labels with clean labels:
    assert P['train_set_variant'] in ['clean', 'observed']
    if P['train_set_variant'] == 'clean':
        print('Using clean labels for training.')
        ds['train'].label_matrix_obs = copy.deepcopy(ds['train'].label_matrix)
    else:
        print('Using single positive labels for training.')
    
    # Optionally overwrite the observed val labels with clean labels:
    assert P['val_set_variant'] in ['clean', 'observed']
    if P['val_set_variant'] == 'clean':
        print('Using clean labels for validation.')
        ds['val'].label_matrix_obs = copy.deepcopy(ds['val'].label_matrix)
    else:
        print('Using single positive labels for validation.')
    
    # We always use a clean test set:
    ds['test'].label_matrix_obs = copy.deepcopy(ds['test'].label_matrix)
            
    return ds

def load_data(base_path, P):
    data = {}
    for phase in ['train', 'val']:
        data[phase] = {}
        data[phase]['labels'] = np.load(os.path.join(base_path, 'formatted_{}_labels.npy'.format(phase)))
        data[phase]['labels_obs'] = np.load(os.path.join(base_path, 'formatted_{}_labels_obs.npy'.format(phase)))
        data[phase]['images'] = np.load(os.path.join(base_path, 'formatted_{}_images.npy'.format(phase)))
        data[phase]['feats'] = np.load(P['{}_feats_file'.format(phase)]) if P['use_feats'] else []
    return data

class multilabel:

    def __init__(self, P, tx):
        
        # get dataset metadata:
        meta = get_metadata(P['dataset'])
        self.base_path = meta['path_to_dataset']
        
        # load data:
        source_data = load_data(self.base_path, P)
        
        # generate indices to split official train set into train and val:
        split_idx = {}
        (split_idx['train'], split_idx['val']) = generate_split(
            len(source_data['train']['images']),
            P['val_frac'],
            np.random.RandomState(P['split_seed'])
            )
        
        # subsample split indices: # commenting this out makes the val set map be low?
        ss_rng = np.random.RandomState(P['ss_seed'])
        temp_train_idx = copy.deepcopy(split_idx['train'])
        for phase in ['train', 'val']:
            num_initial = len(split_idx[phase])
            num_final = int(np.round(P['ss_frac_{}'.format(phase)] * num_initial))
            split_idx[phase] = split_idx[phase][np.sort(ss_rng.permutation(num_initial)[:num_final])]
        
        # define train set:
        self.train = ds_multilabel(
            P['dataset'],
            source_data['train']['images'][split_idx['train']],
            source_data['train']['labels'][split_idx['train'], :],
            source_data['train']['labels_obs'][split_idx['train'], :],
            source_data['train']['feats'][split_idx['train'], :] if P['use_feats'] else [],
            tx['train'],
            P['use_feats']
        )
            
        # define val set:
        self.val = ds_multilabel(
            P['dataset'],
            source_data['train']['images'][split_idx['val']],
            source_data['train']['labels'][split_idx['val'], :],
            source_data['train']['labels_obs'][split_idx['val'], :],
            source_data['train']['feats'][split_idx['val'], :] if P['use_feats'] else [],
            tx['val'],
            P['use_feats']
        )
        
        # define test set:
        self.test = ds_multilabel(
            P['dataset'],
            source_data['val']['images'],
            source_data['val']['labels'],
            source_data['val']['labels_obs'],
            source_data['val']['feats'],
            tx['test'],
            P['use_feats']
        )
        
        # define dict of dataset lengths: 
        self.lengths = {'train': len(self.train), 'val': len(self.val), 'test': len(self.test)}
    
    def get_datasets(self):
        return {'train': self.train, 'val': self.val, 'test': self.test}

class ds_multilabel(Dataset):

    def __init__(self, dataset_name, image_ids, label_matrix, label_matrix_obs, feats, tx, use_feats):
        meta = get_metadata(dataset_name)
        self.num_classes = meta['num_classes']
        self.path_to_images = meta['path_to_images']
        
        self.image_ids = image_ids
        self.label_matrix = label_matrix
        self.label_matrix_obs = label_matrix_obs
        self.feats = feats
        self.tx = tx
        self.use_feats = use_feats

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self,idx):
        if self.use_feats:
            # Set I to be a feature vector:
            I = torch.FloatTensor(np.copy(self.feats[idx, :]))
        else:
            # Set I to be an image: 
            image_path = os.path.join(self.path_to_images, self.image_ids[idx])
            with Image.open(image_path) as I_raw:
                I = self.tx(I_raw.convert('RGB'))
        
        out = {
            'image': I,
            'label_vec_obs': torch.FloatTensor(np.copy(self.label_matrix_obs[idx, :])),
            'label_vec_true': torch.FloatTensor(np.copy(self.label_matrix[idx, :])),
            'idx': idx,
        }
        
        return out

'''
class data_COCO:

    def __init__(self,P,tx):
        self.base_path = get_data_path('coco')
        
        # load official train and val sets: 
        train2014_imgids = np.array(np.load(os.path.join(self.base_path,'annotations','formatted_train2014_imgids.npy')),dtype=int)
        train2014_labels = np.load(os.path.join(self.base_path,'annotations','formatted_train2014_labels.npy'))
        val2014_imgids = np.array(np.load(os.path.join(self.base_path,'annotations','formatted_val2014_imgids.npy')),dtype=int)
        val2014_labels = np.load(os.path.join(self.base_path,'annotations','formatted_val2014_labels.npy'))
        if P['use_feats']:
            train2014_feats = np.load(P['train_feats_file'])
            val2014_feats = np.load(P['test_feats_file'])
        
        # split train2014 into train and val:
        split_rng = np.random.RandomState(P['split_seed'])
        (train_idx,val_idx) = generate_split(
            len(train2014_imgids),
            P['val_frac'],
            split_rng
            )
        train_imgids = train2014_imgids[train_idx]
        train_labels = train2014_labels[train_idx,:]
        val_imgids = train2014_imgids[val_idx]
        val_labels = train2014_labels[val_idx,:]
        test_imgids = val2014_imgids
        test_labels = val2014_labels
        
        if P['use_feats']:
            train_feats = train2014_feats[train_idx,:]
            val_feats = train2014_feats[val_idx,:]
            test_feats = val2014_feats
        else:
            train_feats = []
            val_feats = []
            test_feats = []
        
        # subsample:
        ss_rng = np.random.RandomState(P['ss_seed'])
        
        ss_counts = {}
        ss_counts['train'] = int(np.round(P['ss_frac_train'] * len(train_imgids)))
        ss_counts['val'] = int(np.round(P['ss_frac_val'] * len(train_imgids)))
        ss_counts['test'] = int(np.round(P['ss_frac_test'] * len(train_imgids)))
        
        ss_idx = {}
        ss_idx['train'] = np.sort(ss_rng.permutation(len(train_imgids))[:ss_counts['train']])
        ss_idx['val'] = np.sort(ss_rng.permutation(len(val_imgids))[:ss_counts['val']])
        ss_idx['test'] = np.sort(ss_rng.permutation(len(test_imgids))[:ss_counts['test']])
        
        train_imgids = train_imgids[ss_idx['train']]
        train_labels = train_labels[ss_idx['train'],:]
        
        val_imgids = val_imgids[ss_idx['val']]
        val_labels = val_labels[ss_idx['val'],:]
        
        test_imgids = test_imgids[ss_idx['test']]
        test_labels = test_labels[ss_idx['test'],:]
        
        if P['use_feats']:
            train_feats = train_feats[ss_idx['train'],:]
            val_feats = val_feats[ss_idx['val'],:]
            test_feats = test_feats[ss_idx['test'],:]
        
        # define train set: 
        self.train = ds_multilabel_COCO(
            os.path.join(self.base_path,'train2014','COCO_train2014_'),
            train_imgids,
            train_labels,
            train_feats,
            tx['train'],
            P['use_feats']
            )
        
        # define val set: 
        self.val = ds_multilabel_COCO(
            os.path.join(self.base_path,'train2014','COCO_train2014_'),
            val_imgids,
            val_labels,
            val_feats,
            tx['val'],
            P['use_feats']
            )
        
        # define test set: 
        self.test = ds_multilabel_COCO(
            os.path.join(self.base_path,'val2014','COCO_val2014_'),
            test_imgids,
            test_labels,
            test_feats,
            tx['test'],
            P['use_feats']
            )
        
        # define dict of dataset lengths: 
        self.lengths = {'train': len(self.train), 'val': len(self.val), 'test': len(self.test)}

    def get_datasets(self):
        return {'train': self.train, 'val': self.val, 'test': self.test}

class ds_multilabel_COCO(Dataset):

    def __init__(self,image_path_base,imgids,label_matrix,feats,tx,use_feats):
        self.num_classes = 80
        self.image_path_base = image_path_base
        self.imgids = imgids
        self.label_matrix = label_matrix
        self.label_matrix_obs = None
        self.feats = feats
        self.tx = tx
        self.use_feats = use_feats

    def __len__(self):
        return len(self.imgids)

    def __getitem__(self,idx):
        imgid = self.imgids[idx]
        if self.use_feats:
            I = torch.FloatTensor(np.copy(self.feats[idx,:]))
        else:
            image_path = self.image_path_base + str(imgid).zfill(12) + '.jpg'
            with Image.open(image_path) as I_raw:
                I = self.tx(I_raw.convert('RGB'))
        out = {
            'image': I,
            'label_vec_obs': torch.FloatTensor(np.copy(self.label_matrix_obs[idx,:])),
            'label_vec_true': torch.FloatTensor(np.copy(self.label_matrix[idx,:])),
            'idx': idx,
        }
        return out

class data_NUSWIDE:

    def __init__(self,P,tx):
        self.base_path = get_data_path('nuswide')
        
        # load official train and val sets:
        nuswide_train_labels = np.load(os.path.join(self.base_path,'formatted_train_labels.npy'))
        nuswide_train_paths = np.load(os.path.join(self.base_path,'formatted_train_paths.npy'))
        nuswide_test_labels = np.load(os.path.join(self.base_path,'formatted_test_labels.npy'))
        nuswide_test_paths = np.load(os.path.join(self.base_path,'formatted_test_paths.npy'))

        if P['use_feats']:
            nuswide_train_feats = np.load(P['train_feats_file'])
            nuswide_val_feats = np.load(P['test_feats_file'])
        if P['generate_feats']:
            split_rng = np.random.RandomState(P['split_seed'])
            (idx_train,idx_val) = generate_split(
                len(nuswide_train_labels),
                P['val_frac'],
                split_rng
            )
            
            train_imgids = nuswide_train_paths[idx_train]
            train_labels = nuswide_train_labels[idx_train]
            val_imgids = nuswide_train_paths[idx_val]
            val_labels = nuswide_train_labels[idx_val]
            test_imgids = nuswide_test_paths
            test_labels = nuswide_test_labels
            
        else:
            # generate splits according to standard practice for NUS-WIDE:
            combined_paths = np.concatenate((nuswide_train_paths,nuswide_test_paths),axis=0)
            combined_labels = np.concatenate((nuswide_train_labels,nuswide_test_labels),axis=0)
            if P['use_feats']:
                combined_feats = np.concatenate((nuswide_train_feats,nuswide_val_feats),axis=0)
            num_dev = 150000 # this is standard for NUS-WIDE
            
            nus_rng = np.random.RandomState(101)
            idx_rand = nus_rng.permutation(len(combined_labels))
            idx_dev = idx_rand[:num_dev]
            idx_test = idx_rand[num_dev:]
            
            split_rng = np.random.RandomState(P['split_seed'])
            (dev_idx_train,dev_idx_val) = generate_split(
                num_dev,
                P['val_frac'],
                split_rng
            )
            idx_train = idx_dev[dev_idx_train]
            idx_val = idx_dev[dev_idx_val]
            
            train_imgids = combined_paths[idx_train]
            train_labels = combined_labels[idx_train]
            val_imgids = combined_paths[idx_val]
            val_labels = combined_labels[idx_val]
            test_imgids = combined_paths[idx_test]
            test_labels = combined_labels[idx_test]
            
        if P['use_feats']:
            train_feats = combined_feats[idx_train,:]
            val_feats =  combined_feats[idx_val,:]
            test_feats = combined_feats[idx_test,:]
        else:
            train_feats = []
            val_feats = []
            test_feats = []
        
        # subsample:
        ss_rng = np.random.RandomState(P['ss_seed'])
        
        ss_counts = {}
        ss_counts['train'] = int(np.round(P['ss_frac_train'] * len(train_imgids)))
        ss_counts['val'] = int(np.round(P['ss_frac_val'] * len(train_imgids)))
        ss_counts['test'] = int(np.round(P['ss_frac_test'] * len(train_imgids)))
        
        ss_idx = {}
        ss_idx['train'] = np.sort(ss_rng.permutation(len(train_imgids))[:ss_counts['train']])
        ss_idx['val'] = np.sort(ss_rng.permutation(len(val_imgids))[:ss_counts['val']])
        ss_idx['test'] = np.sort(ss_rng.permutation(len(test_imgids))[:ss_counts['test']])
        
        train_imgids = train_imgids[ss_idx['train']]
        train_labels = train_labels[ss_idx['train'],:]
        
        val_imgids = val_imgids[ss_idx['val']]
        val_labels = val_labels[ss_idx['val'],:]
        
        test_imgids = test_imgids[ss_idx['test']]
        test_labels = test_labels[ss_idx['test'],:]
        
        if P['use_feats']:
            train_feats = train_feats[ss_idx['train'],:]
            val_feats = val_feats[ss_idx['val'],:]
            test_feats = test_feats[ss_idx['test'],:]
        
        # define train set:
        self.train = ds_multilabel_NUSWIDE(
            os.path.join(self.base_path,'image_448'),
            train_imgids,
            train_labels,
            train_feats,
            tx['train'],
            P['use_feats']
        )
        
        # define val set:
        self.val = ds_multilabel_NUSWIDE(
            os.path.join(self.base_path,'image_448'),
            val_imgids,
            val_labels,
            val_feats,
            tx['val'],
            P['use_feats']
        )
        
        # define test set:
        self.test = ds_multilabel_NUSWIDE(
            os.path.join(self.base_path,'image_448'),
            test_imgids,
            test_labels,
            test_feats,
            tx['test'],
            P['use_feats']
        )
        
        self.lengths = {'train': len(self.train), 'val': len(self.val), 'test': len(self.test)}
    
    def get_datasets(self):
        return {'train': self.train, 'val': self.val, 'test': self.test}
        
class ds_multilabel_NUSWIDE(Dataset):

    def __init__(self,image_path_base,imgids,label_matrix,feats,tx,use_feats):
        self.num_classes = 81
        self.image_path_base = image_path_base
        self.imgids = imgids
        self.label_matrix = label_matrix
        self.label_matrix_obs = None
        self.feats = feats
        self.tx = tx
        self.use_feats = use_feats

    def __len__(self):
        return len(self.imgids)

    def __getitem__(self,idx):
        imgid = self.imgids[idx]
        if self.use_feats:
            I = torch.FloatTensor(np.copy(self.feats[idx,:]))
        else:
            image_path = os.path.join(self.image_path_base,self.imgids[idx])
            with Image.open(image_path) as I_raw:
                I = self.tx(I_raw.convert('RGB'))

        out = {
            'image': I,
            'label_vec_obs': torch.FloatTensor(np.copy(self.label_matrix_obs[idx,:])),
            'label_vec_true': torch.FloatTensor(np.copy(self.label_matrix[idx,:])),
            'idx': idx,
        }
        return out

class data_CUB:

    def __init__(self,P,tx):
        self.base_path = get_data_path('cub')
        self.image_path = os.path.join(self.base_path, 'CUB_200_2011', 'images-448')

        # load data per official splits
        official_train_labels = np.load(os.path.join(self.base_path, 'CUB_200_2011', 'formatted_train_labels.npy'))
        official_train_images = np.load(os.path.join(self.base_path, 'CUB_200_2011', 'formatted_train_images.npy'), allow_pickle=True)
        official_train_images = np.array([x[0] for x in official_train_images])
        official_test_labels = np.load(os.path.join(self.base_path, 'CUB_200_2011', 'formatted_test_labels.npy'))
        official_test_images = np.load(os.path.join(self.base_path, 'CUB_200_2011', 'formatted_test_images.npy'), allow_pickle=True)
        official_test_images = np.array([x[0] for x in official_test_images])
        if P['use_feats']:
            cub_train_feats = np.load(P['train_feats_file'])
            cub_test_feats = np.load(P['test_feats_file'])

        # split official train set into train and val
        split_rng = np.random.RandomState(P['split_seed'])
        (train_idx, val_idx) = generate_split(
            len(official_train_images),
            P['val_frac'],
            split_rng
            )

        train_imgids = official_train_images[train_idx]
        train_labels = official_train_labels[train_idx, :]
        val_imgids = official_train_images[val_idx]
        val_labels = official_train_labels[val_idx, :]
        test_imgids = official_test_images
        test_labels = official_test_labels

        if P['use_feats']:
            train_feats = cub_train_feats[train_idx, :]
            val_feats = cub_train_feats[val_idx, :]
            test_feats = cub_test_feats
        else:
            train_feats = []
            val_feats = []
            test_feats = []

        # subsample:
        ss_rng = np.random.RandomState(P['ss_seed'])

        ss_counts = {}
        ss_counts['train'] = int(np.round(P['ss_frac_train'] * len(train_imgids)))
        ss_counts['val'] = int(np.round(P['ss_frac_val'] * len(val_imgids)))
        ss_counts['test'] = int(np.round(P['ss_frac_test'] * len(test_imgids)))

        ss_idx = {}
        ss_idx['train'] = np.sort(ss_rng.permutation(len(train_imgids))[:ss_counts['train']])
        ss_idx['val'] = np.sort(ss_rng.permutation(len(val_imgids))[:ss_counts['val']])
        ss_idx['test'] = np.sort(ss_rng.permutation(len(test_imgids))[:ss_counts['test']])

        train_imgids = train_imgids[ss_idx['train']]
        train_labels = train_labels[ss_idx['train'],:]

        val_imgids = val_imgids[ss_idx['val']]
        val_labels = val_labels[ss_idx['val'],:]

        test_imgids = test_imgids[ss_idx['test']]
        test_labels = test_labels[ss_idx['test'],:]

        if P['use_feats']:
            train_feats = train_feats[ss_idx['train'],:]
            val_feats = val_feats[ss_idx['val'],:]
            test_feats = test_feats[ss_idx['test'],:]

        # define train set:
        self.train = ds_multilabel_CUB(
            self.image_path,
            train_imgids,
            train_labels,
            train_feats,
            tx['train'],
            P['use_feats']
        )

        # define val set:
        self.val = ds_multilabel_CUB(
            self.image_path,
            val_imgids,
            val_labels,
            val_feats,
            tx['val'],
            P['use_feats']
        )

        # define test set:
        self.test = ds_multilabel_CUB(
            self.image_path,
            test_imgids,
            test_labels,
            test_feats,
            tx['test'],
            P['use_feats']
        )

        # define dict of dataset lengths:
        self.lengths = {'train': len(self.train), 'val': len(self.val), 'test': len(self.test)}

    def get_datasets(self):
        return {'train': self.train, 'val': self.val, 'test': self.test}

class ds_multilabel_CUB(Dataset):

    def __init__(self,image_path_base,imgids,label_matrix,feats,tx,use_feats):
        self.num_classes = 312
        self.image_path_base = image_path_base
        self.imgids = imgids
        self.label_matrix = label_matrix
        self.label_matrix_obs = None
        self.feats = feats
        self.tx = tx
        self.use_feats = use_feats

    def __len__(self):
        return len(self.imgids)

    def __getitem__(self,idx):
        imgid = self.imgids[idx]
        if self.use_feats:
            I = torch.FloatTensor(np.copy(self.feats[idx,:]))
        else:
            image_path = os.path.join(self.image_path_base,self.imgids[idx])
            with Image.open(image_path) as I_raw:
                I = self.tx(I_raw.convert('RGB'))

        out = {
            'image': I,
            'label_vec_obs': torch.FloatTensor(np.copy(self.label_matrix_obs[idx,:])),
            'label_vec_true': torch.FloatTensor(np.copy(self.label_matrix[idx,:])),
            'idx': idx,
        }
        return out
'''