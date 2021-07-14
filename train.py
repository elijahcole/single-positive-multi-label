import os
import copy
import time
import json
import numpy as np
import torch
import datasets
import models
from losses import compute_batch_loss
import datetime
from instrumentation import train_logger

def run_train_phase(model, P, Z, logger, epoch, phase):
    
    '''
    Run one training phase.
    
    Parameters
    model: Model to train.
    P: Dictionary of parameters, which completely specify the training procedure.
    Z: Dictionary of temporary objects used during training.
    logger: Object used to track various metrics during training. 
    epoch: Integer index of the current epoch.
    phase: String giving the phase name
    '''
    
    assert phase == 'train'
    model.train()
    for batch in Z['dataloaders'][phase]:
        # move data to GPU: 
        batch['image'] = batch['image'].to(Z['device'], non_blocking=True)
        batch['labels_np'] = batch['label_vec_obs'].clone().numpy() # copy of labels for use in metrics
        batch['label_vec_obs'] = batch['label_vec_obs'].to(Z['device'], non_blocking=True)
        # forward pass: 
        Z['optimizer'].zero_grad()
        with torch.set_grad_enabled(True):
            # batch['logits'], batch['label_vec_est'] = model(batch)
            batch['logits'] = model.f(batch['image'])
            batch['preds'] = torch.sigmoid(batch['logits'])
            if batch['preds'].dim() == 1:
                batch['preds'] = torch.unsqueeze(batch['preds'], 0)
            batch['label_vec_est'] = model.g(batch['idx'])
            batch['preds_np'] = batch['preds'].clone().detach().cpu().numpy() # copy of preds for use in metrics
            batch = compute_batch_loss(batch, P, Z)
        # backward pass:
        batch['loss_tensor'].backward()
        Z['optimizer'].step()
        # save current batch data:
        logger.update_phase_data(batch)
    

def run_eval_phase(model, P, Z, logger, epoch, phase):
    
    '''
    Run one evaluation phase.
    
    Parameters
    model: Model to train.
    P: Dictionary of parameters, which completely specify the training procedure.
    Z: Dictionary of temporary objects used during training.
    logger: Object used to track various metrics during training. 
    epoch: Integer index of the current epoch.
    phase: String giving the phase name
    '''
    
    assert phase in ['val', 'test']
    model.eval()
    for batch in Z['dataloaders'][phase]:
        # move data to GPU: 
        batch['image'] = batch['image'].to(Z['device'], non_blocking=True)
        batch['labels_np'] = batch['label_vec_obs'].clone().numpy() # copy of labels for use in metrics
        batch['label_vec_obs'] = batch['label_vec_obs'].to(Z['device'], non_blocking=True)
        # forward pass: 
        with torch.set_grad_enabled(False):
            batch['logits'] = model.f(batch['image'])
            batch['preds'] = torch.sigmoid(batch['logits'])
            if batch['preds'].dim() == 1:
                batch['preds'] = torch.unsqueeze(batch['preds'], 0)
            batch['preds_np'] = batch['preds'].clone().detach().cpu().numpy() # copy of preds for use in metrics
            batch['loss_np'] = -1
            batch['reg_loss_np'] = -1
        # save current batch data:
        logger.update_phase_data(batch)

def train(model, P, Z):
    
    '''
    Train the model.
    
    Parameters
    P: Dictionary of parameters, which completely specify the training procedure.
    Z: Dictionary of temporary objects used during training.
    '''
    
    best_weights_f = copy.deepcopy(model.f.state_dict())
    best_weights_g = copy.deepcopy(model.g.state_dict())
    logger = train_logger(P) # initialize logger
    
    for epoch in range(P['num_epochs']):
        print('Epoch {}/{}'.format(epoch, P['num_epochs']-1))
        
        for phase in ['train', 'val', 'test']:
            # reset phase metrics:
            logger.reset_phase_data()
            
            # run one phase:
            t_init = time.time()
            if phase == 'train':
                run_train_phase(model, P, Z, logger, epoch, phase)
            else:
                run_eval_phase(model, P, Z, logger, epoch, phase)
                
            # save end-of-phase metrics:
            logger.compute_phase_metrics(phase, epoch, model.g.get_estimated_labels())
            
            # print epoch status:
            logger.report(t_init, time.time(), phase, epoch)
            
            # update best epoch, if applicable:
            new_best = logger.update_best_results(phase, epoch, P['val_set_variant'])
            if new_best:
                print('*** new best weights ***')
                best_weights_f = copy.deepcopy(model.f.state_dict())
                best_weights_g = copy.deepcopy(model.g.state_dict())
    
    print('')
    print('*** TRAINING COMPLETE ***')
    print('Best epoch: {}'.format(logger.best_epoch))
    print('Best epoch validation score: {:.2f}'.format(logger.get_stop_metric('val', logger.best_epoch, P['val_set_variant'])))
    print('Best epoch test score:       {:.2f}'.format(logger.get_stop_metric('test', logger.best_epoch, 'clean')))
    
    return P, model, logger, best_weights_f, best_weights_g

def initialize_training_run(P, feature_extractor, linear_classifier, estimated_labels):
    
    '''
    Set up for model training.
    
    Parameters
    P: Dictionary of parameters, which completely specify the training procedure.
    feature_extractor: Feature extractor model to start from.
    linear_classifier: Linear classifier model to start from.
    estimated_labels: NumPy array containing estimated training set labels to start from (for ROLE).
    '''
    
    os.makedirs(P['save_path'], exist_ok=True)
    np.random.seed(P['seed'])
    
    Z = {}
    
    # accelerator:
    Z['device'] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # data:
    Z['datasets'] = datasets.get_data(P)
    
    # observed label matrix:
    observed_label_matrix = Z['datasets']['train'].label_matrix_obs
    
    # save dataset-specific parameters:
    P['num_classes'] = Z['datasets']['train'].num_classes
    
    # dataloaders:
    Z['dataloaders'] = {}
    for phase in ['train', 'val', 'test']:
        Z['dataloaders'][phase] = torch.utils.data.DataLoader(
            Z['datasets'][phase],
            batch_size = P['bsize'],
            shuffle = phase == 'train',
            sampler = None,
            num_workers = P['num_workers'],
            drop_last = True
        )
        
    # model:
    model = models.MultilabelModel(P, feature_extractor, linear_classifier, observed_label_matrix, estimated_labels)
    
    # optimization objects:
    f_params = [param for param in list(model.f.parameters()) if param.requires_grad]
    g_params = [param for param in list(model.g.parameters()) if param.requires_grad]
    opt_params = [
        {'params': f_params, 'lr': P['lr']}, 
        {'params': g_params, 'lr': P['lr_mult'] * P['lr']}
        ]
    Z['optimizer'] = torch.optim.Adam(
        opt_params,
        lr = P['lr']
    )
    
    return P, Z, model

def execute_training_run(P, feature_extractor, linear_classifier, estimated_labels=None):
    
    '''
    Initialize, run the training process, and save the results.
    
    Parameters
    P: Dictionary of parameters, which completely specify the training procedure.
    feature_extractor: Feature extractor model to start from.
    linear_classifier: Linear classifier model to start from.
    estimated_labels: NumPy array containing estimated training set labels to start from (for ROLE).
    '''
    
    P, Z, model = initialize_training_run(P, feature_extractor, linear_classifier, estimated_labels)
    model.to(Z['device'])
    
    P, model, logger, best_weights_f, best_weights_g = train(model, P, Z)
    
    print('\nSaving best weights for f to {}/best_model_state_f.pt'.format(P['save_path']))
    torch.save(best_weights_f, os.path.join(P['save_path'], 'best_model_state_f.pt'))
    print('\nSaving best weights for g to {}/best_model_state_g.pt'.format(P['save_path']))
    torch.save(best_weights_g, os.path.join(P['save_path'], 'best_model_state_g.pt'))

    final_logs = logger.get_logs()
    print('\nSaving session data to {}/logs.json'.format(P['save_path']))
    with open(os.path.join(P['save_path'], 'logs.json'), 'w') as f:
        json.dump(final_logs, f)
    
    print('\nSaving session data to {}/params.json'.format(P['save_path']))
    with open(os.path.join(P['save_path'], 'params.json'), 'w') as f:
        json.dump(P, f)
                
    print('\nReverting model to best weights.')
    model.f.load_state_dict(best_weights_f)
    model.g.load_state_dict(best_weights_g)
    
    return model.f.feature_extractor, model.f.linear_classifier, model.g.get_estimated_labels(), final_logs

if __name__ == '__main__':
    
    lookup = {
        'feat_dim': {
            'resnet50': 2048
        },
        'expected_num_pos': {
            'pascal': 1.5,
            'coco': 2.5,
            'nuswide': 1.9,
            'cub': 31.4
        }
    }

    P = {}
    
    # Top-level parameters:
    P['dataset'] = 'pascal' # pascal, coco, nuswide, cub
    P['loss'] = 'role' # bce, bce_ls, iun, iu, pr, an, an_ls, wan, epr, role
    P['train_mode'] = 'linear_fixed_features' # linear_fixed_features, end_to_end, linear_init
    P['val_set_variant'] = 'clean' # clean, observed
    
    # Paths and filenames:
    P['experiment_name'] = 'multi_label_experiment'
    P['load_path'] = './data'
    P['save_path'] = './results'

    # Optimization parameters:
    P['lr'] = 1e-4 # learning rate
    P['bsize'] = 16 # batch size
    P['lr_mult'] = 10.0 # learning rate multiplier for the parameters of g
    P['stop_metric'] = 'map' # metric used to select the best epoch
    
    # Loss-specific parameters:
    P['ls_coef'] = 0.1 # label smoothing coefficient

    # Additional parameters:
    P['seed'] = 1200 # overall numpy seed
    P['use_pretrained'] = True # True, False
    P['num_workers'] = 4

    # Dataset parameters:
    P['split_seed'] = 1200 # seed for train / val splitting
    P['val_frac'] = 0.2 # fraction of train set to split off for val
    P['ss_seed'] = 999 # seed for subsampling
    P['ss_frac_train'] = 1.0 # fraction of training set to subsample
    P['ss_frac_val'] = 1.0 # fraction of val set to subsample
    
    # Dependent parameters:
    if P['loss'] in ['bce', 'bce_ls']:
        P['train_set_variant'] = 'clean'
    else:
        P['train_set_variant'] = 'observed'
    if P['train_mode'] == 'end_to_end':
        P['num_epochs'] = 10
        P['freeze_feature_extractor'] = False
        P['use_feats'] = False
        P['arch'] = 'resnet50'
    elif P['train_mode'] == 'linear_init':
        P['num_epochs'] = 25
        P['freeze_feature_extractor'] = True
        P['use_feats'] = True
        P['arch'] = 'linear'
    elif P['train_mode'] == 'linear_fixed_features':
        P['num_epochs'] = 25
        P['freeze_feature_extractor'] = True
        P['use_feats'] = True
        P['arch'] = 'linear'
    else:
        raise NotImplementedError('Unknown training mode.')
    P['feature_extractor_arch'] = 'resnet50'
    P['feat_dim'] = lookup['feat_dim'][P['feature_extractor_arch']]
    P['expected_num_pos'] = lookup['expected_num_pos'][P['dataset']]
    P['train_feats_file'] = './data/{}/train_features_imagenet_{}.npy'.format(P['dataset'], P['feature_extractor_arch'])
    P['val_feats_file'] = './data/{}/val_features_imagenet_{}.npy'.format(P['dataset'], P['feature_extractor_arch'])
    
    # run training process:
    best_params = None
    best_lr = None
    best_bsize = None
    best_val_score = - np.Inf
    best_test_score = None
    for bsize in [8, 16]:
        for lr in [1e-2, 1e-3, 1e-4, 1e-5]:
            now_str = datetime.datetime.now().strftime("%Y_%m_%d_%X").replace(':','-')
            P['bsize'] = bsize
            P['lr'] = lr
            P['save_path'] = './results/' + P['experiment_name'] + '_' + now_str + '_' + P['dataset']
            os.makedirs(P['save_path'], exist_ok=False)
            (feature_extractor, linear_classifier, estimated_labels, logs) = execute_training_run(P, feature_extractor=None, linear_classifier=None)
            if P['train_mode'] == 'linear_init':
                P['save_path'] = './results/' + P['experiment_name'] + '_' + now_str + '_' + P['dataset'] + '_fine_tuned_from_linear'
                os.makedirs(P['save_path'], exist_ok=False)
                P['train_mode'] = 'end_to_end'
                P['num_epochs'] = 10
                P['freeze_feature_extractor'] = False
                P['use_feats'] = False
                P['arch'] = 'resnet50'
                (feature_extractor, linear_classifier, estimated_labels, logs) = execute_training_run(P, feature_extractor=feature_extractor, linear_classifier=linear_classifier, estimated_labels=estimated_labels)
            # keep track of the best run: 
            best_epoch = np.argmax([logs['metrics']['val'][epoch][P['stop_metric'] + '_' + P['val_set_variant']] for epoch in range(P['num_epochs'])])
            val_score = logs['metrics']['val'][best_epoch][P['stop_metric'] + '_' + P['val_set_variant']]
            test_score = logs['metrics']['test'][best_epoch][P['stop_metric'] + '_clean']
            if val_score > best_val_score:
                best_val_score = val_score
                best_test_score = test_score
                best_params = copy.deepcopy(P)
    # report the best run:
    print('best run: {}'.format(best_params['save_path']))
    print('- learning rate: {}'.format(best_params['lr']))
    print('- batch size:    {}'.format(best_params['bsize']))
    print('- val score:     {}'.format(best_val_score))
    print('- test score:    {}'.format(best_test_score))
    
