#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2020 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

""" TCN for lipreading"""

import os
import time
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from lipreading.utils import get_save_folder
from lipreading.utils import load_json, save2npz
from lipreading.utils import load_model, CheckpointSaver
from lipreading.utils import get_logger, update_logger_batch
from lipreading.utils import showLR, calculateNorm2, AverageMeter
from lipreading.model import Lipreading
from lipreading.mixup import mixup_data, mixup_criterion
from lipreading.optim_utils import get_optimizer, CosineScheduler
from lipreading.dataloaders import get_data_loaders, get_preprocessing_pipelines

from datetime import datetime #ADDED
from sklearn.metrics import confusion_matrix, classification_report # added
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.tensorboard import SummaryWriter
import pandas as pd


# ADDED
def check_gpu_memory():
    """Prints current, allocated, and peak memory usage on GPU"""
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # Convert bytes to MB
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)  # Convert bytes to MB
    max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Peak usage

    print(f"🔹 GPU Memory Usage: Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB | Peak: {max_allocated:.2f} MB")


def load_args(default_config=None):
    parser = argparse.ArgumentParser(description='Pytorch Lipreading ')
    
    # LRS2 arguments
    """
    parser.add_argument('--dataset', default='lrs2', help='dataset selection')
    parser.add_argument('--num-classes', type=int, default=6834, help='Number of classes')
    parser.add_argument('--data-dir', default='./datasets/lrs2_words_v2', help='Path to LRS2 dataset')
    parser.add_argument('--label-path', type=str, default='./labels/6834Lrs2List.txt', help='Path to LRS2 labels')
    parser.add_argument('--annonation-direc', default='./datasets/lrs2_words_v2', help='Loaded data directory')
    """

    #LRS3 arguments
    
    parser.add_argument('--dataset', default='lrs3', help='dataset selection')
    #parser.add_argument('--num-classes', type=int, default=6070, help='Number of classes')
    parser.add_argument('--data-dir', default='./datasets/lrs3_words_may', help='Path to LRS3 dataset')
    #parser.add_argument('--label-path', type=str, default='./labels/6070Lrs3List.txt', help='Path to LRS3 labels')
    parser.add_argument('--annonation-direc', default='./datasets/lrs3_words_may', help='Loaded data directory')
    
    
    parser.add_argument('--num-classes', type=int, default=86, help='Number of classes')
    parser.add_argument('--label-path', type=str, default='./labels/86Lrs3List.txt', help='Path to LRS3 labels')
    
    # OULUVS2 arguments
    # test djfsjgf
    """
    parser.add_argument('--dataset', default='ouluvs2', help='dataset selection')
    parser.add_argument('--num-classes', type=int, default=1203, help='Number of classes')
    parser.add_argument('--data-dir', default='./datasets/ouluvs2_words_v2', help='Path to OuluVS2 dataset')
    parser.add_argument('--label-path', type=str, default='./labels/1203Ouluvs2List.txt', help='Path to LRS3 labels')
    parser.add_argument('--annonation-direc', default='./datasets/ouluvs2_words_v2', help='Loaded data directory')
    """
    # -- dataset config
    """
    parser.add_argument('--dataset', default='lrw', help='dataset selection')
    parser.add_argument('--num-classes', type=int, default=500, help='Number of classes')
    """
    parser.add_argument('--modality', default='video', choices=['video', 'audio'], help='choose the modality')
    # -- directory
    """
    parser.add_argument('--data-dir', default='./datasets/LRW_h96w96_mouth_crop_gray', help='Loaded data directory')
    parser.add_argument('--label-path', type=str, default='./labels/500WordsSortedList.txt', help='Path to txt file with labels')
    parser.add_argument('--annonation-direc', default=None, help='Loaded data directory')
    """

    # -- model config
    parser.add_argument('--backbone-type', type=str, default='resnet', choices=['resnet', 'shufflenet'], help='Architecture used for backbone')
    parser.add_argument('--relu-type', type=str, default='relu', choices=['relu','prelu'], help='what relu to use' )
    parser.add_argument('--width-mult', type=float, default=1.0, help='Width multiplier for mobilenets and shufflenets')
    # -- TCN config
    parser.add_argument('--tcn-kernel-size', type=int, nargs="+", help='Kernel to be used for the TCN module')
    parser.add_argument('--tcn-num-layers', type=int, default=4, help='Number of layers on the TCN module')
    parser.add_argument('--tcn-dropout', type=float, default=0.2, help='Dropout value for the TCN module')
    # lrs2
    #parser.add_argument('--tcn-dropout', type=float, default=0.3, help='Dropout value for the TCN module')
    parser.add_argument('--tcn-dwpw', default=False, action='store_true', help='If True, use the depthwise seperable convolution in TCN architecture')
    parser.add_argument('--tcn-width-mult', type=int, default=1, help='TCN width multiplier')
    # -- DenseTCN config
    parser.add_argument('--densetcn-block-config', type=int, nargs = "+", help='number of denselayer for each denseTCN block')
    parser.add_argument('--densetcn-kernel-size-set', type=int, nargs = "+", help='kernel size set for each denseTCN block')
    parser.add_argument('--densetcn-dilation-size-set', type=int, nargs = "+", help='dilation size set for each denseTCN block')
    parser.add_argument('--densetcn-growth-rate-set', type=int, nargs = "+", help='growth rate for DenseTCN')
    parser.add_argument('--densetcn-dropout', default=0.2, type=float, help='Dropout value for DenseTCN')
    parser.add_argument('--densetcn-reduced-size', default=256, type=int, help='the feature dim for the output of reduce layer')
    parser.add_argument('--densetcn-se', default = False, action='store_true', help='If True, enable SE in DenseTCN')
    parser.add_argument('--densetcn-condense', default = False, action='store_true', help='If True, enable condenseTCN')
    # -- train
    parser.add_argument('--training-mode', default='tcn', help='tcn')
    #parser.add_argument('--batch-size', type=int, default=32, help='Mini-batch size')
    parser.add_argument('--batch-size', type=int, default=16, help='Mini-batch size')
    parser.add_argument('--optimizer',type=str, default='adamw', choices = ['adam','sgd','adamw'])
    # parser.add_argument('--lr', default=3e-4, type=float, help='initial learning rate')
    parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')
    #parser.add_argument('--lr', default=5e-5, type=float, help='initial learning rate')
    #parser.add_argument('--lr', default=2.5e-5, type=float, help='initial learning rate')
    parser.add_argument('--init-epoch', default=0, type=int, help='epoch to start at')
    # parser.add_argument('--epochs', default=80, type=int, help='number of epochs')
    # parser.add_argument('--epochs', default=10, type=int, help='number of epochs')
    parser.add_argument('--epochs', default=10, type=int, help='number of epochs')
    parser.add_argument('--test', default=False, action='store_true', help='training mode')
    # -- mixup
    parser.add_argument('--alpha', default=0.4, type=float, help='interpolation strength (uniform=1., ERM=0.)')
    # -- test
    # added model path
    parser.add_argument('--model-path', type=str, default='./models/lrw_resnet18_dctcn_video_boundary.pth', help='Pretrained model pathname')
    parser.add_argument('--allow-size-mismatch', default=True, action='store_true',
                        help='If True, allows to init from model with mismatching weight tensors. Useful to init from model with diff. number of classes')
    # -- feature extractor
    parser.add_argument('--extract-feats', default=False, action='store_true', help='Feature extractor')
    parser.add_argument('--mouth-patch-path', type=str, default=None, help='Path to the mouth ROIs, assuming the file is saved as numpy.array')
    parser.add_argument('--mouth-embedding-out-path', type=str, default=None, help='Save mouth embeddings to a specificed path')
    # -- json pathname
    parser.add_argument('--config-path', type=str, default=None, help='Model configuration with json format')
    # -- other vars
    parser.add_argument('--interval', default=50, type=int, help='display interval')
    parser.add_argument('--workers', default=8, type=int, help='number of data loading workers')
    # paths
    parser.add_argument('--logging-dir', type=str, default='./train_logs', help = 'path to the directory in which to save the log file')
    # use boundaries
    parser.add_argument('--use-boundary', default=False, action='store_true', help='include hard border at the testing stage.')

    args = parser.parse_args()
    return args


args = load_args()

torch.manual_seed(1)
np.random.seed(1)
random.seed(1)
torch.backends.cudnn.benchmark = True


def extract_feats(model):
    """
    :rtype: FloatTensor
    """
    model.eval()
    preprocessing_func = get_preprocessing_pipelines()['test']
    data = preprocessing_func(np.load(args.mouth_patch_path)['data'])  # data: TxHxW
    return model(torch.FloatTensor(data)[None, None, :, :, :].cuda(), lengths=[data.shape[0]])

# added
def save_files(all_labels, all_preds, class_names, pred_path, log_path, log_path2, epoch, save_path):
    os.makedirs((save_path + ("/reports/" + str(epoch))), exist_ok=True)
    # added: =========================
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    # Store predictions and actual labels in a file with emojis
    with open(pred_path, "w") as pred_file:
        pred_file.write("Sample\tTrue Label\tPredicted Label\tResult\n")
        pred_file.write("=" * 60 + "\n")
        
        for idx, (true_label, pred_label) in enumerate(zip(all_labels, all_preds)):
            result_emoji = "✅" if true_label == pred_label else "❌"
            pred_file.write(f"{idx + 1}\t{class_names[true_label]}\t{class_names[pred_label]}\t{result_emoji}\n")
    
    print(f"🔹 Predictions and actual labels saved at: {pred_path}")


    unique_labels = sorted(set(all_labels) | set(all_preds))
    #unique_labels = np.unique(all_labels) # Only labels that appear in the test set
    filtered_class_names = [class_names[i] for i in unique_labels]
    report = classification_report(all_labels, all_preds, target_names=filtered_class_names, digits=4, zero_division=0, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    cm_df = pd.DataFrame(cm, index=[f"True {label}" for label in filtered_class_names], 
                         columns=[f"Pred {label}" for label in filtered_class_names])

    df_report.to_csv(log_path, index=True)
    cm_df.to_csv(log_path2, index=True)
    print(f"Saved report with confusion matrix for epoch {epoch} at: {log_path} and {log_path2}")
    

def evaluate(model, dset_loader, criterion, save_path, epoch):
    model.eval()

    running_loss = 0.
    running_corrects = 0.

    # added:
    all_preds = []
    all_labels = []
    class_names = []
    with open(args.label_path, 'r') as file:
        class_names = [line.strip() for line in file.readlines()]
        
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(dset_loader)):
            if args.use_boundary:
                input, lengths, labels, boundaries = data
                boundaries = boundaries.cuda()
            else:
                input, lengths, labels = data
                boundaries = None
            logits = model(input.unsqueeze(1).cuda(), lengths=lengths, boundaries=boundaries)
            _, preds = torch.max(F.softmax(logits, dim=1).data, dim=1)

            # ADDED: collect true labels and predictions
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
            running_corrects += preds.eq(labels.cuda().view_as(preds)).sum().item()

            loss = criterion(logits, labels.cuda())

            running_loss += loss.item() * input.size(0)
    start_path = save_path + '/reports/' + str(epoch)
    pred_path = start_path + '/predictions_labels_test.csv'
    log_path = start_path + '/classification_report_test.csv'
    log_path2 = start_path + '/confusion_matrix_test.csv'
    save_files(all_labels, all_preds, class_names, pred_path, log_path, log_path2, epoch, save_path)

    print(f"{len(dset_loader.dataset)} in total\tCR: {running_corrects/len(dset_loader.dataset)}")
    return running_corrects/len(dset_loader.dataset), running_loss/len(dset_loader.dataset)


def train(model, dset_loader, criterion, epoch, optimizer, logger, writer, save_path):
    data_time = AverageMeter()
    batch_time = AverageMeter()

    lr = showLR(optimizer)

    logger.info('-' * 10)
    logger.info(f"Epoch {epoch}/{args.epochs - 1}")
    logger.info(f"Current learning rate: {lr}")

    # added: Log learning rate at the start of the epoch
    writer.add_scalar("Learning Rate", lr, epoch)
    
    model.train()
    running_loss = 0.
    running_corrects = 0.
    running_all = 0.

    end = time.time()

    # added:
    all_preds = []
    all_labels = []
    class_names = []
    with open(args.label_path, 'r') as file:
        class_names = [line.strip() for line in file.readlines()]
        
    for batch_idx, data in enumerate(dset_loader):
        if args.use_boundary:
            input, lengths, labels, boundaries = data
            boundaries = boundaries.cuda()
        else:
            input, lengths, labels = data
            boundaries = None
        # measure data loading time
        data_time.update(time.time() - end)

        # --
        input, labels_a, labels_b, lam = mixup_data(input, labels, args.alpha)
        labels_a, labels_b = labels_a.cuda(), labels_b.cuda()

        optimizer.zero_grad()
        
        # forward pass
        logits = model(input.unsqueeze(1).cuda(), lengths=lengths, boundaries=boundaries)

        loss_func = mixup_criterion(labels_a, labels_b, lam)
        '''
        print(f"Logits shape: {logits.shape}")  # Should be [batch_size, num_classes]
        # Print the weight tensor shape (from criterion)
        print(f"Weight tensor shape: {criterion.weight.shape}")
        # Verify the number of classes you are using
        print(f"Number of classes in model: {args.num_classes}")
        '''
        loss = loss_func(criterion, logits)

        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        # -- compute running performance
        _, predicted = torch.max(F.softmax(logits, dim=1).data, dim=1)
        # added =========
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        
        
        running_loss += loss.item()*input.size(0)
        running_corrects += lam * predicted.eq(labels_a.view_as(predicted)).sum().item() + (1 - lam) * predicted.eq(labels_b.view_as(predicted)).sum().item()
        running_all += input.size(0)
        # -- log intermediate results
        if batch_idx % args.interval == 0 or (batch_idx == len(dset_loader)-1):
            update_logger_batch( args, logger, dset_loader, batch_idx, running_loss, running_corrects, running_all, batch_time, data_time )

    start_path = save_path + '/reports/' + str(epoch)
    pred_path = start_path + '/predictions_labels_train.csv'
    log_path = start_path + '/classification_report_train.csv'
    log_path2 = start_path + '/confusion_matrix_train.csv'
    save_files(all_labels, all_preds, class_names, pred_path, log_path, log_path2, epoch, save_path)
    # ==============

    # added: Log epoch-level metrics
    epoch_loss = running_loss / running_all
    epoch_acc = running_corrects / running_all
    #writer.add_scalar("Loss/train_epoch", epoch_loss, epoch)
    #writer.add_scalar("Accuracy/train_epoch", epoch_acc, epoch)
    writer.add_scalars('Train', {
            'Accuracy': epoch_acc,
            'Loss': epoch_loss
        }, epoch)
    
    return model


def get_model_from_json():
    assert args.config_path.endswith('.json') and os.path.isfile(args.config_path), \
        f"'.json' config path does not exist. Path input: {args.config_path}"
    args_loaded = load_json( args.config_path)
    args.backbone_type = args_loaded['backbone_type']
    args.width_mult = args_loaded['width_mult']
    args.relu_type = args_loaded['relu_type']
    args.use_boundary = args_loaded.get("use_boundary", False)

    if args_loaded.get('tcn_num_layers', ''):
        tcn_options = { 'num_layers': args_loaded['tcn_num_layers'],
                        'kernel_size': args_loaded['tcn_kernel_size'],
                        'dropout': args_loaded['tcn_dropout'],
                        'dwpw': args_loaded['tcn_dwpw'],
                        'width_mult': args_loaded['tcn_width_mult'],
                      }
    else:
        tcn_options = {}
    if args_loaded.get('densetcn_block_config', ''):
        densetcn_options = {'block_config': args_loaded['densetcn_block_config'],
                            'growth_rate_set': args_loaded['densetcn_growth_rate_set'],
                            'reduced_size': args_loaded['densetcn_reduced_size'],
                            'kernel_size_set': args_loaded['densetcn_kernel_size_set'],
                            'dilation_size_set': args_loaded['densetcn_dilation_size_set'],
                            'squeeze_excitation': args_loaded['densetcn_se'],
                            'dropout': args_loaded['densetcn_dropout'],
                            }
    else:
        densetcn_options = {}

    model = Lipreading( modality=args.modality,
                        num_classes=args.num_classes,
                        tcn_options=tcn_options,
                        densetcn_options=densetcn_options,
                        backbone_type=args.backbone_type,
                        relu_type=args.relu_type,
                        width_mult=args.width_mult,
                        use_boundary=args.use_boundary,
                        extract_feats=args.extract_feats).cuda()
    calculateNorm2(model)
    return model

# added
def get_loss_class_weights():
    word_counts_train = []
    unique_words = []
    file = "86lrs3_word_counts_sets.txt"

    # change 
    with open(file, "r") as f:
        for line in f:
            parts = line.strip().split(" -> ")
            if len(parts) == 2:
                counts = parts[1].split(", ")  # split Train, Val, Test counts
                #print(counts[0])
                unique_words.append(parts[0])  # Store unique words
                train_count = int(counts[0].split(": ")[1])  # extract the train count
                word_counts_train.append(train_count)

    '''
    # Convert to tensor
    counts_tensor = torch.tensor(word_counts_train, dtype=torch.float)
    total_instances = counts_tensor.sum()  # Total training samples
    num_classes = len(word_counts_train)  # Total number of unique words in training

    # compute class weights using (total instances / (num classes * class frequency))
    class_weights = total_instances / (num_classes * counts_tensor)
    class_weights = class_weights.to(torch.float).to('cuda' if torch.cuda.is_available() else 'cpu')
    '''

    word_counts_train = np.array(word_counts_train)
    class_labels = np.arange(len(unique_words))
    class_weights = compute_class_weight(class_weight="balanced", classes=class_labels, y=np.repeat(class_labels, word_counts_train))
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    return class_weights_tensor
    
def main():

    # -- logging
    save_path = get_save_folder( args)
    print(f"Model and log being saved in: {save_path}")
    logger = get_logger(args, save_path)
    ckpt_saver = CheckpointSaver(save_path)

    # added: Initialize TensorBoard writer
    tb_log_dir = os.path.join(save_path, "tensorboard_logs")
    writer = SummaryWriter(log_dir=tb_log_dir)
    
    # -- get model
    model = get_model_from_json()

    # ADDED: check model mem before training
    #print("\n🔍 Checking GPU Memory Before Training:")
    #check_gpu_memory()
        
    # -- get dataset iterators
    dset_loaders = get_data_loaders(args)
    # -- get loss function
    #criterion = nn.CrossEntropyLoss()
    # added -----------------------------
    class_weights = get_loss_class_weights()
    criterion_train = nn.CrossEntropyLoss(weight=class_weights)
    criterion_eval = nn.CrossEntropyLoss()
    # -- get optimizer
    optimizer = get_optimizer(args, optim_policies=model.parameters())
    # -- get learning rate scheduler
    scheduler = CosineScheduler(args.lr, args.epochs)

    print('checking model path')
    if args.model_path:
        print('has model path')
        assert os.path.isfile(args.model_path), \
            f"Model path does not exist. Path input: {args.model_path}"
        # resume from checkpoint
        if args.init_epoch > 0:
            model, optimizer, epoch_idx, ckpt_dict = load_model(args.model_path, model, optimizer)
            args.init_epoch = epoch_idx
            ckpt_saver.set_best_from_ckpt(ckpt_dict)
            logger.info(f'Model and states have been successfully loaded from {args.model_path}')
        # init from trained model
        else:
            model = load_model(args.model_path, model, allow_size_mismatch=args.allow_size_mismatch)
            logger.info(f'Model has been successfully loaded from {args.model_path}')
        # feature extraction
        if args.mouth_patch_path:
            save2npz( args.mouth_embedding_out_path, data = extract_feats(model).cpu().detach().numpy())
            return
        # if test-time, performance on test partition and exit. Otherwise, performance on validation and continue (sanity check for reload)
        if args.test:
            acc_avg_test, loss_avg_test = evaluate(model, dset_loaders['test'], criterion_eval, save_path, 'test')
            logger.info(f"Test-time performance on partition {'test'}: Loss: {loss_avg_test:.4f}\tAcc:{acc_avg_test:.4f}")
            return

    # -- fix learning rate after loading the ckeckpoint (latency)
    if args.model_path and args.init_epoch > 0:
        scheduler.adjust_lr(optimizer, args.init_epoch-1)

    epoch = args.init_epoch

    # -- ADDED
    log_file = save_path + "/val_metrics.txt"
    # check if file exists, if not, create it with a header including date & time
    with open(log_file, "w") as f:
        current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        f.write(f"# Validation Metrics Log - Created on {current_time}\n")
            
    while epoch < args.epochs:
        model = train(model, dset_loaders['train'], criterion_train, epoch, optimizer, logger, writer, save_path)

        # added: Check memory after training epoch
        #print(f"\n📊 GPU Memory Usage After Training Epoch {epoch}:")
        #check_gpu_memory()
        
        acc_avg_val, loss_avg_val = evaluate(model, dset_loaders['val'], criterion_eval, save_path, epoch)
        logger.info(f"{'val'} Epoch:\t{epoch:2}\tLoss val: {loss_avg_val:.4f}\tAcc val:{acc_avg_val:.4f}, LR: {showLR(optimizer)}")

        # added: log epoch metrics to TensorBoard
        #writer.add_scalar("Loss/val_epoch", loss_avg_val, epoch)
        #writer.add_scalar("Accuracy/val_epoch", acc_avg_val, epoch)
        writer.add_scalars('Validation', {
            'Accuracy': acc_avg_val,
            'Loss': loss_avg_val
        }, epoch)

        # ADDED -- save loss & accuracy to a text file
        with open(log_file, "a") as f:  # "a" for append mode
            f.write(f"{'val'} Epoch:\t{epoch:2}\tLoss val: {loss_avg_val:.4f}\tAcc val:{acc_avg_val:.4f}, LR: {showLR(optimizer)}\n")
    
        # -- save checkpoint
        save_dict = {
            'epoch_idx': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        ckpt_saver.save(save_dict, acc_avg_val)
        scheduler.adjust_lr(optimizer, epoch)
        epoch += 1

    # added: Close the TensorBoard writer after training
    writer.close()

    # -- evaluate best-performing epoch on test partition
    best_fp = os.path.join(ckpt_saver.save_dir, ckpt_saver.best_fn)
    _ = load_model(best_fp, model)
    acc_avg_test, loss_avg_test = evaluate(model, dset_loaders['test'], criterion_eval, save_path, 'test')
    logger.info(f"Test time performance of best epoch: {acc_avg_test} (loss: {loss_avg_test})")

if __name__ == '__main__':
    main()