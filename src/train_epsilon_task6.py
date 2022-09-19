"""

Training and validating models

"""
import argparse
import collections
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.transform
from tqdm import tqdm

import metric
import pydicom
import pytorch_retinanet.dataloader
import pytorch_retinanet.model
import pytorch_retinanet.model_dpn
import pytorch_retinanet.model_incresv2
import pytorch_retinanet.model_nasnet_mobile
import pytorch_retinanet.model_pnasnet
import pytorch_retinanet.model_resnet
import pytorch_retinanet.model_se_resnext
import pytorch_retinanet.model_xception
import torch
from config import IMG_SIZE, RESULTS_DIR, TEST_DIR, WEIGHTS_DIR
from datasets.detection_dataset import DetectionDataset
from datasets.dataset_valid import DatasetValid
from models import MODELS
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from utils.logger import Logger
from utils.my_utils import set_seed
from covidx3_data_loader import get_dataloader_preprocess, get_covid_dataloader_valid
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, recall_score
from sklearn.metrics import roc_curve
from DatasetGenerator import DatasetGenerator
import albumentations as A
from albumentations.pytorch import ToTensorV2

import warnings
warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

model_configs = MODELS.keys()

def get_val_transform_seq(img_size):
    val_transform_seq = A.Compose([A.Resize(img_size, img_size),
                                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    ToTensorV2()
                                    ])
    return lambda img:val_transform_seq(image=np.array(img))['image']



def train(
    model_name: str,
    fold: int,
    debug: bool,
    epochs: int,
    num_workers: int=4,
    run: str=None,
    resume_weights: str="",
    resume_epoch: int=0,
):
    """
    Model training
    
    Args: 
        model_name : string name from the models configs listed in models.py file
        fold: evaluation fold number, 0-3
        debug: if True, runs the debugging on few images 
        epochs: number of epochs to train
        num_workers: number of workers available
        run : experiment run string to add for checkpoints name
        resume_weights: directory with weights (if avaialable)
        resume_epoch: number of epoch to continue training    
    """
    model_info = MODELS[model_name]
    run_str = "" if run is None or run == "" else f"_{run}"

    # creates directories for checkpoints, tensorboard and predicitons
    checkpoints_dir = f"{WEIGHTS_DIR}/resnet_epsilon_task6/{model_name}{run_str}_fold_{fold}"
    tensorboard_dir = f"{RESULTS_DIR}/tensorboard/resnet_epsilon_task6/{model_name}{run_str}_fold_{fold}"
    predictions_dir = f"{RESULTS_DIR}/oof/{model_name}{run_str}_fold_{fold}"
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)
    print("\n", model_name, "\n")

    logger = Logger(tensorboard_dir)
    retinanet = model_info.factory(**model_info.args)

    # load weights to continue training
    if resume_weights != "":
        print("load model from: ", resume_weights)
        retinanet = torch.load(resume_weights).cuda()
    else:
        retinanet = retinanet.cuda()
    retinanet = torch.nn.DataParallel(retinanet).cuda()
    m4_multilabel_fc_criterion = nn.BCEWithLogitsLoss(reduction = 'none')

    ########### COVID HEAD ############
    if debug:
        print("\nM1: COVID dataset: Setting up")
    m1_dir = "/scratch/mariamma/xraysetu/dataset/covidx3_upsampled"
    m1_train_dir = os.path.join(m1_dir, 'train')
    m1_valid_dir = os.path.join(m1_dir, 'test')
    if debug:
        print("M1: COVID dataset: Loading from training dir: ", m1_train_dir,
                " and from validation dir: ", m1_valid_dir)
    m1_batch_size = 4
    m1_num_workers = 12
    if debug:
        print("M1: COVID dataset: Dataloader for training")
    m1_train_loader, m1_train_len = get_dataloader_preprocess(m1_train_dir, batch_size = m1_batch_size, 
        image_size = model_info.img_size, num_workers = m1_num_workers)
    if debug:
        print("M1: COVID dataset: Dataloader for validation")
    m1_valid_loader, m1_valid_len = get_covid_dataloader_valid(m1_valid_dir, batch_size = m1_batch_size, 
        image_size = model_info.img_size, num_workers = m1_num_workers)

    loaders_scratch1 = {
    'm1_train': m1_train_loader,
    'm1_valid': m1_valid_loader,
    #'test': test_loader
    }
    if debug:
        print("m1 data loaded")

    ############# NIH HEAD ############
    nih_classes = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
    nih_pathDirData = "/scratch/mariamma/xraysetu/dataset/images"
    # m4_pathFileTrain = "../datasets/train_1_reordered_balanced_exp1_loss1.txt"
    nih_pathFileTrain = "/scratch/mariamma/xraysetu/dataset/mtl16_txtfiles/train_1346.txt"
    nih_pathFileVal = "/scratch/mariamma/xraysetu/dataset/mtl16_txtfiles/val_1346.txt"
    
    nih_trBatchSize = 5
    nih_valBatchSize = 4
    if debug:
        print("M4: NIH dataset: Loading from dir: ", nih_pathDirData,
                " using training txt: ", nih_pathFileTrain,
                " and validation txt: ", nih_pathFileVal)
    m4_train_data = DatasetGenerator(pathImageDirectory = nih_pathDirData,
                                        pathDatasetFile = nih_pathFileTrain,
                                        transform = transforms.Compose([
                    # transforms.Lambda(croptop),
                    # transforms.RandomApply(torch.nn.ModuleList([
                    # transforms.RandomHorizontalFlip(p=0.5),
                    # transforms.RandomRotation(10),
                    # transforms.RandomAffine(degrees=0,translate=(.1,.1)),
                    # transforms.ColorJitter(brightness=(.9,1.1)),
                    # transforms.RandomAffine(degrees=0,scale=(0.85, 1.15)),
                    # ]), p=0.5),
                    #transforms.Normalize(mean=[ 0.406], std=[0.225]),
                    transforms.ToTensor(),
                    transforms.Resize(size=(model_info.img_size,model_info.img_size)),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]))

    if debug:
        print("M4: NIH dataset: Initializing for validation")
    m4_valid_data = DatasetGenerator(pathImageDirectory = nih_pathDirData,
                                        pathDatasetFile = nih_pathFileVal,
                                        transform = get_val_transform_seq(model_info.img_size))
    
    if debug:
        print("M4: NIH dataset: Dataloader for training")
    nih_dataLoaderTrain = DataLoader(dataset = m4_train_data,
                                    batch_size = nih_trBatchSize,
                                    shuffle = True,
                                    num_workers = 12,
                                    pin_memory = True,
                                    drop_last=True)
    if debug:
        print("M4: NIH dataset: Dataloader for validation")
    nih_dataLoaderVal = DataLoader(dataset = m4_valid_data,
                                    batch_size = nih_valBatchSize,
                                    shuffle = True,
                                    num_workers = 12,
                                    pin_memory = True,
                                    drop_last=True)

    if debug:
        print("m4 data loaded")
  

    ############# RSNA HEAD ############
    # datasets for train and validation
    dataset_train = DetectionDataset(
        fold=fold,
        img_size=model_info.img_size,
        is_training=True,
        debug=debug,
        **model_info.dataset_args,
    )

    dataset_valid = DetectionDataset(
        fold=fold, img_size=model_info.img_size, is_training=False, debug=debug
    )

    # dataloaders for train and validation
    dataloader_train = DataLoader(
        dataset_train,
        num_workers=num_workers,
        batch_size=model_info.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=pytorch_retinanet.dataloader.collater2d,
    )

    dataloader_valid = DataLoader(
        dataset_valid,
        num_workers=num_workers,
        batch_size=4,
        shuffle=False,
        drop_last=True,
        collate_fn=pytorch_retinanet.dataloader.collater2d,
    )
    print("{} training images".format(len(dataset_train)))
    print("{} validation images".format(len(dataset_valid)))

    # set optimiser and scheduler
    retinanet.training = True
    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=4, verbose=True, factor=0.2
    )
    scheduler_by_epoch = False
    data_load_idx = ["covid", "rsna", "nih"] # COVID, NIH, RSNA
    r1_loss_prev = r2_loss_prev = r4_loss_prev = r3_loss_prev = r5_loss_prev = r6_loss_prev = 1 
    # train cycle
    for epoch_num in range(resume_epoch + 1, epochs):
        retinanet.train()
        if epoch_num < 1:
            retinanet.module.freeze_encoder()  # train FC layers with freezed encoder for the first epoch
        else:
            retinanet.module.unfreeze_encoder()
        retinanet.module.freeze_bn()
        # set losses
        epoch_loss, loss_cls_hist, loss_cls_global_hist, loss_reg_hist = [], [], [], []
        m1_iter_train = iter(loaders_scratch1['m1_train'])
        nih_iter_train = iter(nih_dataLoaderTrain)

        with torch.set_grad_enabled(True):
            data_iter = tqdm(enumerate(dataloader_train), total=len(dataloader_train))
            for iter_num, data in data_iter:
                optimizer.zero_grad()
                # nih_loss = r1_loss = classification_loss = regression_loss = 0
                for data_load in data_load_idx:

                    ###### M1 (COVID) Forward pass #####
                    if data_load == "covid":
                        if debug:
                            print("M1: Forward pass")
                        try:
                            m1_input, m1_target = next(m1_iter_train)
                        except StopIteration:
                            m1_iter_train = iter(loaders_scratch1['m1_train'])     
                        if debug:
                            print("M1-target : : ", m1_target)
                        m1_input, m1_target = m1_input.cuda().float(), m1_target.cuda().float()
                        inputs = {"data": m1_input}
                        _, m1_output, __ = retinanet(inputs,
                                                            return_loss=False,
                                                            return_boxes=False,
                                                            fc=True
                                                            )
                        if debug:                                                                
                            print("M1-output : : ", m1_output)                                                            
                        if m1_output.dim() == 1:
                            m1_output = m1_output.unsqueeze(0)

                        m1_target_list = []
                        m1_output_list = torch.cat([m1_output[:,6].unsqueeze(-1), m1_output[:,-1].unsqueeze(-1)],  dim=1)

                        for index, target_val in enumerate(m1_target):
                            if target_val == 0:     # Pneumonia
                                m1_target_list.append([1,0]) 
                            elif target_val == 1:    # Normal
                                m1_target_list.append([0,0]) 
                            else:                    # Covid
                                m1_target_list.append([0,1]) 
                            
                        # print("Output : {}, Target : {}".format(m1_output_list, m1_target_list))        
                        m1_loss = m4_multilabel_fc_criterion(m1_output_list, torch.tensor(m1_target_list).cuda().float())
                        r1_loss = m1_loss.mean() # get 2 classes
                        

                    ####### TRAIN NIH HEAD ######
                    elif data_load == "nih":
                        if debug:
                            print("M4: Forward pass")

                        ###### M4 (NIH) Forward pass #####
                        try:
                            m4_input, m4_target = next(nih_iter_train)
                        except StopIteration:
                            nih_iter_train = iter(nih_dataLoaderTrain)
                        m4_input, m4_target = m4_input.cuda().float(), m4_target.cuda().float()
                        inputs = {"data": m4_input}
                        _, m4_output, __ = retinanet(inputs,
                                                            return_loss=False, return_boxes=False,
                                                            fc=True)

                        # m1_m4_fc_loss = m4_multilabel_fc_criterion(outputs, targets)
                        m3_loss = m4_multilabel_fc_criterion(m4_output[:,3], m4_target[:,3])
                        m4_loss = m4_multilabel_fc_criterion(m4_output[:,2], m4_target[:,2])
                        m0_loss = m4_multilabel_fc_criterion(m4_output[:,0], m4_target[:,0])
                        m5_loss = m4_multilabel_fc_criterion(m4_output[:,5], m4_target[:,5])
                        r3_loss = m3_loss.mean()
                        r4_loss = m4_loss.mean()
                        r5_loss = m0_loss.mean()
                        r6_loss = m5_loss.mean()

                    ####### TRAIN RSNA HEAD ######
                    elif data_load == "rsna":
                        # model inputs
                        inputs = [
                            data["img"].cuda().float(),
                            data["annot"].cuda().float(),
                            data["category"].cuda(),
                        ]
                        # get losses
                        (classification_loss, regression_loss, global_classification_loss,) = retinanet(
                            inputs, return_loss=True, return_boxes=False
                        )
                        classification_loss = classification_loss.mean()
                        regression_loss = regression_loss.mean()
                        global_classification_loss = global_classification_loss.mean()
                        loss = classification_loss + regression_loss + global_classification_loss * 0.1
        #                 # uncomment for regress_only
                        r2_loss = classification_loss + regression_loss
        #                 # uncomment for classification0.1
        #                 loss = classification_loss * 0.1 + regression_loss
                # back prop
                rate1 = r1_loss/r1_loss_prev
                rate2 = r2_loss/r2_loss_prev
                rate4 = r4_loss/r4_loss_prev
                rate3 = r3_loss/r3_loss_prev
                rate5 = r5_loss/r5_loss_prev
                rate6 = r6_loss/r6_loss_prev
                
                # print("Rate1={}, Rate2={}, Rate3={}".format(rate1, rate2, rate4))
                loss_arr = [0]*6
                loss_arr[0] = r1_loss
                loss_arr[1] = r2_loss
                loss_arr[2] = r3_loss
                loss_arr[3] = r4_loss
                loss_arr[4] = r5_loss
                loss_arr[5] = r6_loss

                rate_arr = [0]*6  
                rate_arr[0] = rate1
                rate_arr[1] = rate2
                rate_arr[2] = rate3
                rate_arr[3] = rate4
                rate_arr[4] = rate5
                rate_arr[5] = rate6

                max_loss = 0
                max_rate = rate_arr[0]
                for i in range(len(rate_arr)):
                    if rate_arr[i] > max_rate:
                        max_rate = rate_arr[i]
                        max_loss = loss_arr[i]

                loss = count = 0          
                if r1_loss > (max_loss - 0.1):
                    loss += r1_loss
                    count += 1
                if r2_loss > (max_loss - 0.1):
                    loss += r2_loss
                    count += 1
                if r3_loss > (max_loss - 0.1):
                    loss += r3_loss 
                    count += 1    
                if r4_loss > (max_loss - 0.1):
                    loss += r4_loss 
                    count += 1
                if r5_loss > (max_loss - 0.1):
                    loss += r5_loss 
                    count += 1  
                if r6_loss > (max_loss - 0.1):
                    loss += r6_loss 
                    count += 1                        
                loss = loss/count
                loss.backward()

                r1_loss_prev = r1_loss
                r2_loss_prev = r2_loss
                r3_loss_prev = r3_loss
                r4_loss_prev = r4_loss
                r5_loss_prev = r5_loss
                r6_loss_prev = r6_loss

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.05)
                optimizer.step()
                # loss history
                loss_cls_hist.append(float(classification_loss))
                loss_cls_global_hist.append(float(global_classification_loss))
                loss_reg_hist.append(float(regression_loss))
                epoch_loss.append(float(loss))
                # print losses with tqdm interator
                data_iter.set_description(
                    f"{epoch_num} cls: {np.mean(loss_cls_hist):1.4f} cls g: {np.mean(loss_cls_global_hist):1.4f} Reg: {np.mean(loss_reg_hist):1.4f} Loss: {np.mean(epoch_loss):1.4f}"
                )
                del classification_loss
                del regression_loss

        # save model and log loss history
        torch.save(retinanet.module, f"{checkpoints_dir}/{model_name}_{epoch_num:03}.pt")
        logger.scalar_summary("loss_train", np.mean(epoch_loss), epoch_num)
        logger.scalar_summary("loss_train_classification", np.mean(loss_cls_hist), epoch_num)
        logger.scalar_summary(
            "loss_train_global_classification", np.mean(loss_cls_global_hist), epoch_num
        )
        logger.scalar_summary("loss_train_regression", np.mean(loss_reg_hist), epoch_num)
        logger.scalar_summary("loss_train_rsna", np.mean(loss_cls_hist)+np.mean(loss_reg_hist), epoch_num)
        logger.scalar_summary("loss_train_covid", r1_loss.detach().cpu().numpy(), epoch_num)
        logger.scalar_summary("loss_train_nih", r4_loss.detach().cpu().numpy(), epoch_num)

        # validation
        (
            loss_hist_valid,
            loss_cls_hist_valid,
            loss_cls_global_hist_valid,
            loss_reg_hist_valid, 
            m1_mean_loss,
            nih_mean_loss
        ) = validation(retinanet,dataloader_valid,epoch_num,predictions_dir, \
            m1_valid_loader, nih_dataLoaderVal, save_oof=False
        )

        # log validation loss history
        m1_mean_loss = m1_mean_loss.detach().cpu().numpy()
        nih_mean_loss = nih_mean_loss.detach().cpu().numpy()
        logger.scalar_summary("loss_valid", np.mean(loss_hist_valid) + m1_mean_loss + nih_mean_loss, epoch_num)
        logger.scalar_summary("loss_valid_classification", np.mean(loss_cls_hist_valid), epoch_num)
        logger.scalar_summary(
            "loss_valid_global_classification", np.mean(loss_cls_global_hist_valid), epoch_num,
        )
        logger.scalar_summary("loss_valid_regression", np.mean(loss_reg_hist_valid), epoch_num)
        logger.scalar_summary("RSNA Valid Loss", np.mean(loss_hist_valid), epoch_num)
        logger.scalar_summary("Covid Valid Loss", m1_mean_loss, epoch_num)
        logger.scalar_summary("NIH Valid Loss", nih_mean_loss, epoch_num)

        if epoch_num > 0:
            if epoch_num %3 == 0:
                covid_test_dataset(retinanet, epoch_num, logger)
        if scheduler_by_epoch:
            scheduler.step(epoch=epoch_num)
        else:
            scheduler.step(np.mean(loss_reg_hist_valid))
    retinanet.eval()
    torch.save(retinanet, f"{checkpoints_dir}/{model_name}_final.pt")
   

def validation(
    retinanet: nn.Module, dataloader_valid: nn.Module, epoch_num: int, predictions_dir: str, m1_valid_loader, nih_dataLoaderVal, save_oof=True, 
) -> tuple:
    """
    Validate model at the epoch end 
       
    Args: 
        retinanet: current model 
        dataloader_valid: dataloader for the validation fold
        epoch_num: current epoch
        save_oof: boolean flag, if calculate oof predictions and save them in pickle
        predictions_dir: directory fro saving predictions

    Outputs:
        loss_hist_valid: total validation loss, history 
        loss_cls_hist_valid, loss_cls_global_hist_valid: classification validation losses
        loss_reg_hist_valid: regression validation loss
    """
    m4_multilabel_fc_criterion = nn.BCEWithLogitsLoss(reduction = 'none')
    with torch.no_grad():
        retinanet.eval()
        m1_val_loss = 0
        for batch_no, (m1_input, m1_target) in enumerate(m1_valid_loader):
            m1_input, m1_target = m1_input.cuda().float(), m1_target.cuda()
            inputs = {"data": m1_input, "target": m1_target}
            _, m1_output, _ = retinanet(inputs, return_loss=False, return_boxes=False, fc=True)
            m1_target_list = []
            if m1_output.dim() == 1:
                m1_output = m1_output.unsqueeze(0)
            m1_output_list = torch.cat([m1_output[:,6].unsqueeze(-1), m1_output[:,-1].unsqueeze(-1)],  dim=1)
            for index, target_val in enumerate(m1_target):
                if target_val == 0:     # Pneumonia
                    m1_target_list.append([1,0]) 
                elif target_val == 1:    # Normal
                    m1_target_list.append([0,0]) 
                else:                    # Covid
                    m1_target_list.append([0,1])                           
            m1_loss = m4_multilabel_fc_criterion(m1_output_list, torch.tensor(m1_target_list).cuda().float())
            #r1_loss = m1_loss.mean()  
            r1_loss = m1_loss.sum()     
            m1_val_loss += r1_loss                                                                           
        m1_mean_loss = m1_val_loss/len(m1_valid_loader)

        nih_val_loss = 0
        for batch_no, (m4_input, m4_target) in enumerate(nih_dataLoaderVal):
            m4_input, m4_target = m4_input.cuda().float(), m4_target.cuda()
            inputs = {"data": m4_input, "target": m4_target}
            _, m4_output, _= retinanet(inputs, return_loss=False, return_boxes=False, fc=True)
            m3_loss = m4_multilabel_fc_criterion(m4_output[:,3], m4_target[:,3])
            r3_loss = m3_loss.sum()
            m4_loss = m4_multilabel_fc_criterion(m4_output[:,2], m4_target[:,2])
            r4_loss = m4_loss.sum()
            m0_loss = m4_multilabel_fc_criterion(m4_output[:,0], m4_target[:,0])
            r5_loss = m0_loss.sum()
            m5_loss = m4_multilabel_fc_criterion(m4_output[:,5], m4_target[:,5])
            r6_loss = m5_loss.sum()
            #r4_loss = m4_loss.sum()
            nih_val_loss += (r3_loss + r4_loss + r5_loss + r6_loss)/4
        nih_mean_loss = nih_val_loss/len(nih_dataLoaderVal)    

        loss_hist_valid, loss_cls_hist_valid, loss_cls_global_hist_valid, loss_reg_hist_valid = [],[],[],[]
        data_iter = tqdm(enumerate(dataloader_valid), total=len(dataloader_valid))
        if save_oof:
            oof = collections.defaultdict(list)
        for iter_num, data in data_iter:
            res = retinanet(
                [
                    data["img"].cuda().float(),
                    data["annot"].cuda().float(),
                    data["category"].cuda(),
                ],
                return_loss=True,
                return_boxes=True,
            )

            (
                classification_loss,
                regression_loss,
                global_classification_loss,
                nms_scores,
                global_class,
                transformed_anchors,
            ) = res
            if save_oof:
                # predictions
                oof["gt_boxes"].append(data["annot"].cpu().numpy().copy())
                oof["gt_category"].append(data["category"].cpu().numpy().copy())
                oof["boxes"].append(transformed_anchors.cpu().numpy().copy())
                oof["scores"].append(nms_scores.cpu().numpy().copy())
                oof["category"].append(global_class.cpu().numpy().copy())

            # get losses
            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()
            global_classification_loss = global_classification_loss.mean()
            
            #loss = classification_loss + regression_loss + global_classification_loss * 0.1
            loss = classification_loss + regression_loss 
            # loss history
            loss_hist_valid.append(float(loss))
            loss_cls_hist_valid.append(float(classification_loss))
            loss_cls_global_hist_valid.append(float(global_classification_loss))
            loss_reg_hist_valid.append(float(regression_loss))
            data_iter.set_description(
                f"{epoch_num} cls: {np.mean(loss_cls_hist_valid):1.4f} cls g: {np.mean(loss_cls_global_hist_valid):1.4f} Reg: {np.mean(loss_reg_hist_valid):1.4f} Loss {np.mean(loss_hist_valid):1.4f}"
            )
            del classification_loss
            del regression_loss

        if save_oof:  # save predictions
            pickle.dump(oof, open(f"{predictions_dir}/{epoch_num:03}.pkl", "wb"))

    return loss_hist_valid, loss_cls_hist_valid, loss_cls_global_hist_valid, loss_reg_hist_valid, m1_mean_loss, nih_mean_loss


def get_prediction(retinanet: nn.Module, native_test_dir):
    test_batch_size = 1
    standard_normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])
    transform_seq = transforms.Compose([transforms.Resize(size=(512,512)),
                                            transforms.ToTensor(),
                                            standard_normalization])
    native_test_data = datasets.ImageFolder(native_test_dir, transform = transform_seq)
    native_test_loader = torch.utils.data.DataLoader(native_test_data, batch_size = test_batch_size, num_workers = 2, drop_last = True)            
    all_preds = []
    all_true = []
    native_test_iter = iter(native_test_loader)

    data_iter = tqdm(enumerate(native_test_loader), total=len(native_test_loader))
    #data_iter = enumerate(native_test_loader)
    for idx,d in data_iter:
        native_test_input, native_test_target = next(native_test_iter)
        native_test_data, native_test_target = native_test_input.cuda().float(), native_test_target.cuda()
        inputs = {"data": native_test_data}
        _, __, fc_output_niramai_test = retinanet(inputs, return_loss=False, fc=True, return_boxes=False)
        outputs = fc_output_niramai_test
        targets = native_test_target
        if idx == 0:
            all_preds = outputs.cpu().data.numpy()[14]
            all_true = targets.squeeze().cpu().numpy()
        else:
            all_preds = np.append(all_preds, outputs.cpu().data.numpy()[14])
            all_true = np.append(all_true, targets.squeeze().cpu().numpy())
    return all_true, all_preds

def get_optim_threshols(y_true, probs):
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    accuracy_scores = []
    for thresh in thresholds:
        accuracy_scores.append(accuracy_score(y_true, [m > thresh for m in probs]))

    accuracies = np.array(accuracy_scores)
    max_accuracy = accuracies.max() 
    max_accuracy_threshold =  thresholds[accuracies.argmax()]
    prediction_original = (probs > max_accuracy_threshold).astype('int32')
    recall = recall_score(y_true, prediction_original)
    return max_accuracy, max_accuracy_threshold, recall, prediction_original

def get_flips(prediction_original, prediction_whatsapp):
    """Function for generating flips"""
    flips = 0
    for i in range(prediction_original.shape[0]):
        if prediction_original[i] != prediction_whatsapp[i]:
            flips += 1
    return flips

def get_whatsapp_perf(y_true, probs, opt_thres):
    prediction = (probs > opt_thres).astype('int32')
    recall = recall_score(y_true, prediction)
    accuracy = accuracy_score(y_true, prediction)
    return accuracy, recall, prediction

def covid_test_dataset(retinanet: nn.Module, epoch_num: int, logger):
    native_test_dir = "/scratch/mariamma/xraysetu/dataset/covid_binary_test/"
    native_true, native_pred = get_prediction(retinanet, native_test_dir) 
    native_auroc = roc_auc_score(native_true, native_pred)
    logger.scalar_summary("Native Covid AUROC", native_auroc, epoch_num)
    native_max_accuracy, native_max_accuracy_threshold,  native_recall, native_prediction = \
            get_optim_threshols(native_true, native_pred)
    logger.scalar_summary("Native Accuracy", native_max_accuracy, epoch_num)
    logger.scalar_summary("Native Sensitivity", native_recall, epoch_num)
    print("Native Images :: AUCROC={}, Accuracy={}, Sensitivity={}".format(native_auroc, native_max_accuracy, native_recall))
    
    whatsapp_test_dir = "/scratch/mariamma/xraysetu/dataset/covid_binary_test_whatsapp/"
    whatsapp_true, whatsapp_pred = get_prediction(retinanet, whatsapp_test_dir)
    whatsapp_auroc = roc_auc_score(whatsapp_true, whatsapp_pred)
    logger.scalar_summary("Whatsapp Covid AUROC", whatsapp_auroc, epoch_num)
    
    whatsapp_max_accuracy, whatsapp_max_accuracy_threshold,  whatsapp_recall, whatsapp_prediction = \
            get_optim_threshols(whatsapp_true, whatsapp_pred)
    flips = get_flips(native_prediction, whatsapp_prediction)
    logger.scalar_summary("Whatsapp Accuracy Max", whatsapp_max_accuracy, epoch_num)
    logger.scalar_summary("Whatsapp Sensitivity Max", whatsapp_recall, epoch_num)
    logger.scalar_summary("Flips Max", flips, epoch_num)
    print("Whatsapp Images MAX:: AUCROC={}, Accuracy={}, Sensitivity={}, Flips={}".format(whatsapp_auroc, whatsapp_max_accuracy, whatsapp_recall, flips))

    whatsapp_accuracy_th, whatsapp_recall_th, whatsapp_prediction_th = \
        get_whatsapp_perf(whatsapp_true, whatsapp_pred, native_max_accuracy_threshold)
    flips_th = get_flips(native_prediction, whatsapp_prediction_th)
    logger.scalar_summary("Whatsapp Accuracy Th", whatsapp_accuracy_th, epoch_num)
    logger.scalar_summary("Whatsapp Sensitivity Th", whatsapp_recall_th, epoch_num)
    logger.scalar_summary("Flips Th", flips_th, epoch_num)
    print("Whatsapp Images TH:: AUCROC={}, Accuracy={}, Sensitivity={}, Flips={}".format(whatsapp_auroc, whatsapp_accuracy_th, whatsapp_recall_th, flips_th))
    return 

def test_model(model_name: str, fold: int, debug: bool, checkpoint: str, pics_dir: str):
    """
    Loads model weights from the checkpoint, plots ground truth and predictions
    
    Args: 
        model_name : string name from the models configs listed in models.py file
        fold       : evaluation fold number, 0-3
        debug      : if True, runs debugging on few images 
        checkpoint : directory with weights (if avaialable) 
        pics_dir   : directory for saving prediction images 
       
    """
    model_info = MODELS[model_name]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # load model
    model = torch.load(checkpoint)
    model = model.to(device)
    model.eval()
    # load data
    dataset_valid = DetectionDataset(
        fold=fold, img_size=model_info.img_size, is_training=False, debug=debug
    )
    dataloader_valid = DataLoader(
        dataset_valid,
        num_workers=1,
        batch_size=1,
        shuffle=False,
        collate_fn=pytorch_retinanet.dataloader.collater2d,
    )

    data_iter = tqdm(enumerate(dataloader_valid), total=len(dataloader_valid))
    for iter_num, data in data_iter:
        (
            classification_loss,
            regression_loss,
            global_classification_loss,
            nms_scores,
            nms_class,
            transformed_anchors,
        ) = model(
            [
                data["img"].to(device).float(),
                data["annot"].to(device).float(),
                data["category"].cuda(),
            ],
            return_loss=True,
            return_boxes=True,
        )

        nms_scores = nms_scores.cpu().detach().numpy()
        nms_class = nms_class.cpu().detach().numpy()
        transformed_anchors = transformed_anchors.cpu().detach().numpy()
        print(
            "nms_scores {}, transformed_anchors.shape {}".format(
                nms_scores, transformed_anchors.shape
            )
        )
        print(
            "cls loss:",
            float(classification_loss),
            "global cls loss:",
            global_classification_loss,
            " reg loss:",
            float(regression_loss),
        )
        print(
            "category:",
            data["category"].numpy()[0],
            np.exp(nms_class[0]),
            dataset_valid.categories[data["category"][0]],
        )

        # plot data and ground truth
        plt.figure(iter_num, figsize=(6, 6))
        plt.cla()
        plt.imshow(data["img"][0, 0].cpu().detach().numpy(), cmap=plt.cm.gist_gray)
        plt.axis("off")
        gt = data["annot"].cpu().detach().numpy()[0]
        for i in range(gt.shape[0]):
            if np.all(np.isfinite(gt[i])):
                p0 = gt[i, 0:2]
                p1 = gt[i, 2:4]
                plt.gca().add_patch(
                    plt.Rectangle(
                        p0,
                        width=(p1 - p0)[0],
                        height=(p1 - p0)[1],
                        fill=False,
                        edgecolor="b",
                        linewidth=2,
                    )
                )
        # add predicted boxes to the plot
        for i in range(len(nms_scores)):
            nms_score = nms_scores[i]
            if nms_score < 0.1:
                break
            p0 = transformed_anchors[i, 0:2]
            p1 = transformed_anchors[i, 2:4]
            color = "r"
            if nms_score < 0.3:
                color = "y"
            if nms_score < 0.25:
                color = "g"
            plt.gca().add_patch(
                plt.Rectangle(
                    p0,
                    width=(p1 - p0)[0],
                    height=(p1 - p0)[1],
                    fill=False,
                    edgecolor=color,
                    linewidth=2,
                )
            )
            plt.gca().text(p0[0], p0[1], f"{nms_score:.3f}", color=color)
        plt.show()

        os.makedirs(pics_dir, exist_ok=True)
        plt.savefig(
            f"{pics_dir}/predict_{iter_num}.eps", dpi=300, bbox_inches="tight", pad_inches=0,
        )
        plt.savefig(
            f"{pics_dir}/predict_{iter_num}.png", dpi=300, bbox_inches="tight", pad_inches=0,
        )
        plt.close()
        print(nms_scores)


def generate_predictions(
    model_name: str, fold: int, debug: bool, weights_dir: str, from_epoch: int=0, to_epoch: int=10, save_oof: bool = True, run: str=None,
):
    """
    Loads model weights the epoch checkpoints, 
    calculates oof predictions for and saves them to pickle
    
    Args: 
        model_name : string name from the models configs listed in models.py file
        fold       : evaluation fold number, 0-3
        debug      : if True, runs debugging on few images 
        weights_dir: directory qith model weigths
        from_epoch : the first epoch for predicitions generation 
        to_epoch   : the last epoch for predicitions generation 
        save_oof   : boolean flag weathe rto save precitions
        run        : string name to be added in the experinet name
    """
    predictions_dir = f"{RESULTS_DIR}/test1/{model_name}_fold_{fold}"
    os.makedirs(predictions_dir, exist_ok=True)

    model_info = MODELS[model_name]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    for epoch_num in range(from_epoch, to_epoch):
        prediction_fn = f"{predictions_dir}/{epoch_num:03}.pkl"
        if os.path.exists(prediction_fn):
            continue
        print("epoch", epoch_num)
        # load model checkpoint
        checkpoint = (
            f"{weights_dir}/{model_name}_fold_{fold}/{model_name}_{epoch_num:03}.pt"
        )
        print("load", checkpoint)
        try:
            model = torch.load(checkpoint)
        except FileNotFoundError:
            break
        model = model.to(device)
        model.eval()
        # load data
        dataset_valid = DatasetValid(
            is_training=False,
            meta_file= "stage_1_test_meta.csv", 
            debug=debug, 
            img_size=512,
            )
        dataloader_valid = DataLoader(
            dataset_valid,
            num_workers=2,
            batch_size=4,
            shuffle=False,
            collate_fn=pytorch_retinanet.dataloader.collater2d,
        )

        oof = collections.defaultdict(list)
        for iter_num, data in tqdm(enumerate(dataset_valid), total=len(dataloader_valid)):
            data = pytorch_retinanet.dataloader.collater2d([data])
            img = data["img"].to(device).float()
            nms_scores, global_classification, transformed_anchors = model(
                img, return_loss=False, return_boxes=True
            )
            # model outputs to numpy
            nms_scores = nms_scores.cpu().detach().numpy()
            global_classification = global_classification.cpu().detach().numpy()
            transformed_anchors = transformed_anchors.cpu().detach().numpy()
            # out-of-fold predictions
            oof["gt_boxes"].append(data["annot"].cpu().detach().numpy())
            oof["gt_category"].append(data["category"].cpu().detach().numpy())
            oof["boxes"].append(transformed_anchors)
            oof["scores"].append(nms_scores)
            oof["category"].append(global_classification)
        # save epoch predictions
        if save_oof:  
            pickle.dump(oof, open(f"{predictions_dir}/{epoch_num:03}.pkl", "wb"))
    
        
def p1p2_to_xywh(p1p2: np.ndarray) -> np.ndarray:
    """
    Helper function
    converts box coordinates to 
    x0, y0, width, height format
    """
    xywh = np.zeros((p1p2.shape[0], 4))
    xywh[:, :2] = p1p2[:, :2]
    xywh[:, 2:4] = p1p2[:, 2:4] - p1p2[:, :2]
    return xywh


def check_metric(
    model_name: str,
    run: str,
    fold: int,
    oof_dir: str,
    start_epoch: int,
    end_epoch: int,
    save_metrics: bool=False,
):
    """
    Loads epoch predicitons and
    calculates the metric for a set of thresholds

    Args: 
        model_name  : string name from the models configs listed in models.py file
        run         : experiment run string to add for checkpoints name
        fold        : evaluation fold number, 0-3
        oof_dir     : directory with out-of-fold predictions
        start_epoch, end_epoch: the first ad last epochs for metric calculation
        save_metrics: boolean flag weather to save metrics
        
    Output:
        thresholds: list of thresholds for mean average precision calculation
        epochs    : range of epochs
        all_scores: all metrics values for all thresholds and epochs
    """
    run_str = "" if run is None or run == "" else f"_{run}"
    predictions_dir = f"{oof_dir}/{model_name}{run_str}_fold_{fold}"
    thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.28, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0]
    all_scores = []

    for epoch_num in range(start_epoch, end_epoch):
        fn = f"{predictions_dir}/{epoch_num:03}.pkl"
        try:
            oof = pickle.load(open(fn, "rb"))
        except FileNotFoundError:
            continue

        print("epoch ", epoch_num)
        epoch_scores = []
        nb_images = len(oof["scores"])
        # check range of thresholds
        for threshold in thresholds:
            threshold_scores = []
            for img_id in range(nb_images):
                gt_boxes = oof["gt_boxes"][img_id][0].copy()
                boxes = oof["boxes"][img_id].copy()
                scores = oof["scores"][img_id].copy()
                category = oof["category"][img_id]
                category = np.exp(category[0, 2])

                if len(scores):
                    scores[scores < scores[0] * 0.5] = 0.0
                mask = scores * 5 > threshold

                if gt_boxes[0, 4] == -1.0:
                    if np.any(mask):
                        threshold_scores.append(0.0)
                else:
                    if len(scores[mask]) == 0:
                        score = 0.0
                    else:
                        score = metric.map_iou(
                            boxes_true=p1p2_to_xywh(gt_boxes),
                            boxes_pred=p1p2_to_xywh(boxes[mask]),
                            scores=scores[mask],
                        )
                    # print(score)
                    threshold_scores.append(score)

            print("threshold {}, score {}".format(threshold, np.mean(threshold_scores)))
            epoch_scores.append(np.mean(threshold_scores))
        all_scores.append(epoch_scores)

    best_score = np.max(all_scores)
    epochs = np.arange(start_epoch, end_epoch)
    print("best score: ", best_score)
    plt.imshow(np.array(all_scores))
    plt.show()
    if save_metrics:
        scores_dir = f"{RESULTS_DIR}/scores/{model_name}{run_str}_fold_{fold}"
        os.makedirs(scores_dir, exist_ok=True)
        print(
            "all scores.shape: {}, thresholds {}, epochs {}".format(
                np.array(all_scores).shape, thresholds, epochs
            )
        )
        metric_scores = collections.defaultdict(list)
        metric_scores["scores"] = np.array(all_scores)
        metric_scores["tresholds"] = thresholds
        metric_scores["epochs"] = epochs
        pickle.dump(metric_scores, open(f"{scores_dir}/scores.pkl", "wb"))
    return np.array(all_scores), thresholds, epochs


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--action", type=str, default="train", help="Choose action: train, test_model, check_metric, generate_predictions")
    arg("--model", type=str, default="resnet50_512", help="String model name from models dictionary")
    arg("--run", type=str, default="", help="Experiment id string to be added for saving model")
    arg("--seed", type=int, default=1234, help="Random seed")
    arg("--fold", type=int, default=0, help="Validation fold")
    arg("--weights_dir", type=str, default="../../checkpoints", help="Directory for loading model weights")
    arg("--epoch", type=int, default=12, help="Current epoch")
    arg("--from-epoch", type=int, default=2, help="Resume training from epoch")
    arg("--num-epochs", type=int, default=70, help="Number of epochs to run")
    arg("--batch-size", type=int, default=4, help="Batch size for training")
    arg("--learning-rate", type=float, default=1e-5, help="Initial learning rate")
    arg("--debug", type=bool, default=False, help="If the debugging mode")
    args = parser.parse_args()

    set_seed(args.seed)
  
    if args.action == "train":
        train(
            model_name=args.model,
            run=args.run,
            fold=args.fold,
            debug=args.debug,
            epochs=args.num_epochs,
        )

    if args.action == "test_model":
        run_str = "" if args.run is None or args.run == "" else f"_{args.run}"
        weights = (
            f"{WEIGHTS_DIR}/{args.model}{run_str}_fold_{args.fold}/{args.model}_{args.epoch:03}.pt"
        )
        test_model(
            model_name=args.model,
            fold=args.fold,
            debug=args.debug,
            checkpoint=weights,
            pics_dir=f"{RESULTS_DIR}/pics",
        )

    if args.action == "check_metric":
        all_scores, thresholds, epochs = check_metric(
            model_name=args.model,
            run=args.run,
            fold=args.fold,
            oof_dir=f"{RESULTS_DIR}/oof",
            start_epoch=1,
            end_epoch=15,
            save_metrics=True,
        )

    if args.action == "generate_predictions":
        generate_predictions(
            model_name=args.model,
            run=args.run,
            fold=args.fold,
            weights_dir=WEIGHTS_DIR,
            debug=args.debug,
            from_epoch=args.from_epoch,
            to_epoch=args.num_epochs + 1,
            save_oof=True,
        )


if __name__ == "__main__":
    main()

