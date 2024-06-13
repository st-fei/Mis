import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import h5py
import logging
import random
import time
import transformers
import pandas as pd
from tqdm import tqdm
from args import get_args
from data.loader.DialogRetrieveLoader import TikTalkDataset, TikTalkDataloader
from transformers import BertTokenizer, BartConfig
from pipeline.trainer import Trainer
from model.mine_RVMIS import Case
from tools import create_logger, set_seed, get_labels, txt_writer, save_loss_curve, evaluate_on_tiktalk



def main():

    global logger
    # args
    args = get_args()
    # experiment name
    experiment_name = args.experiment_name
    # lossfig path
    args.lossfig_path = os.path.join(args.lossfig_path, experiment_name)
    if os.path.exists(args.lossfig_path):
        os.mkdir(args.lossfig_path)
    
    # 根据当前实验名称设置表格记录地址
    args.xlsx_dir = os.path.join(args.xlsx_dir, experiment_name)
    if args.do_train and (not os.path.exists(args.xlsx_dir)):
        os.mkdir(args.xlsx_dir)
    
    
    # 根据模式设置log地址
    if args.do_train:
        log_path = args.train_log_path
    else:
        log_path = args.eval_log_path
    # 如果不存在log地址则创建
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    # 获取logger
    logger = create_logger(os.path.join(log_path, experiment_name))
    # 设置device
    if args.gpu_ids == 0:
        args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif args.gpu_ids == 1:
        args.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") 
    args.n_gpu = torch.cuda.device_count()
    # 获取输出地址
    output_dir = args.output_dir
    # 若不存在则创建输出地址
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    # 设置随机数种子
    set_seed(args.seed, args.n_gpu)
    logger.warning("Seed: %s, Device: %s, n_gpu: %s", args.seed, args.device, args.n_gpu)
    
    # 构建tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    logger.info(f"Initialize tokenizer from {args.model_name_or_path} ...")

    # 加载视频特征
    
    logger.info("Start loading video feature ...")
    start = time.time()
    frame_feat = {}
    with h5py.File(args.feat_path, 'r') as f:
        video_ids = f['video_ids']
        frame_feats = f['frame_feats']
        for (ids, feats) in zip(video_ids, frame_feats):
            ids = ids.decode('utf-8')
            frame_feat[ids] = feats # [32, 17, 512]
    cost_time = time.time() - start
    minutes = int(cost_time // 60)
    seconds = int(cost_time % 60)
    logger.info(f"Loading video feature costs {minutes} min {seconds} sec ...")
    


    # 加载数据
    logger.info("Start loading data ...")
    start = time.time()
    train_loader, dev_loader, test_loader = TikTalkDataloader(args=args, tokenizer=tokenizer, frame_feat=frame_feat)
    logger.info(f"Loading data cost {time.time() - start} ...")

    # config
    config = BartConfig.from_pretrained(args.model_name_or_path)
    
    # model
    model = Case(config=config, args=args)

    # frozen parameters
    frozen_list = ['clip_encoder', 'clip_processor']
    for n, p in model.named_parameters():
        if n.split('.')[0] in frozen_list:
            p.requires_grad = False
   
    logger.info(f'Number of trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    # device
    logger.info(f'current device: {args.device}')

    # Trainer
    trainer = Trainer(
        args=args, 
        model=model, 
        tokenizer=tokenizer,
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=test_loader,
        device=args.device
    )

    if args.do_train:
        trainer.train()

    elif args.do_eval:
        # 加载训练模型权重
        checkpoint_dir = args.eval_model_dir
        weight_file = os.path.join(checkpoint_dir, "pytorch_model.bin")
        logger.info(f"Start loading weight file {weight_file} ...")
        model.load_state_dict(torch.load(weight_file))
        # 获取预训练模型信息
        epoch = checkpoint_dir.split("-")[-2]
        step = checkpoint_dir.split("-")[-1]
        # 区分评估验证集or测试集
        mode = args.eval_mode
        result = trainer.eval(mode=mode, model=model, epoch=epoch, step=step, checkpoint_dir=checkpoint_dir)
        
    
    
    
if __name__ == "__main__":
    main()