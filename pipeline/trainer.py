import os
import time
import torch
import json
import logging
import transformers
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
from tools import tsv_writer, get_labels, evaluate_on_tiktalk

logger = logging.getLogger()

class Trainer():
    def __init__(self, args, model, tokenizer, train_loader, dev_loader, test_loader, device):

        # device
        self.device = device

        # args & model & tokenizer
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        # train param
        self.max_steps = args.max_steps
        self.epoch = args.epoch
        self.train_batch_size = args.train_batch_size
        self.test_batch_size = args.test_batch_size
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.weight_decay = args.weight_decay
        self.warm_up = args.warmup
        self.max_grad_norm = args.max_grad_norm
        self.lr = args.lr
        self.eps = args.eps
        self.ddp = args.ddp
        self.n_gpu = args.n_gpu

        if self.max_steps > 0:
            self.num_training_steps = self.max_steps
            self.epoch = self.num_training_steps // (len(train_loader) // self.gradient_accumulation_steps) + 1
        else:
            self.num_training_steps = self.epoch * (len(train_loader) // self.gradient_accumulation_steps)

        # train record
        self.plt_steps = args.plt_steps
        self.logging_steps = args.logging_steps
        self.save_steps = args.save_steps
        self.evaluate_during_training = args.evaluate_during_training
        self.evaluate_testset_during_training = args.evaluate_testset_during_training


        # dev record
        self.best_epoch = 0
        self.record_name = self.args.experiment_name
        self.columns = ['Mode', 'Epoch', 'Step', 'Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'ROUGE_L', 'Nist_1', 'Nist_2', 'Nist_3', 'Nist_4', 'Dist1', 'Dist2', 'Dist3']
        self.dev_df = pd.DataFrame(columns=self.columns)
        self.test_df = pd.DataFrame(columns=self.columns)

        # data loader
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        
        # optimizer & scheduler
        self.optimizer, self.scheduler = self.get_optimizer()
        
    # optimizer & scheduler
    def get_optimizer(self):

        # 定义了不应用权重衰减的模型参数
        no_decay = ['bias', 'LayerNorm.weight'] 
        # 创建一个包含两个元素的列表，每个元素为字典
        # 第一个字典元素用于处理应用权重衰减的参数
        # 第二个字典元素用于处理不应用权重衰减的参数
        grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not \
                any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if \
                any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.0}
        ]

        optimizer = transformers.AdamW(params=grouped_parameters, lr=self.lr, eps=self.eps)
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=self.warm_up * self.num_training_steps, 
            num_training_steps=self.num_training_steps
        )
        logging.info(f'Set Scheduler: count {self.num_training_steps} iterations ')

        return optimizer, scheduler


    
   # 保存checkpoint
    def save_checkpoint(self, args, model, epoch, global_step):

        if not os.path.exists(os.path.join(args.output_dir, args.experiment_name)):
            os.mkdir(os.path.join(args.output_dir, args.experiment_name))

        # 保存当前args到experiment目录
        args_dict = vars(args)
        args_dict['device'] = str(args_dict['device'])
        with open(os.path.join(args.output_dir, args.experiment_name, 'argparse.json'), 'w') as f:
            json.dump(args_dict, f, ensure_ascii=False, indent=4)

        # 保存checkpoint的目录
        checkpoint_dir = os.path.join(args.output_dir, args.experiment_name, 'checkpoint-{}-{}'.format(epoch, global_step))
        # 创建目录
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)

        # 根据模型是否有.module属性来确定要保存的模型。在多GPU训练中，模型可能被包装在nn.DataParallel中，因此需要检查是否有.module属性
        model_to_save = model.module if hasattr(model, 'module') else model 
        model_to_save.save_pretrained(checkpoint_dir)
        logger.info("Save checkpoint to {}".format(checkpoint_dir))
        return checkpoint_dir # 返回保存权重的目录

    def add_gen_cfg(self, checkpoint_dir, mode):
        
        # 初始化列表cc
        cc = ['pred'] 
        
        # 根据模型的一些预测参数来构建字符串列表cc
        cc.append('mode={}'.format(mode))
        cc.append('seed={}'.format(self.args.seed))
        cc.append('beam_size={}'.format(self.args.num_beams))
        cc.append('do_sample={}'.format(self.args.do_sample))
        cc.append('top_k={}'.format(self.args.top_k))
        cc.append('top_p={}'.format(self.args.top_p))
        cc.append('temperature={}'.format(self.args.temperature))
        cc.append('length_penalty={}'.format(self.args.length_penalty))
        cc.append('repetition_penalty={}'.format(self.args.repetition_penalty))
        
        # 返回生成文件的路径
        return os.path.join(checkpoint_dir, '{}.tsv'.format('-'.join(cc)))
        


    # 计算每个样本的预测是否与真实标签相匹配
    def compute_score_with_logits(self, logits, labels):
        ignore_index = -100
        _, logits = logits.max(dim=-1)
        non_pad_mask = labels.ne(ignore_index)
        n_correct = logits.eq(labels).masked_select(non_pad_mask).sum().item()
        n_word = non_pad_mask.sum().item()
        acc = n_correct / n_word
        return acc


    # Train stage
    def train(self):
        
        # 移入GPU
        self.model.to(self.device)
        if self.ddp:
            self.model = nn.DataParallel(self.model, device_ids=list(range(torch.cuda.device_count())))

        logger.info('------- Running training -------')
        logger.info(f'Number of trainable params: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')
        logger.info(f'Number of training samples: {len(self.train_loader.dataset)}')
        logger.info(f'Training Epoch: {self.epoch}')
        logger.info(f'Training Steps: {self.num_training_steps}')

        # 定义全局变量
        loss_list = []
        step_list = []

        global_step, global_loss, global_acc = 0, 0.0, 0.0
        
        self.model.zero_grad()


        # 循环Epoch
        for i in range(self.epoch):
            logging.info(f'Start training epoch {i + 1} ...')
            self.model.train()
            for step, batch in enumerate(tqdm(self.train_loader)):
                self.model.train()
                self.model.zero_grad() 
                # 此时输出的logits和labels已经对齐，不需要额外shift操作
                loss, logits, labels = self.model(
                    desc_clip_input_ids=batch["desc_clip_input_ids"],
                    his_clip_input_ids=batch["his_clip_input_ids"],
                    his_input_ids=batch["his_input_ids"],
                    his_attn_mask=batch["his_attn_mask"],
                    resp_input_ids=batch["resp_input_ids"],
                    resp_attn_mask=batch["resp_attn_mask"],
                    # token_type_ids=batch["token_type_ids"],
                    labels=batch["labels"],
                    glo_feat=batch["glo_feat"],
                    loc_feat=batch["loc_feat"],
                    text_history=batch["text_history"],
                    text_response=batch["text_response"]
                )
                if self.ddp and self.n_gpu > 1:
                    loss = loss.mean()

                step_acc = self.compute_score_with_logits(logits=logits, labels=labels)
                global_acc += (step_acc / self.gradient_accumulation_steps)

                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                global_loss += loss.item()

                # Need Optimizer
                if (step + 1) % self.gradient_accumulation_steps == 0:

                    global_step += 1
                    if global_step % self.plt_steps:
                        loss_list.append(loss.item())
                        step_list.append(global_step)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.model.zero_grad()
                
                    if (step + 1) % self.logging_steps == 0:
                        logger.info(
                            """
                            epoch: {} | step: {} | lr: {:.6f}
                            step_loss: {:.6f} | avg_loss: {:.6f}
                            step_acc: {:.4f}% | avg_acc: {:.4f}%
                            """.format(
                                i + 1, global_step, self.optimizer.param_groups[0]["lr"],
                                loss.item(), global_loss / global_step,
                                step_acc * 100, (global_acc / global_step) * 100
                            )
                        )
                    
                    # Save & Evaluate
                    if (self.save_steps > 0 and global_step % self.save_steps == 0) or \
                        global_step == self.num_training_steps:
                        
                        checkpoint_dir = self.save_checkpoint(args=self.args, model=self.model, epoch=i+1, global_step=global_step)
                
                        if self.evaluate_during_training:
                            logger.info(f"在验证集上执行评估, 当前Epoch: {i+1}, 当前Step: {global_step} ...")
                            result = self.eval(
                                mode="dev",
                                model=self.model,
                                epoch=i+1,
                                step=global_step,
                                checkpoint_dir=checkpoint_dir,
                            )

                            # 将result写入dev_dataframe
                            dev_data = {}
                            for col_name in self.columns:
                                dev_data[col_name] = result[col_name]
                            self.dev_df.loc[len(self.dev_df)] = dev_data
                            # 写入{experiment_name} + _dev.xlsx
                            with pd.ExcelWriter(os.path.join(self.args.xlsx_dir, self.record_name + '_dev.xlsx'), engine='xlsxwriter') as writer:
                                self.dev_df.to_excel(writer, float_format='%.6f',index=False)

                        
                        
                        if self.evaluate_testset_during_training:
                            logger.info(f"在验证集上执行评估, 当前Epoch: {i+1}, 当前Step: {global_step} ...")
                            self.eval(
                                mode="test",
                                model=self.model, 
                                epoch=i+1,
                                step=global_step,
                                checkpoint_dir=checkpoint_dir,
                            )

    # 调用model将test_loader中样本生成至output_file中
    def Generate(self, model, test_loader, output_file):

        # model to gpu
        model.to(self.device)

        # token id
        cls_token_id = self.tokenizer.cls_token_id
        sep_token_id = self.tokenizer.sep_token_id
        pad_token_id = self.tokenizer.pad_token_id

        model.eval()

        def gen_row():
            start = time.time()
            with torch.no_grad():
                for step, batch in tqdm(enumerate(test_loader)):
                    
                    desc_clip_input_ids = batch["desc_clip_input_ids"]
                    his_clip_input_ids = batch["his_clip_input_ids"]
                    his_input_ids = batch["his_input_ids"]
                    his_attn_mask = batch["his_attn_mask"]
                    #token_type_ids = batch["token_type_ids"]
                    glo_feat = batch["glo_feat"]
                    loc_feat = batch["loc_feat"]
                    
                    
                    # 准备输入数据
                    inputs = {
                        # data
                        "desc_clip_input_ids":desc_clip_input_ids,
                        "his_clip_input_ids":his_clip_input_ids,
                        "his_input_ids":his_input_ids,
                        "his_attn_mask":his_attn_mask,
                        # "token_type_ids": token_type_ids,
                        "glo_feat":glo_feat,
                        "loc_feat":loc_feat,
                        # generation config
                        "bos_token_id":cls_token_id,
                        "pad_token_id":pad_token_id,
                        "eos_token_ids":[pad_token_id, sep_token_id],
                        "max_gen_len":self.args.max_gen_len,
                        "do_sample":self.args.do_sample,
                        "num_beams":self.args.num_beams,
                        "temperature":self.args.temperature,
                        "top_k":self.args.top_k,
                        "top_p":self.args.top_p,
                        "length_penalty":self.args.length_penalty,
                        "repetition_penalty":self.args.repetition_penalty
                    }

                    if hasattr(model, "module"):
                        outputs = model.module.generate(**inputs)
                    else:
                        outputs = model.generate(**inputs)

                    
                    # all_caps为生成的文本描述
                    # all_confs包含了与每个生成描述相关的置信度，通过对模型输出进行指数化来计算
                    # [batch_size, 1, 40]
                    all_caps = outputs[0]  
                    
                    all_confs = torch.exp(outputs[1]) # 置信度
                    for caps, confs in zip(all_caps, all_confs):
                        res = []
                        for cap, conf in zip(caps, confs):
                            cap = self.tokenizer.decode(cap.tolist(), skip_special_tokens=True) # 将张量cap转化为文本字符串
                            cap = cap.split('.')[0] # 截取第一个句子
                            res.append({'response': cap, 'conf': conf.item()}) # 在结果中加入生成的回复与其置信度
                            
                        yield res

            logger.info(f"共生成{step + 1}个样本, 总用时{time.time() - start} ...")
    

        # writing to output_file
        tsv_writer(gen_row(), output_file)

        # eval mode -> train mode
        model.train()



                
    def eval(self, mode, model, epoch, step, checkpoint_dir):
        
        # First - Generate
        logger.info(f"Start Generation on {mode} set ...")
        
        # 当前checkpoint_dir格式为result/experiment_v1/checkpoint-1-10000
        # 将checkpoint_dir中添加experiment_name
        # 创建output_file, 在checkpoint_dir下添加生成参数的tsv文件
        checkpoint_dir = os.path.join(checkpoint_dir, self.args.experiment_name)
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        output_file = self.add_gen_cfg(checkpoint_dir=checkpoint_dir, mode=mode)
        # 如果当前预测文件output_dir已经存在, skip
        if os.path.isfile(output_file):
            logger.info(f"skip, {output_file} already exists")
        else:
            if mode == "dev":
                self.Generate(model=model, test_loader=self.dev_loader, output_file=output_file)
            elif mode == "test":
                self.Generate(model=model, test_loader=self.test_loader, output_file=output_file)

        # Second - Evaluate
        logger.info(f"Start Evaluation on {mode} set ...")

        gth_list = get_labels(dialog_path=self.args.text_dir, mode=mode)

        start = time.time()
        if mode == 'dev':
            result = evaluate_on_tiktalk(output_file, gth_list, outdir=checkpoint_dir, mode='dev_metrics')
        elif mode == 'test':
            result = evaluate_on_tiktalk(output_file, gth_list, outdir=checkpoint_dir, mode='test_metrics')
        logger.info(f'evaluation cost time: {time.time() - start}')

        # add epoch-step info to result
        result["Mode"] = mode
        result["Epoch"] = epoch
        result["Step"] = step

        return result

    
    