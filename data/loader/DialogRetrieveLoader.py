# 结合Retrieve的dataloader
import os
import torch
import h5py
import json
import re
import cn_clip.clip as clip
import numpy as np
import time
import logging
from cn_clip.clip import load_from_name
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

# 使用检索增强版本的dataloader

def load_json(data_path):
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
            return data
    except:
        raise FileNotFoundError(f'Fail to load the data file {data_path}, please check if the data file exists')

# remove emoji
def remove_emoji(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  
        u"\U0001F300-\U0001F5FF"  
        u"\U0001F680-\U0001F6FF"  
        u"\U0001F1E0-\U0001F1FF"  
        u"\U00002702-\U000027B0"
        u"\U00010000-\U0010ffff"
        u"\U0001f926-\U0001f937"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  
        u"\u3030"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def TikTalkDataloader(args, tokenizer, frame_feat):
    
    # 加载数据
    train_dataset = TikTalkDataset(
        args=args,
        tokenizer=tokenizer,
        frame_feat=frame_feat,
        mode="train"
    )

    dev_dataset = TikTalkDataset(
        args=args,
        tokenizer=tokenizer,
        frame_feat=frame_feat,
        mode="dev"
    )

    test_dataset = TikTalkDataset(
        args=args,
        tokenizer=tokenizer,
        frame_feat=frame_feat,
        mode="test"
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        collate_fn=train_dataset.collate_fn
    )

    dev_loader = DataLoader(
        dataset=dev_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=dev_dataset.collate_fn
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=test_dataset.collate_fn
    )

    return train_loader, dev_loader, test_loader



class TikTalkDataset(Dataset):
    def __init__(self, args, tokenizer=None, frame_feat=None, mode='train'):
        self.args = args
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained('bert-base-chinese')
        self.mode = mode
        self.metadata = os.path.join(self.args.text_dir, f'{self.mode}.json')
        self.data = load_json(self.metadata)
        self.dialog_ids = list(self.data.keys())
        self.max_desc_length = args.max_desc_length
        self.max_his_length = args.max_his_length
        self.max_resp_length = args.max_resp_length
        self.max_prior_length = args.max_prior_length
        self.retrieve_num = args.retrieve_num
        self.retrieve_enable = args.retrieve_enable
        self.frame_feat = frame_feat
        

    def __len__(self):
        return len(self.dialog_ids)
    

    def __getboxInfo__(self):
        frame_num = 32
        grid = 4
        patch_size = 224
        width, height = grid * patch_size, grid * patch_size
        bbox_list = []
        for i in range(grid):
            for j in range(grid):
                bbox_list.append([i * patch_size, j * patch_size, (i+1) * patch_size, (j+1) * patch_size])
        bbox_feat = torch.tensor(bbox_list).unsqueeze(0).repeat(frame_num, 1, 1) 
        related_bbox = (bbox_feat[:, :, 2] - bbox_feat[:, :, 0]) * (bbox_feat[:, :, 3] - bbox_feat[:, :, 1]) / (width * height)
        related_bbox = related_bbox.unsqueeze(-1)
        bbox_feat = bbox_feat / torch.tensor([height, width, height, width]) 
        bbox_feat = torch.cat([bbox_feat, related_bbox], dim=-1)
        return bbox_feat
    

            
    def __text_process__(self, text):
        # clean the text data
        text = text.strip().lower()
        text = remove_emoji(text)
        text = re.sub(r'(?:@|\（via|\（来源)\S+', '', text)
        text = re.sub(r'[()（）ノ┻━┻Д.。‘’…]', '', text)
        text = re.sub(r'\.{2,}', '', text)
        text = re.sub(r'\s+', '', text)
        return text.strip()
        
    
    def __visual_process__(self, frame_feat):
        '''return global feature and local feature'''
        # [32, 17, 512]
        frame_feat = torch.from_numpy(frame_feat)
        glo_feat = frame_feat[:, 0, :]
        loc_feat = frame_feat[:, 1:, :]
        # get box position information
        # bbox_feat = self.__getboxInfo__() # [32, 16, 5]
        # loc_feat = torch.cat([loc_feat, bbox_feat], dim=-1)
        return glo_feat, loc_feat


    def __getitem__(self, index):
        dialog_id = self.dialog_ids[index]
        meta = self.data[dialog_id]
        vid = meta['video_id']

        '''Process text data'''
        desc = self.__text_process__(meta['description']) 
        
        history = meta['history'] 
        response = meta['response'] 
        ctxs = meta["ctxs"]

        prior_history = []
        prior_response = []
        for i in range(min(len(ctxs), self.retrieve_num)):
            prior_data = ctxs[i]
            tmp_history = prior_data["retrieved_history"]
            prior_response_list = prior_data["retrieved_response"]
            select_index = np.random.randint(0, len(prior_response_list))
            tmp_response = prior_response_list[select_index]
            prior_history.append(tmp_history)
            prior_response.append(tmp_response)
        
        # Process history info
        his = ''
        cnt = 0
        for hisdata in history:
            cnt += 1
            his += self.__text_process__(hisdata)
            if cnt < len(history):
                his += "[SEP]"

        # Process response info
        if self.mode == 'train':
            response = self.__text_process__(response)
        else:
            response = self.__text_process__(response[0])


        # Process video info
        video_feat = self.frame_feat[vid] # [32, 17, 512]
        glo_feat, loc_feat = self.__visual_process__(video_feat)
       
        return {
            "description": desc,
            "prior_history": prior_history,
            "prior_response": prior_response,
            "history": his,
            "response": response,
            "glo_feat": glo_feat,
            "loc_feat": loc_feat
        }
    
    def pad(self, raw_array, pad_id):
        max_len = max(len(i) for i in raw_array)
        for i in range(len(raw_array)):
            if len(raw_array[i]) < max_len:
                padding_len = max_len - len(raw_array[i])
                raw_array[i] = raw_array[i] + [pad_id] * padding_len
        return raw_array
        
    
    def collate_fn(self, batch):

        description = [x["description"] for x in batch]
        prior_history = [x["prior_history"] for x in batch]
        prior_response = [x["prior_response"] for x in batch]
        history = [x["history"] for x in batch]
        response = [x["response"] for x in batch]
        
        glo_feat = torch.stack([data["glo_feat"] for data in batch])
        loc_feat = torch.stack([data["loc_feat"] for data in batch])

        desc_clip_input_ids = clip.tokenize(description, context_length=self.max_desc_length)
        his_clip_input_ids = clip.tokenize(history, context_length=self.max_his_length)
        
        
        prior_input_ids = []
        prior_attn_mask = []
        token_type_ids = []


        his_input_ids = []
        his_attn_mask = []
        
        resp_input_ids = []
        resp_attn_mask = []
        labels = []

        # 按batch进行对齐

        # prior info
        for i in range(len(prior_history)):
            prior_his_list = prior_history[i]
            prior_resp_list = prior_response[i]

            # 添加[SEP]信息
            for j in range(len(prior_his_list)):
                prior_his_list[j] = prior_his_list[j] + "[SEP]"
                prior_resp_list[j] = prior_resp_list[j] + "[SEP]"

            # 组织成prior info
            prior_data = []
            token_type_data = []
            for j in range(min(len(prior_his_list), self.retrieve_num)):
                prior_his = self.tokenizer.encode(prior_his_list[j], add_special_tokens=False)
                prior_resp = self.tokenizer.encode(prior_resp_list[j], add_special_tokens=False)
                prior_data = prior_data + prior_his + prior_resp
                token_type_data = token_type_data + [0] * len(prior_his) + [1] * len(prior_resp)
            
            if len(prior_data) > self.max_prior_length:
                prior_data = prior_data[: self.max_prior_length]
                token_type_data = token_type_data[: self.max_prior_length]

            prior_input_ids.append(prior_data)
            prior_attn_mask.append([1] * len(prior_data))
            token_type_ids.append(token_type_data)

        # history
        for i in range(len(history)):
            his_data = history[i]
            his_data = self.tokenizer.encode(his_data, add_special_tokens=False)
            if len(his_data) > self.max_his_length - 2:
                his_data = his_data[:self.max_his_length - 2]
            his_data = [self.tokenizer.cls_token_id] + his_data + [self.tokenizer.sep_token_id]
            his_input_ids.append(his_data)
            his_attn_mask.append([1] * len(his_data))
        

        # 如果启用retrieve, 则拼接prior data & history
        if self.retrieve_enable:
            for i in range(len(history)):
                cur_prior_input_ids = prior_input_ids[i]
                cur_prior_attn_mask = prior_attn_mask[i]
                cur_his_input_ids = his_input_ids[i]
                cur_his_attn_mask = his_attn_mask[i]
                
                cur_token_type_ids = token_type_ids[i]
                
                his_input_ids[i] = cur_prior_input_ids + cur_his_input_ids
                his_attn_mask[i] = cur_prior_attn_mask + cur_his_attn_mask
                token_type_ids[i] = cur_token_type_ids + [0] * len(cur_his_input_ids)
            

        self.pad(his_input_ids, 0)
        self.pad(his_attn_mask, 0)
        self.pad(token_type_ids, 0)

        # add vision token_type_ids
        for i in range(len(history)):
            token_type_ids[i] = token_type_ids[i] + [2] * 104

        # response
        for i in range(len(response)):
            resp_data = response[i]
            resp_data = self.tokenizer.encode(resp_data, add_special_tokens=False)
            if len(resp_data) > self.max_resp_length - 2:
                resp_data = resp_data[: self.max_resp_length - 2]
            resp_data = [self.tokenizer.cls_token_id] + resp_data + [self.tokenizer.sep_token_id]
            resp_input_ids.append(resp_data)
            labels.append(resp_input_ids[i])
            resp_attn_mask.append([1] * len(resp_data))
        
        self.pad(resp_input_ids, 0)
        self.pad(resp_attn_mask, 0)
        self.pad(labels, -100)

        # Tensor(clip feature已经张量化)
        his_input_ids = torch.tensor(his_input_ids, dtype=torch.long)
        his_attn_mask = torch.tensor(his_attn_mask, dtype=torch.long)
        resp_input_ids = torch.tensor(resp_input_ids, dtype=torch.long)
        resp_attn_mask = torch.tensor(resp_attn_mask, dtype=torch.long)
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        
        return {
            "desc_clip_input_ids": desc_clip_input_ids,
            "his_clip_input_ids": his_clip_input_ids,
            "his_input_ids": his_input_ids,
            "his_attn_mask": his_attn_mask, 
            "resp_input_ids": resp_input_ids,
            "resp_attn_mask": resp_attn_mask,
            "token_type_ids": token_type_ids,
            "labels": labels,
            "glo_feat": glo_feat,
            "loc_feat": loc_feat,
            "text_history": history,
            "text_response": response
        }
        






