import os
import torch
import numpy as np
import math
import cn_clip.clip as clip
import torch.nn.functional as F
import torch.nn as nn
import logging
from cn_clip.clip import load_from_name
from transformers import AutoModel, BertConfig
from model.modeling_bart import BartEncoder, BartDecoder, BartForConditionalGeneration

logger = logging.getLogger()


# 将patch position feature加到局部特征中
class EncoderVid(nn.Module):
    def __init__(self, feat_dim, bbox_dim, feat_hidden, pos_hidden, input_dropout_p=0.3):
        super(EncoderVid, self).__init__()
        self.dim_feat = feat_dim # 512
        self.dim_bbox = bbox_dim # 5
        self.dim_hidden = feat_hidden # 512 
        self.input_dropout_p = input_dropout_p # 0.3

        input_dim = feat_dim # 输入维度为512
        
        input_dim += pos_hidden # pos_hidden = 128
        
        self.bbox_conv = nn.Sequential(
            nn.Conv2d(self.dim_bbox, pos_hidden, kernel_size=1),
            nn.BatchNorm2d(pos_hidden),
            nn.ReLU(),
            nn.Conv2d(pos_hidden, pos_hidden, kernel_size=1),
            nn.BatchNorm2d(pos_hidden),
            nn.ReLU(),
        )
        
        self.tohid = nn.Sequential(
            nn.Linear(feat_dim + pos_hidden, feat_hidden),
            nn.ELU(inplace=True),
        )
    
    def forward(self, loc_feat):
        bs, seg_num, seg_numf, numr, fdim = loc_feat.shape
        loc_feat = loc_feat.view(bs, seg_num * seg_numf, numr, fdim)
        roi_feat = loc_feat[:, :, :, :self.dim_feat]
        roi_bbox = loc_feat[:, :, :, self.dim_feat:(self.dim_feat + self.dim_bbox)]
        
        bbox_pos = self.bbox_conv(roi_bbox.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        bbox_features = torch.cat([roi_feat, bbox_pos], dim=-1)
        bbox_feat = self.tohid(bbox_features)
        return bbox_feat
    

# Selector选择器
class Selector(nn.Module):
    def __init__(self, topk, selection_method='gumbel', feat_dim=512, d_model=512):
        super(Selector, self).__init__()

        self.topk = topk
        self.selection_method = selection_method

        self.Q_lin = nn.Linear(feat_dim, d_model)
        self.K_lin = nn.Linear(d_model, d_model)

        self.Q_norm = nn.LayerNorm(d_model, eps=1e-12)
        self.K_norm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, Q, K, V):

        # 输入Q维度为[bs, q_dim, 1]
        # 输入K维度为[bs，num_seg, k_dim]或者为[bs * num_seg * seg_frame, num_obj, k_dim] 其为片段特征
        # V.dim() == 3时为选择patch区域，V.dim() == 5时为选择片段
        # 输入V的维度为[bs，num_seg, seg_frame, num_obj, dim] 或 [bs * num_seg * seg_frame, num_obj, dim]
        
        '''
        reg_selector:
            Q: [bs * topk * seg_numf, d_model, 1]
            K: [bs * topk * seg_numf, obj_num, d_model]
            V: [bs * topk * seg_numf, obj_num, d_model]
            return: [bs * topk * seg_numf , topj, d_model]
        seg_selector:
            Q: [bs, d_model, 1]
            K: [bs, seg_num, d_model]
            V: [bs, seg_num, seg_numf * patch_num * d_model]
            return: [bs, topk, seg_numf, patch_num, d_model]
        '''
        
        # n_select为可供选择的数量
        bs, n_select, _ = K.shape 
        
        obj_num, obj_dim = V.shape[-2:]

        Q = self.Q_norm(self.Q_lin(Q.squeeze(-1))) # [bs, d_model]
        K = self.K_norm(self.K_lin(K)) # [bs, seg_num, d_model]

        logit_scale = 1
        x_logits = logit_scale * K @ Q.unsqueeze(dim=-1) # [bs, n_select, 1]
        x_logits = torch.softmax(x_logits.squeeze(dim=-1), dim=-1) # [bs, n_select]

        selected_list = []
        for _ in range(self.topk):
            selection = F.gumbel_softmax(logits=x_logits, tau=1, dim=-1) # [bs, n_select]
            selection = selection.unsqueeze(dim=1) # [bs, 1, n_select]
            
            if V.dim() == 3:
                # [bs, 1, n_select] * [bs, n_select, dim] = [bs, 1, dim]
                selected_list.append(
                    torch.matmul(selection, V.view(bs, n_select, -1))) 
            else:
                 # [bs, seg_numf, obj_num, obj_dim]
                selected_list.append(torch.matmul(selection, V.view(bs, n_select, -1)).view(bs, -1, obj_num, obj_dim))
        # 如果是片段选择，则为[bs, topk * seg_numf, num_obj, dim]
        # 如果是区域选择，则为[bs * topk * seg_numf, topj, dim]
        selected_list = torch.cat(selected_list, dim=1) 
        return selected_list






class Case(nn.Module):
    def __init__(
        self,
        config,
        args,
        feature_dim = 512,   
        word_dim = 768,
        seg_num = 8,              
        d_model = 768,          
        dropout = 0.1,    
        max_desc_length = 40,
        max_prior_length = 100,
        max_his_length = 140,
        max_resp_length = 40,   
        max_video_length = 50,   
        vocab_size = 21128,
    ):
        super(Case, self).__init__()
        # data
        self.args = args
        self.max_desc_length = max_desc_length
        self.max_prior_length = max_prior_length
        self.max_his_length = max_his_length
        self.max_resp_length = max_resp_length
        self.max_video_length = max_video_length
        self.topk = args.topk
        self.topj = args.topj
        self.feature_dim = feature_dim
        self.word_dim = word_dim
        self.seg_num = seg_num
        self.seg_numf = int(32 / self.seg_num)
        self.d_model = d_model
        self.dropout = dropout
        self.vocab_size = vocab_size
        self.device = args.device

        # config
        self.config = config
        
        # pretrained_bart
        pretrained_bart = BartForConditionalGeneration.from_pretrained(self.args.model_name_or_path)

        # word embedding & type embedding
        
        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)
        
        
        # Load Pretrained State
        embed_tokens_weight = nn.Parameter(pretrained_bart.model.shared.weight)
        
        # 绑定model中的shared_embedding
        self.embed_tokens.weight = embed_tokens_weight

        # clip_encoder
        self.clip_encoder, self.clip_processor = load_from_name('ViT-B-16')
       
        # Selector
        self.seg_selector = Selector(topk=self.topk)
        self.reg_selector = Selector(topk=self.topj)

        self.seg_lin = nn.Linear(self.feature_dim, self.d_model)
        self.seg_norm = nn.LayerNorm(self.d_model, eps=1e-12)

        self.sel_lin = nn.Linear(self.feature_dim, self.d_model)
        self.sel_norm = nn.LayerNorm(self.d_model, eps=1e-12)

        self.video_drop = nn.Dropout(dropout)

        # Self-Attention
        self.encoder = BartEncoder(config=config)

        # Load Pretrained encoder State
        pretrained_dict = pretrained_bart.state_dict()
        base_prefix = "model.encoder."
        required_params = {k.replace(base_prefix, ""): v for k, v in pretrained_dict.items() if k.startswith(base_prefix)}
        self.encoder.load_state_dict(required_params, strict=True)

        # check loading
        for name, param in self.encoder.named_parameters():
            pretrained_name = base_prefix + name
            if pretrained_name in pretrained_dict:
                pretrained_param = pretrained_dict[pretrained_name]
                assert torch.allclose(pretrained_param.data, param.data, atol=1e-6), f"error in loading Pretrained Key: {name} ..."
            else:
                raise ValueError(f"Warning: {pretrained_name} not found in pretrained dictionary ...")

        self.decoder = BartDecoder(config=config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Load Pretrained decoder State
        base_prefix = "model.decoder."
        required_params = {k.replace(base_prefix, ""): v for k, v in pretrained_dict.items() if k.startswith(base_prefix)}
        self.decoder.load_state_dict(required_params, strict=True)
        self.lm_head.load_state_dict(pretrained_bart.lm_head.state_dict())

        # check loading
        for name, param in self.decoder.named_parameters():
            pretrained_name = base_prefix + name
            if pretrained_name in pretrained_dict:
                pretrained_param = pretrained_dict[pretrained_name]
                assert torch.allclose(pretrained_param.data, param.data, atol=1e-6), f"error in loading Pretrained Key: {name} ..."
            else:
                raise ValueError(f"Warning: {pretrained_name} not found in pretrained dictionary ...")

        self.criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=-100)

        
    def forward(
            self, 
            desc_clip_input_ids=None,
            his_clip_input_ids=None,
            his_input_ids=None,
            his_attn_mask=None,
            resp_input_ids=None,
            resp_attn_mask=None,
            token_type_ids=None,
            labels=None,
            glo_feat=None,
            loc_feat=None,
            text_history=None,
            text_response=None,
            is_training=True
        ):
        
        # import pdb; pdb.set_trace()

        # 传入GPU
        desc_clip_input_ids = desc_clip_input_ids.to(self.device)
        his_clip_input_ids = his_clip_input_ids.to(self.device)
        his_input_ids = his_input_ids.to(self.device)
        his_attn_mask = his_attn_mask.to(self.device)
        resp_input_ids = resp_input_ids.to(self.device)
        resp_attn_mask = resp_attn_mask.to(self.device)
        # token_type_ids = token_type_ids.to(self.device)
        glo_feat = glo_feat.to(self.device)
        loc_feat = loc_feat.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        

        # loc_feat: [bs, numf, numr, fdim]
        # numf: frame number
        # numr: patch number
        # fdim: visual dimension
        bs, numf, numr, fdim  = loc_feat.size()
        # seg_num: 片段数量
        # seg_numf: 一个片段包含多少帧
        seg_num, seg_numf = self.seg_num, self.seg_numf
        # 归一化
        loc_feat /= loc_feat.norm(dim=-1, keepdim=True) 
        glo_feat /= glo_feat.norm(dim=-1, keepdim=True) 
        # [bs, 8, 4, 16, 517] (现在是512，因为暂时不做空间的编码)
        loc_feat = loc_feat.view(bs, seg_num, seg_numf, numr, fdim).float()
        # [bs, 8, 4, 16, 512]
        # loc_feat = self.encode_vid(loc_feat).view(bs, seg_num, seg_numf, numr, -1)
        # [bs, 8, 4, 512]
        glo_feat = glo_feat.view(bs, seg_num, seg_numf, -1).float()
        # 全局片段特征 [bs, 8, 512]
        glo_seg_feat = torch.mean(glo_feat, dim=-2, keepdim=False) 
        
        # [bs, 512]
        desc_sentence_embedding = self.clip_encoder.encode_text(desc_clip_input_ids).float()
        desc_sentence_embedding = desc_sentence_embedding / desc_sentence_embedding.norm(dim=-1, keepdim=True)
        his_sentence_embedding = self.clip_encoder.encode_text(his_clip_input_ids).float()
        his_sentence_embedding = his_sentence_embedding / his_sentence_embedding.norm(dim=-1, keepdim=True)
        # 转变维度为[bs, 512, 1]
        desc_sentence_embedding = desc_sentence_embedding.view(bs, -1, 1)
        his_sentence_embedding = his_sentence_embedding.view(bs, -1, 1)
        
        # 即[bs, topk * 4, 16, 512]=[bs, 4*topk, 16, 512]
        selected_patches = self.seg_selector(his_sentence_embedding, glo_seg_feat, loc_feat) 
        # his_feat_tmp维度为[bs, topk*seg_numf, 512, 1]
        his_feat_tmp = his_sentence_embedding.unsqueeze(dim=1).repeat(1, selected_patches.shape[1], 1, 1) 
        # his_feat_tmp维度转变为[bs * topk * seg_numf, 512, 1]
        his_feat_tmp = his_feat_tmp.view(-1, his_feat_tmp.shape[-2], his_feat_tmp.shape[-1]) 
        # selected_patches维度转变为[bs * topk * seg_numf, obj_num, obj_dim]
        selected_patches = selected_patches.view(-1, selected_patches.shape[-2], selected_patches.shape[-1]) 
        # 得到选择的视频特征维度为[bs * topk * seg_numf, topj, obj_dim]
        selected_patches = self.reg_selector(his_feat_tmp, selected_patches, selected_patches) 
        # 将选择的视频特征维度转变为[bs, topk * seg_numf * topj, obj_dim]
        # 即[bs, topk * 4 * topj, 512]=[bs, topk*4*topj, 512]
        selected_patches = selected_patches.view(bs, -1, self.feature_dim)

        visual_len = seg_num + self.topk * seg_numf * self.topj

        glo_seg_feat = self.seg_norm(F.gelu(self.seg_lin(glo_seg_feat))) 
        selected_feat = self.sel_norm(F.gelu(self.sel_lin(selected_patches))) 



        # [bs, seq_len, 768]
        his_feat = self.embed_tokens(his_input_ids) * self.embed_scale

        video_feat = torch.cat([glo_seg_feat, selected_feat], dim=1)
        video_feat = self.video_drop(video_feat)
        video_attn_mask = torch.ones((bs, visual_len), dtype=torch.long, device=his_feat.device)

        hidden_states = torch.cat([his_feat, video_feat], dim=1)
        attention_mask = torch.cat([his_attn_mask, video_attn_mask], dim=1)



        src_feat = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids
        )[0]

        outputs = self.decoder(
            input_ids=resp_input_ids,
            attention_mask=resp_attn_mask,
            encoder_hidden_states=src_feat,
            encoder_attention_mask=attention_mask,
        )

        logits = self.lm_head(outputs[0])
        
        if is_training:
            # [bs, tgt_len-1, vocab_size]
            logits = logits[:, :-1, :].contiguous()
            # [bs, tgt_len-1]
            labels = labels[:, 1:].contiguous()
            
            loss = self.criterion(logits.view(-1, self.vocab_size), labels.view(-1))

            return (loss, logits, labels)
        else:

            return (logits, )
    
        

    def prepare_inputs_for_generation(self, curr_ids):
        
        desc_clip_input_ids = self.expand_for_beams(self.desc_clip_input_ids, num_beams=self.num_beams)
        his_clip_input_ids = self.expand_for_beams(self.his_clip_input_ids, num_beams=self.num_beams)
        his_input_ids = self.expand_for_beams(self.his_input_ids, num_beams=self.num_beams)
        his_attn_mask = self.expand_for_beams(self.his_attn_mask, num_beams=self.num_beams)
        # token_type_ids = self.expand_for_beams(self.token_type_ids, num_beams=self.num_beams)
        glo_feat = self.expand_for_beams(self.glo_feat, num_beams=self.num_beams)
        loc_feat = self.expand_for_beams(self.loc_feat, num_beams=self.num_beams)
        resp_attn_mask = torch.ones(curr_ids.size(), dtype=torch.long, device=curr_ids.device)


        return {
            "desc_clip_input_ids": desc_clip_input_ids,
            "his_clip_input_ids": his_clip_input_ids,
            "his_input_ids": his_input_ids,
            "his_attn_mask": his_attn_mask,
            # "token_type_ids": token_type_ids,
            "glo_feat": glo_feat,
            "loc_feat": loc_feat,
            "resp_input_ids": curr_ids,
            "resp_attn_mask": resp_attn_mask,
            "is_training": False,
        }

    
    # 适应beam search作扩展
    def expand_for_beams(self, x, num_beams):
        # 计算需要扩展的数量
        num_expand = num_beams
        # 如果x为None或num_expand == 1，则不需要进行扩展，直接返回x
        if x is None or num_expand == 1:
            return x
        # 获取输入张量x的形状信息
        input_shape = list(x.shape) 
        # 计算扩展后的形状
        expanded_shape = input_shape[:1] + [num_expand] + input_shape[1:] 
        x = x.unsqueeze(1).expand(expanded_shape)
        # (batch_size * num_beams, ...)
        x = x.contiguous().view([input_shape[0] * num_expand] + input_shape[1:])
        return x
    
        
        
        


    def generate(
        self, 
        # data
        desc_clip_input_ids,
        his_clip_input_ids,
        his_input_ids,
        his_attn_mask,
        # token_type_ids,
        glo_feat,
        loc_feat,
        # generation config
        bos_token_id,
        pad_token_id,
        eos_token_ids,
        max_gen_len,
        do_sample,
        num_beams,
        temperature,
        top_k,
        top_p,
        length_penalty,
        repetition_penalty
        
    ):
        
        # 记录参数
        batch_size = his_input_ids.shape[0]
        vocab_size = self.vocab_size
        self.max_gen_len = max_gen_len
        self.num_beams = num_beams

        # 存入self, 以便在prepare_inputs_for_generation时调用
        self.desc_clip_input_ids = desc_clip_input_ids
        self.his_clip_input_ids = his_clip_input_ids
        self.his_input_ids = his_input_ids
        self.his_attn_mask = his_attn_mask
        # self.token_type_ids = token_type_ids
        self.glo_feat = glo_feat
        self.loc_feat = loc_feat
        
        # 准备resp_input_ids
        resp_input_ids = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=self.device)
        cur_len = 1

        if self.num_beams > 1:
            output = self.beam_search(
                input_ids=resp_input_ids,
                cur_len=cur_len,
                max_gen_len=max_gen_len,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=pad_token_id,
                eos_token_ids=eos_token_ids,
                batch_size=batch_size,
                length_penalty=length_penalty,
                num_beams=num_beams,
                vocab_size=vocab_size
            )

        else:
            output = self.greedy_search(
                input_ids=resp_input_ids,
                cur_len=cur_len,
                max_gen_len=max_gen_len,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=pad_token_id,
                eos_token_ids=eos_token_ids,
                batch_size=batch_size,
            )

        return output
        


    def save_pretrained(self, save_directory):
        save_file = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), save_file)



    def greedy_search(
        self,
        input_ids,
        cur_len,
        max_gen_len,
        do_sample,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        pad_token_id,
        eos_token_ids,
        batch_size
    ):
        
        
        unfinished_sents = []
        # 初始喂入时，input_ids即为[bs, 1]，包含了一个<BOS> token
        # cur_unfinished指示哪些句子还没有生成完毕，初始创建为形状为[bs]，且填满1的张量
        cur_unfinished = input_ids.new(batch_size).fill_(1)
        # log of scores for each sentence in the batch
        # batch中每个句子的log分数
        logprobs = []
        
        # 循环中的input_ids一直只为生成回复部分的input_ids
        # 真正输入模型的input_ids会在prepare_inputs_for_generation时拼接历史上下文进行返回
        while cur_len < max_gen_len:
            # prepare_inputs_for_generation负责完成上下文及视频序列的拼接
            model_inputs = self.prepare_inputs_for_generation(input_ids)
            # model_outputs返回的是class_logits,维度为[bs, max_text_len, vocab_size]
            outputs = self(**model_inputs)
            next_token_idx = cur_len - 1
            # next_token_logits为该word index的词汇表概率，形状为[bs, vocab_size]
            next_token_logits = outputs[0][:, next_token_idx, :]
           
            # repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
            # 重复惩罚默认为3
            if repetition_penalty != 1.0:
                # 遍历每个batch
                for i in range(batch_size):
                    # previous_token为每个batch中所有input_ids的单词集合
                    for previous_token in set(input_ids[i].tolist()):
                        # 如果next_token_logits[i, previous_token] < 0，则负数 * 重复惩罚变为更小的负数使其概率降低
                        if next_token_logits[i, previous_token] < 0:
                            next_token_logits[i, previous_token] *= repetition_penalty
                        # 如果next_token_logits[i, previous_token] > 0，则正数 / 重复惩罚使其变为更小的正数使其概率降低
                        else:
                            next_token_logits[i, previous_token] /= repetition_penalty
            # Greedy decoding
            # next_token维度为[bs]
            # next_token为greedy decoding下解码出的最大概率的word
            next_token = torch.argmax(next_token_logits, dim=-1) 
            # 计算词汇表空间的得分
            _scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size, vocab_size)
            # 使用torch.gather函数选取特定单词的log概率
            # torch.gather(input, dim, index)：沿dim轴收集input的index值
            # 将next_token维度扩展至[bs, 1]以匹配_scores的[bs, vocab_size]
            _scores = torch.gather(_scores, -1, next_token.unsqueeze(-1))  # (batch_size, 1) 这是作者写的
            # 将当前next_token的log概率值加入logprobs列表
            logprobs.append(_scores)  # (batch_size, 1)
            unfinished_sents.append(cur_unfinished)

            # update generations and finished sentences
            # tokens_to_add维度为[bs]，指示每个batch要加入哪个新token
            # input_ids维度为[bs, cur_len]
            # tokens_to_add: 在batch generation中，batch中已完成的生成pad_token_id，batch中未完成的生成next_token
            tokens_to_add = next_token * cur_unfinished + pad_token_id * (1 - cur_unfinished)
            input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
            # eos_token_ids中包含了[sep_token_id，pad_token_id]
            for eos_token_id in eos_token_ids:
                # cur_unfinished用于跟踪哪些句子尚未完成生成
                # 若token_to_add.ne(eos_token_id): 当前tokens_to_add中不为结束标记的置为1，为结束标记的置为0
                cur_unfinished = cur_unfinished.mul(tokens_to_add.ne(eos_token_id).long())
            # 当前长度 + 1
            cur_len = cur_len + 1

            # 检查是否所有句子都已经生成结束，如果是则跳出循环表示生成完成
            if cur_unfinished.max() == 0:
                break

        # add eos_token_ids to unfinished sentences
        # 如果生成长度达到了最大长度(max_length)，则将结束标记添加到未完成的句子中
        if cur_len == max_gen_len:
            input_ids[:, -1].masked_fill_(cur_unfinished.to(dtype=torch.bool), eos_token_ids[0])

        # 将每次target word的输出概率拼接在一起
        # 由于每次logprobs添加的均为形状为[bs, 1]的log probability，因此cat后为[bs, max_gen_len]
        # 注意此时的max_gen_len为一个batch中生成文本的最大长度，如果提前结束则为最长的那个，未提前结束则为max_len
        logprobs = torch.cat(logprobs, dim=1)
        # unfinished_sents通过stack拼接为形状[bs, max_gen_len]的矩阵，还没结束时为1，结束则为0
        unfinished_sents = torch.stack(unfinished_sents, dim=1).float()
        # logprobs与unfinished_sents相乘得到形状为[bs, max_gen_len]，将结束后生成的log分数置0的矩阵
        # 再在dim=1上作累加，得[bs]的log总计分数，记录batch中每个生成句log分数累积
        sum_logprobs = (logprobs * unfinished_sents).sum(dim=1)

        # 此时的input_ids已经完成，一般来说长度为[bs, max_length]，不过如果提前结束需要作padding
        pad_len = max_gen_len - input_ids.shape[1]
        if pad_len > 0:
            padding_ids = input_ids.new(batch_size, pad_len).fill_(pad_token_id)
            input_ids = torch.cat([input_ids, padding_ids], dim=1)

        # (batch_size, n_best, max_len), (batch_size, n_best)
        # 返回的input_ids维度为[bs, 1, max_length]
        # 返回的sum_logprobs维度为[bs, 1]
        return input_ids.unsqueeze(1), sum_logprobs.unsqueeze(1)



    # beam search
    def beam_search(
        self,
        input_ids,
        cur_len,
        max_gen_len,
        do_sample,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        pad_token_id,
        eos_token_ids,
        batch_size,
        length_penalty,
        num_beams,
        vocab_size,
    ):
        

        """ Generate sequences for each example with beam search.
        """
        # Expand input to num beams
        input_ids = input_ids.unsqueeze(1).expand(batch_size, num_beams, cur_len)
        # (batch_size * num_beams, cur_len)
        input_ids = input_ids.contiguous().view(batch_size * num_beams, cur_len)  

        # generated hypotheses
        num_keep_best = 1
        generated_hyps = [
            BeamHypotheses(num_keep_best, max_gen_len, length_penalty, early_stopping=False) for _ in range(batch_size)
        ]
        # NOTE: Expand >1 words to leave some spare tokens to keep the
        # beam size, because some sentences may end here and cannot expand
        # in the next level
        # 扩展到下一级的候选词数量最多为2个
        TOPN_PER_BEAM = 2

        # scores for each sentence in the beam
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        # 保持每个beam中只有一个候选项被考虑
            # 确保每个步骤只保留当前最有可能的候选项
            # 其余候选项的分数设置为很小的负数，使其不会被选择
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)  # shape (batch_size * num_beams,)

        # cache compute states
        past = None

        # done sentences
        done = [False for _ in range(batch_size)]

        while cur_len < max_gen_len:
            model_inputs = self.prepare_inputs_for_generation(input_ids)
            outputs = self(**model_inputs)  # (batch_size * num_beams, cur_len, vocab_size)
            
            
            next_token_idx = cur_len - 1
            # (batch_size * num_beams, vocab_size)
            scores = outputs[0][:, next_token_idx, :]  
            assert outputs[0].shape[1] == model_inputs['resp_input_ids'].shape[1]


            # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
            if repetition_penalty != 1.0:
                for i in range(batch_size * num_beams):
                    for previous_token in set(input_ids[i].tolist()):
                        # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                        if scores[i, previous_token] < 0:
                            scores[i, previous_token] *= repetition_penalty
                        else:
                            scores[i, previous_token] /= repetition_penalty

            if do_sample:
                # Temperature (higher temperature => more likely to sample low probability tokens)
                if temperature != 1.0:
                    scores = scores / temperature
                # Top-p/top-k filtering
                scores = top_k_top_p_filtering(
                    scores, top_k=top_k, top_p=top_p, min_tokens_to_keep=2
                )  # (batch_size * num_beams, vocab_size)
                # Sample [TOPN_PER_BEAM] next words for each beam (so we have some spare tokens and match output of greedy beam search)
                next_words = torch.multinomial(F.softmax(scores, dim=-1),
                        num_samples=TOPN_PER_BEAM)  # (batch_size * num_beams, TOPN_PER_BEAM)
                # Compute next scores
                _scores = F.log_softmax(scores, dim=-1)  # (batch_size * num_beams, vocab_size)
                _scores = torch.gather(_scores, -1, next_words)  # (batch_size * num_beams, TOPN_PER_BEAM)
                next_scores = _scores + beam_scores[:, None].expand_as(_scores)  # (batch_size * num_beams, TOPN_PER_BEAM)
                # Match shape of greedy beam search
                beam_indices = torch.arange(num_beams) * vocab_size
                beam_indices = beam_indices.repeat(batch_size, TOPN_PER_BEAM).to(next_words.device)
                next_words = next_words.view(batch_size, TOPN_PER_BEAM * num_beams)  # (batch_size, TOPN_PER_BEAM * num_beams)
                next_words = next_words + beam_indices
                next_scores = next_scores.view(batch_size, TOPN_PER_BEAM * num_beams)  # (batch_size, TOPN_PER_BEAM * num_beams)
            else:
                # do greedy beam search
                scores = F.log_softmax(scores, dim=-1)  # (batch_size * num_beams, vocab_size)
                assert scores.size() == (batch_size * num_beams, vocab_size)
                # Add the log prob of the new beams to the log prob of the beginning of the sequence (sum of logs == log of the product)
                # beam_scores shape: [bs * num_beams]
                # 变为_scores后，只有第一个beam中的log prob被保留，其余beam变为-1e9
                _scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)
                # re-organize to group the beam together (we are keeping top hypothesis accross beams)
                _scores = _scores.view(batch_size, num_beams * vocab_size)  # (batch_size, num_beams * vocab_size)
                # [bs, TOPN_PER_BEAM]
                # 为每个beam保留TOPN_PER_BEAM个候选词汇
                # [bs, TOPN_PER_BEAM * num_beams]
                next_scores, next_words = torch.topk(_scores, TOPN_PER_BEAM * num_beams, dim=1, largest=True, sorted=True)

            assert next_scores.size() == next_words.size() == (batch_size, TOPN_PER_BEAM * num_beams)


            # 接下来，对于每个输入句子(batch_ex)，检查是否已经生成完毕
            # 如果已经生成完毕：
                # 则在"next_batch_beam"中添加num_beams个(0, pad_token_id)三元组，用于填充batch
            # 对于未生成完毕的句子：
                # 遍历next_words[batch_ex]和next_scores[batch_ex]


            # next batch beam content
            # list of (batch_size * num_beams) tuple(next hypothesis score, next word, current position in the batch)
            next_batch_beam = []

            # for each sentence
            for batch_ex in range(batch_size):

                # if we are done with this sentence
                # next_scores shape: [bs, TOPN_PER_BEAM * num_beams]
                done[batch_ex] = done[batch_ex] or generated_hyps[batch_ex].is_done(next_scores[batch_ex].max().item())
                if done[batch_ex]:
                    next_batch_beam.extend([(0, pad_token_id, 0)] * num_beams)  # pad the batch
                    continue

                # next sentence beam content
                next_sent_beam = []

                # next words for this sentence
                # 遍历当前time step的候选next_word_idx和next_word_score
                for idx, score in zip(next_words[batch_ex], next_scores[batch_ex]):

                    # get beam and word IDs
                    # 当前词属于哪个beam
                    beam_id = idx // vocab_size
                    # 当前词的true word id
                    word_id = idx % vocab_size

                    # end of sentence, or next word
                    # 如果当前word id是结束词或者已经到达最大生成长度
                    if word_id.item() in eos_token_ids or cur_len + 1 == max_gen_len:
                        # 当前input_ids shape: [bs * num_beams, cur_len]
                        # 因为已经达到终止条件，尝试向generated_hyps中加入当前生成的input_ids序列
                        generated_hyps[batch_ex].add(
                            # 得到候选word对应beam的input_ids, score
                            input_ids[batch_ex * num_beams + beam_id, :cur_len].clone(), score.item()
                        )
                    # 向next_sent_beam中加入num_beams个候选词的信息
                        # (score, true_word_id, beam_info)三元组
                        # beam_info是在综合考虑batch_size和beam_size下的beam序列id
                    else:
                        next_sent_beam.append((score, word_id, batch_ex * num_beams + beam_id))

                    # the beam for next step is full
                    if len(next_sent_beam) == num_beams:
                        break

                # update next beam content
                # 如果当前全局cur_len已经达到max_length终止条件, 则next_sent_beam数组中不能有内容
                if cur_len + 1 == max_gen_len:
                    assert len(next_sent_beam) == 0
                # 否则, next_sent_beam数组长度应为num_beams
                else:
                    assert len(next_sent_beam) == num_beams

                if len(next_sent_beam) == 0:
                    next_sent_beam = [(0, pad_token_id, 0)] * num_beams  # pad the batch
                # batch_beam添加展平的next_sent_beam
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == num_beams * (batch_ex + 1)

            # sanity check / prepare next batch
            assert len(next_batch_beam) == batch_size * num_beams
            # beam_scores: [bs*num_beams]
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            # beam_words: [bs*num_beams]
            beam_words = input_ids.new([x[1] for x in next_batch_beam])
            # beam_idx: [bs*num_beams]
            beam_idx = input_ids.new([x[2] for x in next_batch_beam])

            # re-order batch
            # 从候选词中挑出对应beam_info的input_ids,并将其与对应的beam_words结合
            input_ids = input_ids[beam_idx, :]
            input_ids = torch.cat([input_ids, beam_words.unsqueeze(1)], dim=-1)

            # re-order internal states
            if past:
                reordered_past = []
                for layer_past in past:
                    # get the correct batch idx from layer past batch dim
                    # batch dim of `past` and `mems` is at 1st position
                    reordered_layer_past = [layer_past[i].unsqueeze(0).clone().detach() for i in beam_idx]
                    reordered_layer_past = torch.cat(reordered_layer_past, dim=0)
                    # check that shape matches
                    assert reordered_layer_past.shape == layer_past.shape
                    reordered_past.append(reordered_layer_past)
                past = tuple(reordered_past)

            # update current length
            cur_len = cur_len + 1

            # stop when we are done with each sentence
            if all(done):
                break

        # 根据beam_search的结果选择最佳的文本序列，并生成相应的目标序列
        # 创建tgt_len用于存储每个生成文本序列的长度
        tgt_len = torch.ones(batch_size, num_keep_best, dtype=torch.long)
        # 创建logprobs用于存储每个生成文本序列的对数概率
        logprobs = torch.zeros(batch_size, num_keep_best,
                dtype=torch.float).fill_(-1e5).to(input_ids.device)
        # 创建all_best用于存储每个batch中最佳的生成文本序列
        all_best = []
        # 遍历每个batch对应的generated_hyps
        for i, hypotheses in enumerate(generated_hyps):
            best = []
            # hyp中的信息以(score, hyp)的二元组形式保存
            hyp_scores = torch.tensor([x[0] for x in hypotheses.hyp])
            # 得到hyp中前min(num_keep_best, len(hyp_scores))个序列的索引
            _, best_indices = torch.topk(hyp_scores,
                    min(num_keep_best, len(hyp_scores)), largest=True)
            # 遍历这些最佳索引
            for best_idx, hyp_idx in enumerate(best_indices):
                conf, best_hyp = hypotheses.hyp[hyp_idx]
                # 将best_hyp存入best数组
                best.append(best_hyp)
                # logprobs存储置信度
                logprobs[i, best_idx] = conf
                # tgt_len存储最佳文本的长度
                tgt_len[i, best_idx] = len(best_hyp) + 1  # +1 for the <EOS> symbol

            # 将batch中该样本对应的best数组存入all_best
            # all_best中存在batch中每个样本对应的best数组
            all_best.append(best)

        # generate target batch, pad to the same length
        # 填入生成文本
        decoded = input_ids.new(batch_size, num_keep_best, max_gen_len).fill_(pad_token_id)
        for batch_idx, best in enumerate(all_best):
            for best_idx, hypo in enumerate(best):
                decoded[batch_idx, best_idx, : tgt_len[batch_idx, best_idx] - 1] = hypo
                decoded[batch_idx, best_idx, tgt_len[batch_idx, best_idx] - 1] = eos_token_ids[0]

        return decoded, logprobs




def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


    
class BeamHypotheses(object):
    def __init__(self, n_hyp, max_length, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.n_hyp = n_hyp
        self.hyp = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.hyp)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.n_hyp or score > self.worst_score:
            self.hyp.append((score, hyp))
            if len(self) > self.n_hyp:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.hyp)])
                del self.hyp[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """
        if len(self) < self.n_hyp:
            return False
        elif self.early_stopping:
            return True
        else:
            return self.worst_score >= best_sum_logprobs / self.max_length ** self.length_penalty
