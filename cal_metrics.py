# tfshen 2024.2.13
# 汇集不同类型的中文评价指标
# 该文件假定输出回复和真实标签均没有作分词或空格处理，可通过mode来指定
import os
import torch
import json
import re
import nltk
import math
import jieba
import logging
from nlgeval import NLGEval
from tqdm import tqdm
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.translate.nist_score import sentence_nist
from nltk.util import ngrams 
from rouge_chinese import Rouge
from bert_score import score

# 定义评估指标对应logger
def create_logger(log_path):
    """
    将日志输出到日志文件和控制台
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(
        filename=log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger

# 文件写入函数
def write_json(result, outfile):
    # 如果outfile不为None，则判断其是文件夹还是json文件，将其输出
    if outfile is not None:
        assert outfile.endswith('.json'), 'outfile需为json文件'
        # 如果outfile路径已存在，则不再写入
        if os.path.exists(outfile):
            logger.info(f'评估文件{outfile}已存在 ...')
        else:
            with open(outfile, 'w') as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
            logger.info(f'成功写入评估文件{outfile} ...')
    else:
        logger.info(f'outfile文件为None，不进行写入')

# 创建logger
log_path = './metric.log'
logger = create_logger(log_path=log_path)

# 定义全局变量nlgeval
n = NLGEval(metrics_to_omit=['METEOR', 'SPICE', 'SkipThoughtCS','EmbeddingAverageCosineSimilarity','VectorExtremaCosineSimilarity','GreedyMatchingScore'])

# 计算Dist
def distinct_n_sentence_level(sentence, n):
    if len(sentence) == 0:
        return 0.0  # Prevent a zero division
    distinct_ngrams = set(ngrams(sentence, n))
    return len(distinct_ngrams) / len(sentence)


# 使用nlgeval库计算中文评价指标(默认n=n)
# METEOR指标使用nltk库进行计算，而不使用nlgeval库，这是因为使用nlgeval库进行计算可能会导致显存不足的问题
# 两种模式：word-level和jieba分词
# outfile为指定文件，以json结尾
def cal_nlgeval_metrics(hyp, ref, n=n, mode='word', outfile=None):
    assert mode in ['word', 'jieba'], '指定评估标准的计算模式须指定为[word]或[jieba]'
    # hyp为预测文本列表
    # ref为参考文本列表，len(ref)代表每个hyp有多少条参考文本
    # 定义result集合存储最终指标
    result = {}
    # 定义all_caps，用来计算dist指标
    all_caps = []
    
    # 定义global metrics
    global_bleu1 = 0
    global_bleu2 = 0
    global_bleu3 = 0
    global_bleu4 = 0
    # global_meteor = 0
    global_cider = 0
    global_rouge = 0

    # 定义hyp_list与ref_list用于CIDEr分数计算
    hyp_list = []
    ref_list = [[] for i in range(len(ref))]
    
    # 遍历所有hyp,每条hyp对应多个ref
    for sample_idx in tqdm(range(len(hyp))):
        # 获得该hyp，初始化为cap
        cap = hyp[sample_idx]

        # 根据模式先将cap加入all_caps，再对cap进行分词
        if mode == 'word':
            cap = ' '.join(cap)
            all_caps += cap.split()
        elif mode == 'jieba':
            jieba_cap = jieba.cut(cap)
            cap = ' '.join(jieba_cap)
            all_caps += cap.split()
        
        # 将分词后的cap添加进hyp_list
        hyp_list.append(cap)

        # 初始化该对话的指标，最后对所有ref取平均（除了CIDEr，CIDEr需要对整个文档作TF-IDF计算）
        unit_bleu1 = 0
        unit_bleu2 = 0
        unit_bleu3 = 0
        unit_bleu4 = 0
        # unit_meteor = 0
        unit_rouge = 0

        # 遍历该hyp对应的多个ref
        for ref_idx in range(len(ref)):
            # 依次得到该hyp对应的每个label
            label = ref[ref_idx][sample_idx]
            # 根据mode对label进行分词
            if mode == 'word':
                label = ' '.join(label)
            elif mode == 'jieba':
                label = ' '.join(jieba.cut(label))

            # 将分词后的label添加进ref_list
            ref_list[ref_idx].append(label)

            # 调用nlgeval库得到对应score
            try:
                score = n.compute_individual_metrics(hyp=cap, ref=[label])
            except:
                logger.info(f'nlgeval在计算第{sample_idx}个hyp样本，第{ref_idx}个ref时出现错误，跳过该hyp-ref样本(以上计数均从0开始)')
                continue
            unit_bleu1 += score['Bleu_1']
            unit_bleu2 += score['Bleu_2']
            unit_bleu3 += score['Bleu_3']
            unit_bleu4 += score['Bleu_4']
            # unit_meteor += score['METEOR']
            unit_rouge += score['ROUGE_L']

        # 将unit对应的指标在参考数量上取平均
        unit_bleu1 = unit_bleu1 / len(ref)
        unit_bleu2 = unit_bleu2 / len(ref)
        unit_bleu3 = unit_bleu3 / len(ref)
        unit_bleu4 = unit_bleu4 / len(ref)
        # unit_meteor = unit_meteor / len(ref)
        unit_rouge = unit_rouge / len(ref)
        # 将global对应的指标加上unit对应的指标
        global_bleu1 += unit_bleu1
        global_bleu2 += unit_bleu2
        global_bleu3 += unit_bleu3
        global_bleu4 += unit_bleu4
        # global_meteor += unit_meteor
        
        global_rouge += unit_rouge

    # 对所有global指标在hyp数量上取平均
    global_bleu1 = global_bleu1 / len(hyp)
    global_bleu2 = global_bleu2 / len(hyp)
    global_bleu3 = global_bleu3 / len(hyp)
    global_bleu4 = global_bleu4 / len(hyp)
    # global_meteor = global_meteor / len(hyp)
    global_rouge = global_rouge / len(hyp)

    # 最后单独计算CIDEr分数
    try:
        score4CIDEr = n.compute_metrics(ref_list=ref_list, hyp_list=hyp_list)
    except:
        raise SyntaxError('在最终计算CIDEr分数的compute_metrics函数调用时发生错误')
    global_cider = score4CIDEr['CIDEr']

    # 计算Dist指标
    dist1 = distinct_n_sentence_level(all_caps, 1)
    dist2 = distinct_n_sentence_level(all_caps, 2)
    dist3 = distinct_n_sentence_level(all_caps, 3)

    # 将指标添加到result字典中
    result['Bleu_1'] = global_bleu1
    result['Bleu_2'] = global_bleu2
    result['Bleu_3'] = global_bleu3
    result['Bleu_4'] = global_bleu4
    # result['METEOR'] = global_meteor
    result['CIDEr'] = global_cider
    result['ROUGE_L'] = global_rouge
    result['Dist_1'] = dist1
    result['Dist_2'] = dist2
    result['Dist_3'] = dist3

    
    # 文件写入
    write_json(result=result, outfile=outfile)
    # 最终返回result字典
    return result

# 使用nltk库计算中文评价指标
def cal_nltk_metrics(hyp, ref, mode='word', use_smooth=False, outfile=None, cal_bert_score=False):
    # hyp为预测文本列表
    # ref为参考文本列表，len(ref)代表每个hyp有多少条参考文本
    # 定义result集合存储最终指标
    result = {}

    # =============== BERT SCORE ================ #
    precisions, recalls, f1 = 0, 0, 0
    if cal_bert_score == True:
        for idx in range(len(ref)):
            unit_p, unit_r, unit_f1 = score(cands=hyp, refs=ref[idx], model_type='bert-base-chinese', lang='zh', verbose=True)
            precisions += unit_p.mean()
            recalls += unit_r.mean()
            f1 += unit_f1.mean()
        precisions = round(precisions / len(ref), 3)
        recalls = round(recalls / len(ref), 3)
        f1 = round(f1 / len(ref), 3)

    # 根据cal_bert_score计算BertScore
    result['Precisions'] = precisions
    result['Recalls'] = recalls
    result['F1'] = f1

    # ============================================ #

    # 如果use_smooth=True，则初始化smooth
    if use_smooth == True:
        smooth = SmoothingFunction()
    else:
        smooth = None

    # 定义ROUGE
    rouge = Rouge()
    # 定义all_caps，用来计算dist指标
    all_caps = []
    
    # nltk库只记录全局指标
    cnt = 0 
    bleu1 = 0.
    bleu2 = 0.
    bleu3 = 0.
    bleu4 = 0.
    rouge1 = 0.
    rouge2 = 0.
    rougeL = 0.
    meteor =0.
    nist1 = 0.
    nist2 = 0.
    nist3 = 0.
    nist4 = 0.
    
    # 遍历所有hyp,每条hyp对应多个ref
    for sample_idx in tqdm(range(len(hyp))):
        # 获得该hyp，初始化为cap
        cap = hyp[sample_idx]

        # 根据模式先将cap加入all_caps，再对cap进行分词
        if mode == 'word':
            cap = (' '.join(cap)).split()
            all_caps += cap
        elif mode == 'jieba':
            jieba_cap = jieba.cut(cap)
            cap = (' '.join(jieba_cap)).split()
            all_caps += cap
        # 依次遍历该hyp对应的所有ref
        for ref_idx in range(len(ref)):
            # 依次得到该hyp对应的每个label
            label = ref[ref_idx][sample_idx]
            # 根据mode对label进行分词
            if mode == 'word':
                label = (' '.join(label)).split()
            elif mode == 'jieba':
                label = (' '.join(jieba.cut(label))).split()
            # min_len为predict和label较小的长度
            min_len = min(len(cap), len(label)) 
            lens = min(min_len, 4)
            if lens == 0:
                continue
            
            # 获取temp_cap和temp_label用于计算ROUGE
            temp_cap = ' '.join(cap)
            temp_label = ' '.join(label)

            try:
                rouge_score = rouge.get_scores(temp_cap, temp_label)
                rouge1 += rouge_score[0]["rouge-1"]['r']
                rouge2 += rouge_score[0]["rouge-2"]['r']
                rougeL += rouge_score[0]["rouge-l"]['f']
            
                label = [label]
                # 如果使用平滑函数计算bleu
                if use_smooth:
                    if lens >= 1:
                        bleu1 += sentence_bleu(label, cap, weights=(1, 0, 0, 0), smoothing_function=smooth.method1)
                        nist1 += sentence_nist(label, cap, 1)
                    if lens >= 2:
                        bleu2 += sentence_bleu(label, cap, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth.method1)
                        nist2 += sentence_nist(label, cap, 2)
                    if lens >= 3:
                        bleu3 += sentence_bleu(label, cap, weights=(0.333, 0.333, 0.333, 0), smoothing_function=smooth.method1)
                        nist3 += sentence_nist(label, cap, 3)
                    if lens >= 4:
                        bleu4 += sentence_bleu(label, cap, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth.method1)
                        nist4 += sentence_nist(label, cap, 4)
                # 如果不使用平滑函数计算bleu
                else:
                    if lens >= 1:
                        bleu1 += sentence_bleu(label, cap, weights=(1, 0, 0, 0))
                        nist1 += sentence_nist(label, cap, 1)
                    if lens >= 2:
                        bleu2 += sentence_bleu(label, cap, weights=(0.5, 0.5, 0, 0))
                        nist2 += sentence_nist(label, cap, 2)
                    if lens >= 3:
                        bleu3 += sentence_bleu(label, cap, weights=(0.333, 0.333, 0.333, 0))
                        nist3 += sentence_nist(label, cap, 3)
                    if lens >= 4:
                        bleu4 += sentence_bleu(label, cap, weights=(0.25, 0.25, 0.25, 0.25))
                        nist4 += sentence_nist(label, cap, 4)
         
                # 总计数cnt+1
                cnt += 1
            except:
                logger.info(f'nltk在计算第{sample_idx}个hyp样本，第{ref_idx}个ref时出现错误，跳过该hyp-ref样本(以上计数均从0开始)')
                continue

    # 计算Dist指标
    dist1 = distinct_n_sentence_level(all_caps, 1)
    dist2 = distinct_n_sentence_level(all_caps, 2)
    dist3 = distinct_n_sentence_level(all_caps, 3)
    # 所有指标在cnt(总样本数量上取平均)
    bleu1 /= cnt
    bleu2 /= cnt
    bleu3 /= cnt
    bleu4 /= cnt
    rouge1 /= cnt
    rouge2 /= cnt
    rougeL /= cnt
    # meteor /= cnt
    nist1 /= cnt
    nist2 /= cnt
    nist3 /= cnt
    nist4 /= cnt
    # 将最终指标加入result字典
    result['Bleu_1'] = bleu1
    result['Bleu_2'] = bleu2
    result['Bleu_3'] = bleu3
    result['Bleu_4'] = bleu4
    result['ROUGE_L'] =rougeL
    result['METEOR'] = 0 # 暂时为0
    result['CIDEr'] = 0 # 暂时为0
    result['Nist_1'] = nist1
    result['Nist_2'] = nist2
    result['Nist_3'] = nist3
    result['Nist_4'] = nist4
    result['Dist1'] = dist1
    result['Dist2'] = dist2
    result['Dist3'] = dist3

    # 文件写入
    write_json(result=result, outfile=outfile)
    # 返回result字典
    return result


# 不同metric汇总函数
# 前面参数均不变，将outfile变为outdir，指定一个文件夹，在其中生成不同设定的评估尺度
# mode用于指定当前指标是用于评估测试集还是训练集
# 最终评估文件存储路径在outdir的mode文件夹下
def cal_multi_metrics(hyp, ref, n=None, outdir=None, mode='test_metrcis', cal_bert_score=False):
    # 处理outdir
    if outdir is None:
        outdir_name = './metric_result'
        os.mkdir(outdir_name)
    else:
        outdir = os.path.join(outdir, mode)
        if os.path.exists(outdir):
            assert os.path.isdir(outdir), '指定路径须为文件夹路径'
        else:
            os.mkdir(outdir)

    # nlgeval-word-level
    nlg_word_name = os.path.join(outdir, 'nlg_word_level.json')
    nlg_word_result = cal_nlgeval_metrics(hyp=hyp, ref=ref, mode='word', outfile=nlg_word_name)
    # nlgeval-jieba-level
    nlg_jieba_name = os.path.join(outdir, 'nlg_jieba_level.json')
    nlg_jieba_result = cal_nlgeval_metrics(hyp=hyp, ref=ref, mode='jieba',outfile=nlg_jieba_name)
    # nltk-word-level
    nltk_word_name = os.path.join(outdir, 'nltk_word_level.json')
    nltk_word_result = cal_nltk_metrics(hyp=hyp, ref=ref, mode='word', use_smooth=False, outfile=nltk_word_name, cal_bert_score=cal_bert_score)
    # nltk-jieba-level
    nltk_jieba_name = os.path.join(outdir, 'nltk_jieba_level.json')
    nltk_jieba_result = cal_nltk_metrics(hyp=hyp, ref=ref, mode='jieba', use_smooth=False, outfile=nltk_jieba_name)
    # nltk-word_smooth-level
    nltk_word_sm_name = os.path.join(outdir, 'nltk_word_sm_level.json')
    nltk_word_sm_result = cal_nltk_metrics(hyp=hyp, ref=ref,mode='word', use_smooth=True, outfile=nltk_word_sm_name)
    # nltk-jieba-smooth-level
    nltk_jieba_sm_name = os.path.join(outdir, 'nltk_jieba_sm_level.json')
    nltk_jieba_sm_result = cal_nltk_metrics(hyp=hyp, ref=ref,mode='jieba', use_smooth=True, outfile=nltk_jieba_sm_name)
    
    logger.info('=========nlgeval-word-level============')
    logger.info(nlg_word_result)
    logger.info('=======================================')
    logger.info('=========nlgeval-jieba-level============')
    logger.info(nlg_jieba_result)
    logger.info('=======================================')
    logger.info('=========nltk-word-level============')
    logger.info(nltk_word_result)
    logger.info('=======================================')
    logger.info('=========nltk-jieba-level============')
    logger.info(nltk_jieba_result)
    logger.info('=======================================')
    logger.info('=========nltk-word-smooth-level============')
    logger.info(nltk_word_sm_result)
    logger.info('=======================================')
    logger.info('=========nltk-jieba-smooth-level============')
    logger.info(nltk_jieba_sm_result)
    logger.info('=======================================')
    
    return nltk_word_result



