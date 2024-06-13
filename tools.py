# tfshen 2024.2.14
# 一些可复用的工具函数
import os
import re
import json
import torch
import logging
import random
import numpy as np
import matplotlib.pyplot as plt
from cal_metrics import cal_multi_metrics
from nlgeval import NLGEval

# 定义全局对话文件路径
dialog_path = '/home/tfshen/pyproject/moe/true_data'

def create_logger(log_path):
    """
    将日志输出到日志文件和控制台
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(
        filename=log_path, mode='w')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger

# 用于加载json数据
def load_json(data_path):
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
            return data
    except:
        raise FileNotFoundError(f'Fail to load the data file {data_path}, please check if the data file exists')


def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

# 移除对话文本中的表情
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

# 用于清洗文本
def text_process(text):
    # clean the text data
    text = text.strip().lower()
    text = remove_emoji(text)
    text = re.sub(r'(?:@|\（via|\（来源)\S+', '', text)
    text = re.sub(r'[()（）ノ┻━┻Д.。‘’…]', '', text)
    text = re.sub(r'\.{2,}', '', text)
    text = re.sub(r'\s+', '', text)
    return text.strip()


# 指定对话文件，获取未分词的ground-truth列表
# mode指定获取dev_labels还是test_labels
def get_labels(dialog_path=dialog_path, mode='test'):
    assert mode in ['dev', 'test'], 'mode需设定为dev或test'
    dialog_file = os.path.join(dialog_path, f'{mode}.json')
    data = load_json(dialog_file)
    # 获得所有对话id的列表
    dialog_ids = list(data.keys())
    # 创建存储ground-truth的二维列表
    gth_response = [[] for i in range(5)]
    # 遍历对话id列表，依次获得每个对话的ground-truth
    for dialog_id in dialog_ids:
        metadata = data[dialog_id]
        # 得到回复列表
        response_list = metadata['response'] 
        # 取回复列表前5项作为ground-truth
        response_list = response_list[:5]
        # 如果回复列表不足5个，进行复制
        if len(response_list) < 5:
            response_list = [response_list[0] for i in range(5)]
        # 依次将每个ground-truth存入gth_response
        for idx in range(5):
            gth_response[idx].append(text_process(response_list[idx]))
    # 返回ground-truth list
    return gth_response

# 将一组数据写入到TSV文件中
def tsv_writer(values, tsv_file_name, sep='\t'):
    if not os.path.exists(os.path.dirname(tsv_file_name)):
        os.mkdir(os.path.dirname(tsv_file_name))
    tsv_file_name_tmp = tsv_file_name + '.tmp'
    with open(tsv_file_name_tmp, 'wb') as fp:
        assert values is not None
        for value in values:
            assert value is not None
            v = sep.join(map(lambda v: v.decode() if type(v) == bytes else str(v), value)) + '\n'
            v = v.encode()
            fp.write(v)
    os.rename(tsv_file_name_tmp, tsv_file_name)


def txt_writer(caps, predict_file):
    with open(predict_file, "w", encoding="utf-8") as f:
        for element in caps:
            f.write(element + "\n")

def evaluate_on_tiktalk(res_file, gth_list, outdir=None, mode='test_metrics'):
    if res_file.endswith('.tsv'):
        results = []
        # fp为预测结果的文件，fr为要写入结果的文件
        with open(res_file,'rb') as fp, open(res_file + '.txt','wb') as fr:
            for line in fp:
                cap = eval(line.decode())['response'] # 读取生成的描述, 如'我 是 的'
                fr.write((cap+'\n').encode())
                cap_ = cap.replace(' ', '')
                results.append(cap_) 
        pred = results # 验证/测试集中所有生成描述的列表

    elif res_file.endswith('.txt'):
        results = []
        with open(res_file, 'r') as f:
            for line in f:
                cap = line.strip()
                cap = cap.replace(' ', '')
                results.append(cap)
        pred = results 

    print('start caculating metrics ...')
    n = NLGEval(metrics_to_omit=['METEOR', 'SPICE', 'SkipThoughtCS','EmbeddingAverageCosineSimilarity','VectorExtremaCosineSimilarity','GreedyMatchingScore'])
    result = cal_multi_metrics(hyp=pred, ref=gth_list, n=n, outdir=outdir, mode=mode)
    return result

def save_loss_curve(loss_list, step_list, save_dir):
    fig, ax = plt.subplots()
    # 绘制损失曲线
    ax.plot(step_list, loss_list, label='Loss', color='blue')
    # 设置图像标题和标签
    ax.set_title('Loss Curve')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    # 添加图例
    ax.legend()
    # 保存图形
    plt.savefig(os.path.join(save_dir, 'loss.png'))
    # 关闭画布
    plt.close()


