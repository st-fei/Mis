import os
import argparse

def get_args():
    
    parser = argparse.ArgumentParser(description='arguments of case')

    # 随机数种子
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    # 实验名称
    parser.add_argument('--experiment_name', type=str, default='experiment_1', help='experiment name')
    # 指定GPU
    parser.add_argument('--ddp', action='store_true', help='是否多卡并行')
    parser.add_argument('--gpu_ids', type=int, default=0, help='使用哪块gpu')

    # 模型名称 or 模型路径
    parser.add_argument("--model_name_or_path", default='/home/tfshen/pyproject/moe/chinese_bert', type=str, help="Path to pre-trained model or model type.")
    
    # 检索数量
    parser.add_argument("--retrieve_num", type=int, default=1, help="检索数量")
    # chunk selection
    parser.add_argument("--topk", type=int, default=2, help="视频块检索数量")
    # region selection
    parser.add_argument("--topj", type=int, default=12, help="视频区域检索数量")

    # num_query_tokens
    parser.add_argument("--num_query_tokens", default=32, type=int, help="qformer query tokens")

    # moe_enable
    parser.add_argument("--moe_enable", action="store_true", help="是否启用moe")
    parser.add_argument("--router_jitter_noise", type=float, default=0)
    parser.add_argument("--num_local_experts", type=int, default=3)
    parser.add_argument("--num_experts_per_tok", type=int, default=2)
    parser.add_argument("--router_aux_loss_coef", type=float, default=0.1)

    # retrieve_enable
    parser.add_argument("--retrieve_enable", action="store_true", help="是否启用检索")

    # 训练参数
    parser.add_argument("--max_desc_length", type=int, default=40, help="最大视频描述长度")
    parser.add_argument("--max_his_length", type=int, default=140, help="最大历史文本长度")
    parser.add_argument("--max_resp_length", type=int, default=40, help="最大回复文本长度")
    parser.add_argument("--max_prior_length", type=int, default=100, help="最大先验信息长度")
    parser.add_argument("--max_steps", type=int, default=0, help='是否定义最大优化轮数，此参数>0则覆盖epoch参数')
    parser.add_argument("--epoch", type=int, default=20, help='Training epoches')
    parser.add_argument("--train_batch_size", type=int, default=2, help='训练时的batch_size')
    parser.add_argument("--test_batch_size", type=int, default=1, help="解码时的batch_size")
    parser.add_argument("--num_workers", type=int, default=1, help='num workers of training')
    parser.add_argument("--lr", type=float, default=3e-5, help='learning rate')
    parser.add_argument("--eps", default=1.0e-09, type=float, required=False, help='衰减率')
    parser.add_argument("--weight_decay", type=float, default=0.05, help='weight decay')
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam.")
    parser.add_argument("--warmup", default=0.01, type=float, help="Linear warmup.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="梯度累积步骤")
    
    # 模式
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run evaluation.")
    parser.add_argument('--multi_eval', action='store_true', help='批量评估')

    # 记录参数
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every X steps.")
    parser.add_argument("--plt_steps", type=int, default=1000, help="图表记录轮数")
    parser.add_argument("--save_steps", type=int, default=10000, help="权重保存轮数")
    parser.add_argument("--evaluate_during_training", action='store_true', help="Run evaluation during training at each save_steps.")
    parser.add_argument("--evaluate_testset_during_training", action='store_true', help="Run testset evaluation during training at each save_steps")

    # 生成参数
    parser.add_argument('--do_sample', action='store_true', help="Whether to do sampling.")
    parser.add_argument('--max_gen_len', type=int, default=40, help="最大生成长度")
    parser.add_argument('--num_beams', type=int, default=3, help="beam search width")
    parser.add_argument('--temperature', type=float, default=1.0, help="temperature in softmax for sampling")
    parser.add_argument('--top_k', type=int, default=1, help="filter distribution for sampling")
    parser.add_argument('--top_p', type=float, default=1, help="filter distribution for sampling")
    parser.add_argument('--repetition_penalty', type=float, default=1.2,help="repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)")
    parser.add_argument('--length_penalty', type=float, default=0.3, help="beam search length penalty")
    
    # 路径
    parser.add_argument('--lossfig_path', type=str, default='/home/tfshen/pyproject/case/lossfig', help='loss曲线保存地址')
    parser.add_argument('--text_dir', type=str, default='/home/tfshen/pyproject/case/data/dialogs', help='对话文件地址')
    parser.add_argument('--feat_path', type=str, default='/data/tfshen/video_feats.h5', help='视频特征地址')
    parser.add_argument('--video_dir', type=str, default='/data/tfshen/my_video', help='视频文件地址')
    parser.add_argument('--kg_path', type=str, default="/home/tfshen/pyproject/new_vt_bart/kg_store", help="知识图谱存放位置")
    parser.add_argument('--output_dir', type=str, default='/data/tfshen/case_result', help='结果保存地址')
    parser.add_argument('--xlsx_dir', type=str, default='/home/tfshen/pyproject/case/record', help='表格记录地址')
    parser.add_argument('--train_log_path', type=str, default='/home/tfshen/pyproject/case/train_logs', help='训练日志存放')
    parser.add_argument('--eval_log_path', type=str, default='/home/tfshen/pyproject/case/test_logs', help='测试日志存放')
    parser.add_argument('--eval_model_dir', type=str, default='', help='用于评估的文件夹')
    parser.add_argument('--eval_mode', type=str, choices=["dev", "test"], default="test", help="用于评估验证集or测试集")
    
    args = parser.parse_args()
    
    return args

















