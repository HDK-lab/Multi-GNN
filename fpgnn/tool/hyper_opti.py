from argparse import ArgumentParser, Namespace
from hyperopt import fmin, tpe, hp
import numpy as np
import os
import torch
from copy import deepcopy
from fpgnn.tool import set_hyper_argument, set_log
from train_main import training
# 记录开始时间
import time

start_time = time.time()
space = {
    'hidden_size': hp.quniform('hidden_size', low=256, high=1024, q=128),
    'dropout': hp.quniform('dropout', low=0.0, high=0.2, q=0.05),
    'hide_features': hp.quniform('hide_features', low=64, high=1024, q=64),
    'epochs': hp.quniform('epochs', low=20, high=50, q=10)

}


def fn(space):
    search_no = args.search_now
    log_name = 'train' + str(search_no)
    log = set_log(log_name, args.log_path)
    result_path = os.path.join(args.log_path, 'hyper_para_result.txt')
    list = ['hidden_size', 'hide_features','batch_size']  # , 'nhid', 'nheads'
    for one in list:
        space[one] = int(space[one])
    hyperp = deepcopy(args)
    name_list = []
    change_args = []
    for key, value in space.items():
        name_list.append(str(key))
        name_list.append('-')
        name_list.append((str(value))[:5])
        name_list.append('-')
        setattr(hyperp, key, value)
    dir_name = "".join(name_list)
    dir_name = dir_name[:-1]
    # 确保保存模型的目录存在
    hyperp.save_path = os.path.join(hyperp.save_path, args.save_model_name, dir_name)
    # hyperp.save_path = os.path.join(hyperp.save_path, dir_name)
    os.makedirs(hyperp.save_path, exist_ok=True)  # 如果目录不存在，则创建它

    ave, std, _, _, _, _, _, _, _, _, _ = training(hyperp, log)
    if ave is None:
        if hyperp.dataset_type == 'classification':
            ave = 0
        else:
            raise ValueError('Result of model is error.')
    args.search_now += 1

    if hyperp.dataset_type == 'classification':
        return -ave
    else:
        return ave


def hyper_searching(args):
    result_path = os.path.join(args.log_path, 'hyper_para_result.txt')
    result = fmin(fn, space, tpe.suggest, args.search_num)
    print('result:',result)
    print('space:',space)
    with open(result_path, 'a+') as file:
        file.write(r'{}_Best Hyperparameters : '.format(args.save_model_name))
        file.write(str(result) + '\n')


if __name__ == '__main__':
    args = set_hyper_argument()
    for seed in [24]:
        for task_name in ['HCAR1_13086_1']:
            args.ablation_mode = 1
            args.cuda = torch.cuda.is_available()
            os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
            args.data_path = "C:\Project\code\MGCR_SH\\fpgnn\dataset\sirt_s\{}.csv".format(task_name)
            args.log_path = 'log'
            args.num_folds = 1
            args.search_num = 30

            args.save_model_name = r'{}_hyper'.format(task_name)
            args.seed = seed
            if task_name in ['bbbp', 'bace','HCT116','HCAR1_','HT_29','HCAR1_13086_1','A549_6382', 'H460','CD73', 'A549']:
                args.dataset_type = 'classification'
                args.is_multitask = 0
                args.task_num = 1
                args.metric = 'auc'  # ['auc','f1','recall','precision','acc']
            elif task_name in ['esol', 'freesolv_1', 'lipo']:
                args.dataset_type = 'regression'
                args.is_multitask = 0
                args.task_num = 1
                args.metric = 'rmse'
            elif task_name == 'qm7':
                args.dataset_type = 'regression'
                args.is_multitask = 0
                args.task_num = 1
                args.metric = 'mae'
            elif task_name == 'clintox':
                args.dataset_type = 'classification'
                args.is_multitask = 1
                args.task_num = 2
                args.metric = 'auc'
            elif task_name == 'sider':
                args.dataset_type = 'classification'
                args.is_multitask = 1
                args.task_num = 27
                args.metric = 'auc'
            elif task_name == 'tox21':
                args.dataset_type = 'classification'
                args.is_multitask = 1
                args.task_num = 12
                args.metric = 'auc'
            elif task_name == 'qm8':
                args.dataset_type = 'regression'
                args.is_multitask = 1
                args.task_num = 12
                args.metric = 'mae'
            elif task_name == 'ce':
                args.dataset_type = 'classification'
                args.is_multitask = 0
                args.task_num = 1
                args.metric = 'prc-auc'
            hyper_searching(args)
# 记录结束时间
end_time = time.time()
elapsed_time_seconds = end_time - start_time
elapsed_time_minutes = elapsed_time_seconds / 60
print(f"运行时间：{elapsed_time_minutes:.6f} 分钟")
