from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')
import torch
import numpy as np
import os
import pandas as pd
from fpgnn.train import fold_train
from fpgnn.tool import set_log, set_train_argument, get_task_name, mkdir
# 记录开始时间
import time

start_time = time.time()
local_time = time.localtime(start_time)
formatted_time = time.strftime('%Y-%m-%d %H:%M', local_time)
print("当前时间是：", formatted_time)

os.environ['CUDA_LAUNCH_BLOCKING'] = "0"


def training(args, log):
    args.cuda = False
    info = log.info
    seed_first = args.seed
    data_path = args.data_path
    save_path = args.save_path

    score = []
    all_results = []
    for num_fold in range(args.num_folds):
        args.seed = seed_first + num_fold
        fold_score, results = fold_train(args, log)
        all_results.append(results)
        score.append(fold_score)
    score = np.array(score)
    all_results = np.array(all_results)

    info(f'Running {args.num_folds} folds in total.')
    if args.num_folds > 1:
        for num_fold, fold_score in enumerate(score):
            info(f'Seed {seed_first + num_fold} : test {args.metric} = {np.nanmean(fold_score):.6f}')

    # 计算AUC的均值和标准差
    auc_ave = np.nanmean(score)
    auc_std = np.nanstd(score)
    best_fold_index = np.nanargmax(score)
    best_seed = seed_first + best_fold_index
    info(f'Average test {args.metric} = {auc_ave:.6f} +/- {auc_std:.6f}')

    acc_values = np.array([fold_results['acc'] for fold_results in all_results])
    precision_values = np.array([fold_results['precision'] for fold_results in all_results])
    recall_values = np.array([fold_results['recall'] for fold_results in all_results])
    f1_values = np.array([fold_results['f1'] for fold_results in all_results])

    # 计算各指标的均值和标准差
    acc_mean = np.nanmean(acc_values)
    acc_std = np.nanstd(acc_values)
    precision_mean = np.nanmean(precision_values)
    precision_std = np.nanstd(precision_values)
    recall_mean = np.nanmean(recall_values)
    recall_std = np.nanstd(recall_values)
    f1_mean = np.nanmean(f1_values)
    f1_std = np.nanstd(f1_values)

    # 打印结果
    info(f'Average Accuracy = {acc_mean:.6f} +/- {acc_std:.6f}')
    info(f'Average Precision = {precision_mean:.6f} +/- {precision_std:.6f}')
    info(f'Average Recall = {recall_mean:.6f} +/- {recall_std:.6f}')
    info(f'Average F1-Score = {f1_mean:.6f} +/- {f1_std:.6f}')

    # 直接返回具体值（单任务场景）
    return auc_ave, auc_std, acc_mean, acc_std, precision_mean, precision_std, recall_mean, recall_std, f1_mean, f1_std, best_seed

if __name__ == '__main__':

    for seed in [7]:
        args = set_train_argument()
        args.cuda = torch.cuda.is_available()
        args.ablation_mode = 1
        for task_name in ['HCAR1_random']:
            #args.split_type = 'bm_scaffold'
            args.data_path = r"C:\Project\code\MGCR_SH\fpgnn/dataset/sirt_s/HCAR1_13384.csv"
            args.save_train_data_path = r'C:\Project\code\MGCR_SH\fpgnn/dataset/data_split/train'
            args.save_val_data_path = r'C:\Project\code\MGCR_SH\fpgnn\dataset/data_split/val'
            args.save_test_data_path = r'C:\Project\code\MGCR_SH\fpgnn/dataset/data_split/test'
            args.log_path = 'log'
            args.save_model_name = task_name
            args.seed = seed
            args.dataset_type = 'classification'
            args.is_multitask = 0
            args.task_num = 1
            args.metric = 'auc'
            log = set_log('train', args.log_path)
            auc_ave, auc_std, acc_mean, acc_std, precision_mean, precision_std, recall_mean, recall_std, f1_mean, f1_std, best_seed = training(
                args, log)
            # 四舍五入到小数点后四位
            auc_ave = round(auc_ave, 3)
            auc_std = round(auc_std, 3)
            acc_mean = round(acc_mean, 3)
            acc_std = round(acc_std, 3)
            precision_mean = round(precision_mean, 3)
            precision_std = round(precision_std, 3)
            recall_mean = round(recall_mean, 3)
            recall_std = round(recall_std, 3)
            f1_mean = round(f1_mean, 3)
            f1_std = round(f1_std, 3)
            file_exists = os.path.exists(args.save_path_result)
            # 修改结果保存的代码
            clm = ['model ', 'auc_ave', 'auc_std', 'acc_mean', 'acc_std', 'precision_mean', 'precision_std',
                   'recall_mean',
                   'recall_std', 'f1_mean', 'f1_std', 'best_seed']
            data = [
                [args.save_model_name, auc_ave, auc_std, acc_mean, acc_std, precision_mean, precision_std, recall_mean,
                 recall_std, f1_mean,
                 f1_std, best_seed]]
            df = pd.DataFrame(data, columns=clm)
            if not file_exists:
                df.to_csv(args.save_path_result, mode="w", header=True, index=False)
            else:
                df.to_csv(args.save_path_result, mode="a+", header=False, index=False)

# 记录结束时间
end_time = time.time()
elapsed_time_seconds = end_time - start_time
elapsed_time_minutes = elapsed_time_seconds / 60
print(f"运行时间：{elapsed_time_minutes:.6f} 分钟")
