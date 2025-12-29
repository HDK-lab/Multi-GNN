import os
import csv
import logging
import math
import sys
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import random
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from sklearn.metrics import auc, mean_squared_error, precision_recall_curve, roc_auc_score, r2_score, \
    mean_absolute_error,f1_score,recall_score,precision_score,accuracy_score,mean_absolute_percentage_error
from fpgnn.data import MoleDataSet, MoleData
from fpgnn.model import FPFG
import torch as th


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def mkdir(path,isdir = True):
    if isdir == False:
        path = os.path.dirname(path)
    if path != '':
        os.makedirs(path, exist_ok = True)

def set_log(name,save_path):

    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)

    log.handlers.clear()
    log_stream = logging.StreamHandler(stream=sys.stdout)
    log_stream.setLevel(logging.DEBUG)
    log.addHandler(log_stream)
    
    mkdir(save_path)
    
    log_file_d = logging.FileHandler(os.path.join(save_path, 'debug.log'))
    log_file_d.setLevel(logging.DEBUG)
    log.addHandler(log_file_d)
    
    return log

def get_header(path):
    with open(path) as file:
        header = next(csv.reader(file))
    
    return header

def get_task_name(path):
    task_name = get_header(path)[1:]
    
    return task_name

def load_data(path,args):
    with open(path) as file:
        reader = csv.reader(file)
        next(reader)
        lines = []
        for line in reader:
            lines.append(line)
        data = []
        for line in lines:
            one = MoleData(line,args)
            data.append(one)
        data = MoleDataSet(data)
        
        fir_data_len = len(data)
        data_val = []
        smi_exist = []
        for i in range(fir_data_len):
            if data[i].mol is not None:
                smi_exist.append(i)
        data_val = MoleDataSet([data[i] for i in smi_exist])
        now_data_len = len(data_val)
        print('There are ',now_data_len,' smiles in total.')
        if fir_data_len - now_data_len > 0:
            print('There are ',fir_data_len , ' smiles first, but ',fir_data_len - now_data_len, ' smiles is invalid.  ')
        
    return data_val


def split_data(data, split_type, size, seed, log=None):
    """
    数据集划分：支持random/bm_scaffold

    Args:
        data: MoleDataSet对象
        split_type: 划分类型（random/bm_scaffold）
        size: [train, val, test]比例列表
        seed: 随机种子
        log: 日志对象（可选）

    Returns:
        train_data, val_data, test_data: MoleDataSet对象
    """
    assert len(size) == 3, "Size must have 3 elements (train/val/test)"
    assert sum(size) == 1, "Sum of size must be 1"
    random.seed(seed)

    # 提取MoleData的SMILES（适配MoleData结构）
    def get_smiles(mole_data):
        return mole_data.smile

    if split_type == 'random':
        # 随机打乱数据
        shuffled_list = [d for d in data]
        random.shuffle(shuffled_list)

        # 计算划分索引
        train_size = int(size[0] * len(shuffled_list))
        val_size = int(size[1] * len(shuffled_list))
        train_val_size = train_size + val_size

        # 划分数据集
        train_data = MoleDataSet(shuffled_list[:train_size])
        val_data = MoleDataSet(shuffled_list[train_size:train_val_size])
        test_data = MoleDataSet(shuffled_list[train_val_size:])

        if log:
            log.info(f"Random split result: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        else:
            print(f"Random split result: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        return train_data, val_data, test_data

    elif split_type == 'bm_scaffold':
        if log:
            log.info("Splitting dataset by Bemis-Murcko scaffold...")
        else:
            print("Splitting dataset by Bemis-Murcko scaffold...")

        scaffold_to_indices = defaultdict(list)
        invalid_scaffold = 0

        for idx, mole in enumerate(data):
            smiles = get_smiles(mole)  # 现在调用会返回mole.smile
            try:
                mol = mole.mol
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
                scaffold_to_indices[scaffold].append(idx)
            except Exception as e:
                invalid_scaffold += 1
                if log:
                    # 同样改为smile
                    log.warning(f"Failed to get scaffold for SMILES {mole.smile}: {e}")
                else:
                    print(f"Warning: Failed to get scaffold for SMILES {mole.smile}: {e}")

        if invalid_scaffold > 0:
            if log:
                log.warning(f"Skipped {invalid_scaffold} molecules due to scaffold calculation failure")
            else:
                print(f"Skipped {invalid_scaffold} molecules due to scaffold calculation failure")

        # 2. 打乱骨架顺序，按比例划分骨架组
        scaffolds = list(scaffold_to_indices.keys())
        random.shuffle(scaffolds)

        total_valid = len(data) - invalid_scaffold
        train_cutoff = size[0] * total_valid
        val_cutoff = (size[0] + size[1]) * total_valid

        train_indices = []
        val_indices = []
        test_indices = []
        current_count = 0

        for scaffold in scaffolds:
            indices = scaffold_to_indices[scaffold]
            if current_count + len(indices) <= train_cutoff:
                train_indices.extend(indices)
            elif current_count + len(indices) <= val_cutoff:
                val_indices.extend(indices)
            else:
                test_indices.extend(indices)
            current_count += len(indices)

        # 3. 根据索引提取数据
        all_data_list = [d for d in data]
        train_data = MoleDataSet([all_data_list[idx] for idx in train_indices])
        val_data = MoleDataSet([all_data_list[idx] for idx in val_indices])
        test_data = MoleDataSet([all_data_list[idx] for idx in test_indices])

        # 输出划分结果
        if log:
            log.info(f"Scaffold split result: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        else:
            print(f"Scaffold split result: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

        return train_data, val_data, test_data

    else:
        raise ValueError(f"Unsupported split type: {split_type}. Use 'random' or 'bm_scaffold'.")


# 为什么
def get_label_scaler(data):
    smile = data.smile()
    label = data.label()
    
    label = np.array(label).astype(float)
    ave = np.nanmean(label,axis=0)
    ave = np.where(np.isnan(ave),np.zeros(ave.shape),ave)  # 条件语句，当第一个值成立时，返回第二个值，否则返回第三个值
    std = np.nanstd(label,axis=0)  # 计算标准差，同时忽略nan
    std = np.where(np.isnan(std),np.ones(std.shape),std)
    std = np.where(std==0,np.ones(std.shape),std)
    
    change_1 = (label-ave)/std  # 计算标准分数
    label_changed = np.where(np.isnan(change_1),None,change_1)
    label_changed.tolist()
    data.change_label(label_changed)
    
    return [ave,std]


def pos_weight(train_set, out_features):  # 不平衡数据处理，正样本权重
    nnn = []
    for i in range(out_features):
        for j in range(len(train_set)):
            nnn.append(train_set[j].label[i])
    mmm = np.array(nnn)
    mmm = mmm.reshape(-1, len(train_set))
    task_pos_weight_list = []
    for i in range(out_features):
        labels = []
        for j in range(len(mmm[i])):
            labels.append(mmm[i][j])
        labels = np.array(labels)
        num_pos = 0
        num_neg = 0
        for i in labels:
            if i == 1:
                num_pos = num_pos + 1
            if i == 0:
                num_neg = num_neg + 1
        weight = num_neg / (num_pos + 0.00000001)
        task_pos_weight_list.append(weight)
    task_pos_weight = th.tensor(task_pos_weight_list)
    # print(task_pos_weight)
    return task_pos_weight


def get_loss(type, train_data, task_name):
    if type == 'classification':
        # pos_weight_np = pos_weight(train_data, task_name)
        return nn.BCEWithLogitsLoss(reduction='none')  # , pos_weight=pos_weight_np.to(device))
    elif type == 'regression':
        return nn.MSELoss(reduction='none')
    else:
        raise ValueError('Data type Error.')

def prc_auc(label,pred):
    prec, recall, _ = precision_recall_curve(label,pred)
    result = auc(recall,prec)
    return result
def rmse(label,pred):
    result = mean_squared_error(label,pred)
    return math.sqrt(result)
def mre(label,pred):
    mape = mean_absolute_percentage_error(label, pred)
    result=mape / 100
    return result
def r2(label,pred):
    result = r2_score(label, pred)
    return result

def mae(label,pred):
    result = mean_absolute_error(label, pred)
    return result
def f1(label,pred):
    pred = soft(pred)
    label = soft(label)
    result = f1_score(label,pred)
    return result
def recall(label, pred):
    pred = soft(pred)
    label = soft(label)
    result = recall_score(label,pred)
    return result
def precision(label,pred):
    pred = soft(pred)
    label = soft(label)
    result = precision_score(label,pred)
    return result
def acc(label,pred):
    pred = soft(pred)
    label = soft(label)
    result = accuracy_score(label,pred)
    return result
def soft(pred):
    res = []
    for i in pred:
        if i > 0.5 or i == 0.5:
            res.append(1)
        else:
            res.append(0)
    return res
def get_metric(metric):
    if metric == 'auc':
        return roc_auc_score
    elif metric == 'prc-auc':
        return prc_auc
    elif metric == 'rmse':
        return rmse
    elif metric == 'r2':
        return r2
    elif metric == 'mae':
        return mae
    elif metric == 'mre':
        return mre
    elif metric == 'f1':
        return f1
    elif metric == 'recall':
        return recall
    elif metric == 'precision':
        return precision
    elif metric == 'acc':
        return acc
    else:
        raise ValueError('Metric Error.')

def save_model(path,model,scaler,args):
    if scaler != None:
        state = {
            'args':args,
            'state_dict':model.state_dict(),
            'data_scaler':{
                'means':scaler[0],
                'stds':scaler[1]
            }
        }
    else:
        state = {
            'args':args,
            'state_dict':model.state_dict(),
            'data_scaler':None
            }
    torch.save(state,path)

# def load_model(path,cuda,log=None,pred_args=None):
#     if log is not None:
#         debug = log.debug
#     else:
#         debug = print
#
#     state = torch.load(path,map_location=lambda storage, loc: storage)
#     args = state['args']
#
#     if pred_args is not None:
#         for key,value in vars(pred_args).items():
#             if not hasattr(args,key):
#                 setattr(args, key, value)
#
#     state_dict = state['state_dict']
#
#     # model = FPGNN(args)
#     model = FPFG(args)
#     model_state_dict = model.state_dict()
#
#     load_state_dict = {}
#     for param in state_dict.keys():
#         if param not in model_state_dict:
#             debug(f'Parameter is not found: {param}.')
#         elif model_state_dict[param].shape != state_dict[param].shape:
#             debug(f'Shape of parameter is error: {param}.')
#         else:
#             load_state_dict[param] = state_dict[param]
#             debug(f'Load parameter: {param}.')
#
#     model_state_dict.update(load_state_dict)
#     model.load_state_dict(model_state_dict)
#
#     if cuda:
#         model = model.to(torch.device("cuda"))
#
#     return model
import argparse
import torch
import os
from torch.version import __version__ as torch_version


def load_model(path, cuda, log=None, pred_args=None):
    if log is not None:
        debug = log.debug
    else:
        debug = print

    # -------------------------- 新增：修复 PyTorch 2.6+ 安全加载问题 --------------------------
    # 1. 允许加载 argparse.Namespace 类型（安全白名单）
    torch.serialization.add_safe_globals([argparse.Namespace])

    # 2. 处理 PyTorch 版本兼容性：低于 2.6 无 weights_only 参数
    torch_ver_tuple = tuple(map(int, torch_version.split('.')[:2]))
    load_kwargs = {}
    if torch_ver_tuple >= (2, 6):
        load_kwargs['weights_only'] = True  # 优先安全模式

    # 3. 尝试加载模型（安全模式 -> 兼容模式降级）
    try:
        state = torch.load(path, map_location=lambda storage, loc: storage, **load_kwargs)
        debug(
            f"模型加载成功（PyTorch {torch_version}，模式：{'安全模式' if load_kwargs.get('weights_only') else '兼容模式'}）")
    except Exception as e:
        debug(f"安全模式加载失败：{str(e)}")
        if torch_ver_tuple >= (2, 6):
            load_kwargs['weights_only'] = False  # 降级为兼容模式（仅信任模型来源时）
            state = torch.load(path, map_location=lambda storage, loc: storage, **load_kwargs)
            debug(f"已切换为兼容模式加载模型，请确保模型文件来源可信！")
    # ----------------------------------------------------------------------------------------

    args = state['args']

    if pred_args is not None:
        for key, value in vars(pred_args).items():
            if not hasattr(args, key):
                setattr(args, key, value)

    state_dict = state['state_dict']


    model = FPFG(args)
    model_state_dict = model.state_dict()

    load_state_dict = {}
    for param in state_dict.keys():
        if param not in model_state_dict:
            debug(f'Parameter is not found: {param}.')
        elif model_state_dict[param].shape != state_dict[param].shape:
            debug(f'Shape of parameter is error: {param}.')
        else:
            load_state_dict[param] = state_dict[param]
            debug(f'Load parameter: {param}.')

    model_state_dict.update(load_state_dict)
    model.load_state_dict(model_state_dict)

    if cuda and torch.cuda.is_available():
        model = model.to(torch.device("cuda"))
        debug(f"模型已移至 CUDA 设备")
    else:
        debug(f"CUDA 不可用，模型保留在 CPU 设备")

    return model
def get_scaler(path):
    import torch
    import argparse  # 必须导入argparse

    # 方案1：添加安全全局对象（推荐，无安全风险）
    torch.serialization.add_safe_globals([argparse.Namespace])
    state = torch.load(path, map_location=lambda storage, loc: storage)
    if state['data_scaler'] is not None:
        ave = state['data_scaler']['means']
        std = state['data_scaler']['stds']
        return [ave,std]
    else:
        return None

def load_args(path):
    state = torch.load(path, map_location=lambda storage, loc: storage)
    
    return state['args']

def rmse(label,pred):
    result = mean_squared_error(label,pred)
    result = math.sqrt(result)
    return result


"""

Noam learning rate scheduler with piecewise linear increase and exponential decay.

The learning rate increases linearly from init_lr to max_lr over the course of
the first warmup_steps (where warmup_steps = warmup_epochs * steps_per_epoch).
Then the learning rate decreases exponentially from max_lr to final_lr over the
course of the remaining total_steps - warmup_steps (where total_steps =
total_epochs * steps_per_epoch). This is roughly based on the learning rate
schedule from Attention is All You Need, section 5.3 (https://arxiv.org/abs/1706.03762).

"""

class NoamLR(_LRScheduler):
    def __init__(self,optimizer,warmup_epochs,total_epochs,steps_per_epoch,
                 init_lr,max_lr,final_lr):
        assert len(optimizer.param_groups) == len(warmup_epochs) == len(total_epochs) == len(init_lr) == \
               len(max_lr) == len(final_lr)

        self.num_lrs = len(optimizer.param_groups)

        self.optimizer = optimizer
        self.warmup_epochs = np.array(warmup_epochs)
        self.total_epochs = np.array(total_epochs)
        self.steps_per_epoch = steps_per_epoch
        self.init_lr = np.array(init_lr)
        self.max_lr = np.array(max_lr)
        self.final_lr = np.array(final_lr)

        self.current_step = 0
        self.lr = init_lr
        self.warmup_steps = (self.warmup_epochs * self.steps_per_epoch).astype(int)
        self.total_steps = self.total_epochs * self.steps_per_epoch
        self.linear_increment = (self.max_lr - self.init_lr) / self.warmup_steps

        self.exponential_gamma = (self.final_lr / self.max_lr) ** (1 / (self.total_steps - self.warmup_steps))

        super(NoamLR, self).__init__(optimizer)

    def get_lr(self):
        return list(self.lr)

    def step(self,current_step=None):
        if current_step is not None:
            self.current_step = current_step
        else:
            self.current_step += 1

        for i in range(self.num_lrs):
            if self.current_step <= self.warmup_steps[i]:
                self.lr[i] = self.init_lr[i] + self.current_step * self.linear_increment[i]
            elif self.current_step <= self.total_steps[i]:
                self.lr[i] = self.max_lr[i] * (self.exponential_gamma[i] ** (self.current_step - self.warmup_steps[i]))
            else:  
                self.lr[i] = self.final_lr[i]

            self.optimizer.param_groups[i]['lr'] = self.lr[i]
