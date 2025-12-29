from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.optim.lr_scheduler import ExponentialLR
from fpgnn.tool.tool import mkdir, get_task_name, load_data, split_data, get_label_scaler, get_loss, get_metric, \
    save_model, NoamLR, load_model
from fpgnn.model import FPFG
from fpgnn.data import MoleDataSet
from sklearn.metrics import accuracy_score
import os
import pandas as pd
import numpy as np
import torch
from torch.optim import Adam

def epoch_train(model, data, loss_f, optimizer, scheduler, args):
    model.train()
    data.random_data(args.seed)
    loss_sum = 0
    device = torch.device('cuda' if args.cuda else 'cpu')
    data_used = 0
    iter_step = args.batch_size

    for i in range(0, len(data), iter_step):
        if data_used + iter_step > len(data):
            break

        data_now = MoleDataSet(data[i:i + iter_step])
        smile = data_now.smile()
        label = data_now.label()

        mask = torch.tensor([[x is not None for x in tb] for tb in label], dtype=torch.float).to(device)
        target = torch.tensor([[0 if x is None else x for x in tb] for tb in label], dtype=torch.float).to(device)
        weight = torch.ones(target.shape).to(device)
        model.to(device)
        model.zero_grad()
        pred, attn = model(smile, 0)
        # 处理 attention
        # tensor = attn["tensor"].detach().cpu().numpy()
        # names = attn["names"]
        #
        # # 如果是多头注意力：需要 reshape 成二维
        # if tensor.ndim > 2:
        #     tensor = tensor.reshape(tensor.shape[0], -1)
        #
        # # 保存
        # df = pd.DataFrame(tensor)
        # if "names" in attn:
        #     df.columns = names
        # df.to_csv(r"C:\Project\code\MGCR_SH\fpgnn\dataset\results\Attn\attn_weights.csv", index=False)

        loss = loss_f(pred, target) * weight * mask
        loss = loss.sum() / mask.sum()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        data_used += len(smile)
        if isinstance(scheduler, NoamLR):
            scheduler.step()

    # 学习率调度
    if isinstance(scheduler, ExponentialLR):
        scheduler.step()

    return loss_sum / (data_used / iter_step)


def predict(model, data, batch_size, scaler):
    model.eval()
    pred = []
    data_total = len(data)

    for i in range(0, data_total, batch_size):
        data_now = MoleDataSet(data[i:i + batch_size])
        smile = data_now.smile()
        with torch.no_grad():
            pred_now,_ = model(smile, 0)###
        pred_now = pred_now.data.cpu().numpy()
        if scaler is not None:
            ave = scaler[0]
            std = scaler[1]
            pred_now = np.array(pred_now).astype(float)
            change_1 = pred_now * std + ave
            pred_now = np.where(np.isnan(change_1), None, change_1)

        pred_now = pred_now.tolist()
        pred.extend(pred_now)
    return pred


def reward_predict(model, smile1, batch_size, scaler):
    model.eval()
    pred = []
    smile = []
    smile.append(smile1)
    with torch.no_grad():
        pred_now,_ = model(smile, 0)
    pred_now = pred_now.data.cpu().numpy()
    if scaler is not None:
        ave = scaler[0]
        std = scaler[1]
        pred_now = np.array(pred_now).astype(float)
        change_1 = pred_now * std + ave
        pred_now = np.where(np.isnan(change_1), None, change_1)

    pred_now = pred_now.tolist()
    pred.extend(pred_now)
    return pred


def sub_predict(model, smile1, batch_size, scaler, num):
    model.eval()
    pred = []

    smile = []
    smile.append(smile1)
    with torch.no_grad():
        pred_now,_ = model(smile, num)
    pred_now = pred_now.data.cpu().numpy()
    if scaler is not None:
        ave = scaler[0]
        std = scaler[1]
        pred_now = np.array(pred_now).astype(float)
        change_1 = pred_now * std + ave
        pred_now = np.where(np.isnan(change_1), None, change_1)

    pred_now = pred_now.tolist()
    pred.extend(pred_now)
    return pred

def compute_score(pred, label, metric_f, args, log):
    info = log.info

    batch_size = args.batch_size
    task_num = args.task_num
    data_type = args.dataset_type

    if len(pred) == 0:
        return [float('nan')] * task_num

    pred_val = []
    label_val = []
    for i in range(task_num):
        pred_val_i = []
        label_val_i = []
        for j in range(len(pred)):
            if label[j][i] is not None:
                pred_val_i.append(pred[j][i])
                label_val_i.append(label[j][i])
        pred_val.append(pred_val_i)
        label_val.append(label_val_i)

    result = []
    resultacc = []
    for i in range(task_num):
        if data_type == 'classification':
            if all(one == 0 for one in label_val[i]) or all(one == 1 for one in label_val[i]):
                info('Warning: All labels are 1 or 0.')
                result.append(float('nan'))
                continue
            if all(one == 0 for one in pred_val[i]) or all(one == 1 for one in pred_val[i]):
                info('Warning: All predictions are 1 or 0.')
                result.append(float('nan'))
                continue
        re = metric_f(label_val[i], pred_val[i])

        result.append(re)

    return result

def compute_score_test(pred, label, metric_f, args, log):
    info = log.info if log else print
    task_num = args.task_num

    test_score = []
    results = []

    for i in range(task_num):
        # 筛选出有效标签和预测值
        pred_val_i = [pred[j][i] for j in range(len(pred)) if label[j][i] is not None]
        label_val_i = [label[j][i] for j in range(len(label)) if label[j][i] is not None]

        if len(pred_val_i) == 0 or len(label_val_i) == 0:
            info(f'Warning: No valid predictions or labels for task {i}.')
            test_score.append(float('nan'))
            results.append({'acc': float('nan'), 'precision': float('nan'), 'recall': float('nan'), 'f1': float('nan')})
            continue

        # 转为 numpy 数组
        pred_val_i = np.array(pred_val_i)
        label_val_i = np.array(label_val_i)

        # 将概率预测转换为二分类标签（如 >0.5 为 1）
        binary_pred = (pred_val_i >= 0.5).astype(int)
        args.average = "binary"
        task_results = {}
        try:
            task_results['acc'] = accuracy_score(label_val_i, binary_pred)
            task_results['precision'] = precision_score(label_val_i, binary_pred, average=args.average, zero_division=0)
            task_results['recall'] = recall_score(label_val_i, binary_pred, average=args.average, zero_division=0)
            task_results['f1'] = f1_score(label_val_i, binary_pred, average=args.average, zero_division=0)
        except Exception as e:
            info(f"Warning: Error calculating classification metrics for task {i}: {e}")
            task_results = {'acc': float('nan'), 'precision': float('nan'), 'recall': float('nan'), 'f1': float('nan')}

        if metric_f == 'auc':
            try:
                auc_score = roc_auc_score(label_val_i, pred_val_i)
            except ValueError as e:
                info(f'Warning: Error calculating AUC for task {i}: {e}')
                auc_score = float('nan')
            test_score.append(auc_score)
        else:
            test_score.append(float('nan'))
        results.append(task_results)

    return test_score, results


def soft(pred):
    res = []
    for i in pred:
        if i > 0.5 or i == 0.5:
            res.append(1)
        else:
            res.append(0)
    return res

def compute_score_test(pred, label, metric_f, args, log):
    info = log.info if log else print

    # 确保预测值和标签是 NumPy 数组
    pred = np.array(pred)
    label = np.array(label)
    args.average = 'binary'
    # 如果是二分类任务，将预测值转换为类别标签
    if args.dataset_type == 'classification':
        # 使用阈值 0.5 转换为类别标签
        pred_labels = (pred > 0.5).astype(int)
    else:
        pred_labels = pred

    results = {}
    results['acc'] = accuracy_score(label, pred_labels)
    results['precision'] = precision_score(label, pred_labels, average=args.average, zero_division=0)
    results['recall'] = recall_score(label, pred_labels, average=args.average, zero_division=0)
    results['f1'] = f1_score(label, pred_labels, average=args.average, zero_division=0)
    if metric_f == 'auc':
        try:
            test_score = roc_auc_score(label, pred)
        except ValueError as e:
            info(f'Warning: Error calculating AUC: {e}')
            test_score = float('nan')
    else:
        test_score = None

    # 记录日志
    if log:
        info(f"Test Accuracy: {results['acc']:.4f}")
        info(f"Test Precision: {results['precision']:.4f}")
        info(f"Test Recall: {results['recall']:.4f}")
        info(f"Test F1 Score: {results['f1']:.4f}")
        if test_score is not None:
            info(f"Test AUC: {test_score:.4f}")

    return test_score, results

def fold_train(args, log):
    info = log.info
    debug = log.debug

    debug('Start loading data')

    args.task_names = get_task_name(args.data_path)
    data = load_data(args.data_path, args)
    args.task_num = data.task_num()
    data_type = args.dataset_type
    if args.task_num > 1:
        args.is_multitask = 1

    debug(f'Splitting dataset with Seed = {args.seed}.')
    if args.val_path:
        val_data = load_data(args.val_path, args)
    if args.test_path:
        test_data = load_data(args.test_path, args)
    if args.val_path and args.test_path:
        train_data = data
    elif args.val_path:
        split_ratio = (args.split_ratio[0], 0, args.split_ratio[2])
        train_data, _, test_data = split_data(data, args.split_type, split_ratio, args.seed, log)
    elif args.test_path:
        split_ratio = (args.split_ratio[0], args.split_ratio[1], 0)
        train_data, val_data, _ = split_data(data, args.split_type, split_ratio, args.seed, log)
    else:

        train_data, val_data, test_data = split_data(data, args.split_type, args.split_ratio, args.seed, log)
    debug(
        f'Dataset size: {len(data)}    Train size: {len(train_data)}    Val size: {len(val_data)}    Test size: {len(test_data)}')

    # -------------------------- 核心修改：调整数据保存路径 --------------------------
    # 1. 定义根保存路径（固定为你指定的目录）
    root_save_dir = r'C:\Project\code\MGCR_SH\fpgnn\dataset\data_split'
    # 2. 拼接完整保存路径：根路径 + 模型名称（args.save_model_name）
    data_save_dir = os.path.join(root_save_dir, args.save_model_name)
    # 3. 创建文件夹（递归创建，避免路径不存在报错）
    os.makedirs(data_save_dir, exist_ok=True)
    info(f'划分后的数据集将保存到：{data_save_dir}')
    # ------------------------------------------------------------------------------

    # 整理数据集字典
    nnn = val_data.label()
    mmm = train_data.label()
    lll = test_data.label()
    train_dict = {}
    val_dict = {}
    test_dict = {}
    train_dict['smiles'] = train_data.smile()
    val_dict['smiles'] = val_data.smile()
    test_dict['smiles'] = test_data.smile()

    for ii in range(len(mmm[0])):
        ccc = []
        bbb = []
        aaa = []
        for i in range(len(mmm)):
            ccc.append(mmm[i][ii])
        for j in range(len(nnn)):
            bbb.append(nnn[j][ii])
        for k in range(len(lll)):
            aaa.append(lll[k][ii])
        train_dict[f'label{ii}'] = ccc
        val_dict[f'label{ii}'] = bbb
        test_dict[f'label{ii}'] = aaa

    # -------------------------- 保存训练数据 --------------------------
    train_filename = f"train_seed{args.seed}_{args.save_model_name}.csv"
    train_save_path = os.path.join(data_save_dir, train_filename)
    pd.DataFrame(train_dict).to_csv(train_save_path, index=False)
    info(f'训练数据已保存：{train_save_path}')

    # -------------------------- 保存验证数据 --------------------------
    val_filename = f"val_seed{args.seed}_{args.save_model_name}.csv"
    val_save_path = os.path.join(data_save_dir, val_filename)
    pd.DataFrame(val_dict).to_csv(val_save_path, index=False)
    info(f'验证数据已保存：{val_save_path}')

    # -------------------------- 保存测试数据 --------------------------
    test_filename = f"test_seed{args.seed}_{args.save_model_name}.csv"
    test_save_path = os.path.join(data_save_dir, test_filename)
    pd.DataFrame(test_dict).to_csv(test_save_path, index=False)
    info(f'测试数据已保存：{test_save_path}')
    # ------------------------------------------------------------------------------

    if data_type == 'regression':
        label_scaler = get_label_scaler(train_data)  # 计算得到label的均值和标准差
    else:
        label_scaler = None
    args.train_data_size = len(train_data)

    loss_f = get_loss(data_type, train_data, args.task_num)

    metric_f = get_metric(args.metric)

    debug('Training Model')
    model = FPFG(args)


    debug(model)
    save_name_kk = f'seed_{args.seed}_metric_{args.metric}.pt'

    if args.cuda and torch.cuda.is_available():
        model = model.to(torch.device("cuda"))
        debug("模型已移至CUDA设备")

    # 修改保存模型的路径（保持原有逻辑，仅优化路径拼接）
    save_path_full = os.path.join(args.save_path, args.save_model_name)
    os.makedirs(save_path_full, exist_ok=True)
    info(f'模型将保存到：{save_path_full}')

    optimizer = Adam(params=model.parameters(), lr=args.init_lr, weight_decay=0)
    scheduler = NoamLR(
        optimizer=optimizer,
        warmup_epochs=[args.warmup_epochs],
        total_epochs=[args.epochs] * args.num_lrs,
        steps_per_epoch=args.train_data_size // args.batch_size if args.batch_size > 0 else 1,
        init_lr=[args.init_lr],
        max_lr=[args.max_lr],
        final_lr=[args.final_lr]
    )

    # 初始化最佳分数
    if data_type == 'classification':
        best_score = -float('inf')
    else:
        best_score = float('inf')
    best_epoch = 0

    args.epochs = int(args.epochs)
    for epoch in range(args.epochs):
        info(f'Epoch {epoch}')

        epoch_train(model, train_data, loss_f, optimizer, scheduler, args)

        # 训练集预测与评分
        train_pred = predict(model, train_data, args.batch_size, label_scaler)
        train_label = train_data.label()
        if args.dataset_type == 'regression' and label_scaler is not None:
            train_label = train_label * label_scaler[1] + label_scaler[0]
        train_score = compute_score(train_pred, train_label, metric_f, args, log)

        # 验证集预测与评分
        val_pred = predict(model, val_data, args.batch_size, label_scaler)
        val_label = val_data.label()
        val_score = compute_score(val_pred, val_label, metric_f, args, log)

        ave_train_score = np.nanmean(train_score)
        info(f'Train {args.metric} = {ave_train_score:.6f}')

        ave_val_score = np.nanmean(val_score)
        info(f'Validation {args.metric} = {ave_val_score:.6f}')

        if args.task_num > 1:
            for one_name, one_score in zip(args.task_names, val_score):
                info(f'Validation {one_name} {args.metric} = {one_score:.6f}')

        # 保存最佳模型
        save_model_flag = False
        if data_type == 'classification' and ave_val_score > best_score:
            save_model_flag = True
        elif args.metric == 'r2' and ave_val_score > best_score:
            save_model_flag = True
        elif data_type == 'regression' and ave_val_score < best_score:
            save_model_flag = True

        if save_model_flag:
            best_score = ave_val_score
            best_epoch = epoch
            model_save_path = os.path.join(save_path_full, save_name_kk)
            save_model(model_save_path, model, label_scaler, args)
            info(f'最佳模型已更新并保存：{model_save_path}')

    info(f'Best validation {args.metric} = {best_score:.6f} on epoch {best_epoch}')

    # 加载最佳模型进行测试
    best_model_path = os.path.join(save_path_full, save_name_kk)
    model = load_model(best_model_path, args.cuda, log)
    test_smile = test_data.smile()
    test_label = test_data.label()

    test_pred = predict(model, test_data, args.batch_size, label_scaler)
    test_score, results = compute_score_test(test_pred, test_label, metric_f='auc', args=args, log=log)

    ave_test_score = np.nanmean(test_score)
    info(f'Seed {args.seed} : test {args.metric} = {ave_test_score:.6f}')

    if args.task_num > 1:
        for one_name, one_score in zip(args.task_names, test_score):
            info(f'Task {one_name} {args.metric} = {one_score:.6f}')

    return test_score, results