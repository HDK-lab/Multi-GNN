from argparse import ArgumentParser, Namespace
import torch
from .tool import mkdir
"""
    hide_features  epochs  hidden_size    dropout

HCAR1_:'dropout': 0.0, 'epochs': 30.0, 'hidden_size': 768.0, 'hide_features': 768.0

"""

def add_train_argument(p):
    p.add_argument('--batch_size',type=int,default=50,
                   help='The size of batch.')
    p.add_argument('--hide_features', type=int, default=768)
    p.add_argument('--epochs',type=int,default=30,
                   help='The number of epochs.')
    p.add_argument('--hidden_size',type=int,default=768,
                   help='The dim of hidden layers in model.')
    p.add_argument('--dropout',type=float,default=0.0,
                   help='The dropout of fpn and ffn.')
    p.add_argument('--data_path', type=str, default='C:\Project\code\Multi_GNN\\fpgnn\dataset\\freesolv.csv',
                   help='The path of input CSV file.')
    p.add_argument('--save_path',type=str,default='C:\Project\code\Multi_GNN\\fpgnn/dataset/save_model\HCAR1_1',
                   help='The path to save output model.pt.,default is "model_save/"')
    p.add_argument('--save_path_result',type=str, default='C:\Project\code\Multi_GNN\\fpgnn\dataset/results/results_2025/results_2025.csv',
                   help='The path to save output model.pt.,default is "model_save/"')
    p.add_argument('--save_train_data_path',type=str,default='C:\Project\code\Multi_GNN\\fpgnn/dataset/data_split/train_dataset.csv')
    p.add_argument('--save_val_data_path',type=str,default='C:\Project\code\Multi_GNN\\fpgnn/dataset/data_split/val_dataset.csv')
    p.add_argument('--save_test_data_path',type=str,default='C:\Project\code\Multi_GNN\\fpgnn/dataset/data_split/test_dataset.csv')
    p.add_argument('--log_path',type=str,default='log',
                   help='The dir of output log file.')
    p.add_argument('--save_model_name', type=str, default='log')
    p.add_argument('--dataset_type',type=str,choices=['classification', 'regression'],default='classification',
                   help='The type of dataset.')
    p.add_argument('--is_multitask',type=int,default=0,
                   help='Whether the dataset is multi-task. 0:no  1:yes.')
    p.add_argument('--task_num',type=int,default=1,
                   help='The number of task in multi-task training.')
    p.add_argument('--split_type',type=str,choices=['random'],default='random',
                   help='The type of data splitting.')
    p.add_argument('--split_ratio',type=float,nargs=3,default=[0.8,0.1,0.1],
                   help='The ratio of data splitting.[train,valid,test]')
    p.add_argument('--val_path',type=str,
                   help='The path of excess validation data.')
    p.add_argument('--test_path',type=str,
                   help='The path of excess testing data.')
    p.add_argument('--seed',type=int,default=24,
                   help='The random seed of model. Using in splitting data.')
    p.add_argument('--num_folds',type=int,default=5,
                   help='The number of folds in cross validation.')#交叉验证
    p.add_argument('--metric',type=str,choices=['auc', 'prc-auc', 'rmse', 'r2', 'mae'], default=None,
                   help='The metric of data evaluation.')

    p.add_argument('--dropout_gat',type=float,default=0.2,
                   help='The dropout of gnn.')
    p.add_argument('--MASK', type=int, default=0)



def add_hyper_argument(p):
    p.add_argument('--search_num', type=int,default=200,
                   help='The number of hyperparameters searching.')

def add_interfp_argument(p):
    p.add_argument('--log_path', type=str,default='log',
                   help='The path of log file.')

def add_intergraph_argument(p):
    p.add_argument('--predict_path', type=str,
                   help='The path of input CSV file to predict.')
    p.add_argument('--figure_path', type=str,default='figure',
                   help='The path of output figure file.')
    p.add_argument('--model_path', type=str,
                   help='The path of model.pt.')
    p.add_argument('--batch_size', type=int,default=50,
                   help='The size of batch.')

def set_train_argument():
    p = ArgumentParser()
    add_train_argument(p)
    args = p.parse_args()

    assert args.data_path
    assert args.dataset_type

    mkdir(args.save_path)

    if args.metric is None:
        if args.dataset_type == 'classification':
            args.metric = 'auc'
        elif args.dataset_type == 'regression':
            args.metric = 'rmse'

    if args.dataset_type == 'classification' and args.metric not in ['auc', 'prc-auc']:
        raise ValueError('Metric or data_type is error.')
    if args.dataset_type == 'regression' and args.metric not in ['rmse','mae','r2']:
        raise ValueError('Metric or data_type is error.')

    args.cuda = torch.cuda.is_available()
    # args.cuda = False
    args.init_lr = 1e-4
    args.max_lr = 1e-3
    args.final_lr = 1e-4
    args.warmup_epochs = 2
    args.num_lrs = 1

    return args


def add_optimization_argument(p):
    p.add_argument('--optimization_path', type=str, default='D:\PycharmProjects\FP_GNN_20240901\FP_GNN\\fpgnn\dataset\\res\BBBP_OPT.csv',
                   help='The path of input CSV file to optimization.')
    p.add_argument('--result_path', type=str,default='',
                   help='The path of output CSV file.')
    p.add_argument('--model_path', type=str,default='',
                   help='The path of model.pt.')
    p.add_argument('--batch_size', type=int, default=64,
                   help='The size of batch.')


def set_optimization_argument():
    p = ArgumentParser()
    add_optimization_argument(p)
    args = p.parse_args()

    # assert args.optimization_path
    # assert args.model_path

    args.cuda = torch.cuda.is_available()
    # args.cuda = False
    # mkdir(args.result_path, isdir = False)

    return args




def set_hyper_argument():
    p = ArgumentParser()
    add_train_argument(p)
    add_hyper_argument(p)
    args = p.parse_args()

    assert args.data_path
    assert args.dataset_type

    mkdir(args.save_path)

    if args.metric is None:
        if args.dataset_type == 'classification':
            args.metric = 'auc'
        elif args.dataset_type == 'regression':
            args.metric = 'rmse'

    if args.dataset_type == 'classification' and args.metric not in ['auc', 'prc-auc']:
        raise ValueError('Metric or data_type is error.')
    if args.dataset_type == 'regression' and args.metric not in ['rmse', 'r2', 'mae']:
        raise ValueError('Metric or data_type is error.')
    # if args.fp_type not in ['mixed', 'morgan']:
    #     raise ValueError('Fingerprint type is error.')

    args.cuda = torch.cuda.is_available()
    # args.cuda = False
    args.init_lr = 1e-4
    args.max_lr = 1e-3
    args.final_lr = 1e-4
    args.warmup_epochs = 2
    args.num_lrs = 1
    args.search_now = 0

    return args



def set_intergraph_argument():
    p = ArgumentParser()
    add_intergraph_argument(p)
    args = p.parse_args()


    assert args.predict_path
    assert args.model_path

    args.cuda = torch.cuda.is_available()
    # args.cuda = False
    args.inter_graph = 1

    mkdir(args.figure_path, isdir = True)

    return args