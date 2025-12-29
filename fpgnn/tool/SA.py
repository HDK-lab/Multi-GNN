import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
import math


def calculate_sa_score(mol):
    # 计算环的数量
    n_ring = rdMolDescriptors.CalcNumRings(mol)
    # 计算分子的 Murcko 骨架
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    # 计算骨架的原子数
    n_scaffold = scaffold.GetNumAtoms()
    # 计算分子的原子数
    n_atoms = mol.GetNumAtoms()

    # 计算 SA 分数的几个组成部分
    # 环的惩罚项
    ring_penalty = 0
    if n_ring > 4:
        ring_penalty = 0.4 * (n_ring - 4)
    # 骨架大小的奖励项
    scaffold_reward = 0
    if n_scaffold < 12:
        scaffold_reward = 0.2 * (12 - n_scaffold)
    # 原子数的惩罚项
    atom_penalty = 0
    if n_atoms > 50:
        atom_penalty = 0.2 * (n_atoms - 50)

    # 计算 SA 分数
    sa_score = 1.0 + ring_penalty - scaffold_reward + atom_penalty

    # 边界检查，确保 SA 分数在 1 - 10 之间
    sa_score = max(1, min(10, sa_score))

    return sa_score


def main(csv_file):
    # 读取 CSV 文件
    df = pd.read_csv(csv_file)
    sa_scores = []
    for smiles in df['SMILES']:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            sa_score = calculate_sa_score(mol)
            sa_scores.append(sa_score)
        else:
            sa_scores.append(None)
    df['SA_Score'] = sa_scores
    # 将结果保存到新的 CSV 文件
    df.to_csv('D:\PycharmProjects\FP_GNN_20240901\FP_GNN\\fpgnn\dataset\save_dataset\ESOL_SIRT_BBBP_TEST_4.csv', index=False)


if __name__ == "__main__":
    csv_file = 'D:\PycharmProjects\FP_GNN_20240901\FP_GNN\\fpgnn\dataset\esol_bbbp_opt_data\ESOL_SIRT_BBBP.csv'  # 请替换为你的 CSV 文件路径
    main(csv_file)
