import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED

# 输入分子的 SMILES 字符串
# smiles = 'O=C(Nc1ccc2[nH]c(=O)oc2c1)C(=O)N1CCC(Cc2ccc(F)cc2)CC1'  # 例如阿司匹林（Aspirin）
data = pd.read_csv('D:\PycharmProjects\FP_GNN_20240901\FP_GNN\\fpgnn\dataset\qed\\testqed2.csv')  # 读取文件中所有数据
header = data.columns
header_list = header.tolist()
data_smiles = data[header_list[0]]
qed_list = []
for smiles in data_smiles:
    # 将 SMILES 转换为 RDKit 分子对象
    mol = Chem.MolFromSmiles(smiles)

    # 计算 QED 值
    qed_value = QED.qed(mol)

    qed_list.append(qed_value)

    # 打印 QED 值
    print(f"QED Value: {qed_value}")
QEDdata_dict = {}
QEDdata_dict['smiles'] = data_smiles
QEDdata_dict['qed'] = qed_list
df = pd.DataFrame(QEDdata_dict)
df.to_csv('D:\PycharmProjects\FP_GNN_20240901\FP_GNN\\fpgnn\dataset\qed\\testqed2_res.csv')