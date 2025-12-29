from rdkit import Chem
from rdkit.Chem import AllChem

# 替换为你的 SMILES 字符串
smiles = "O=c1c(CO)cn(C2CC2)c2cc(N3CCNCC3)c(F)cc12"

# 创建分子对象
mol = Chem.MolFromSmiles(smiles)

# 添加氢原子（3D结构需要氢）
mol = Chem.AddHs(mol)

# 生成3D坐标（使用ETKDG算法）
AllChem.EmbedMolecule(mol, useRandomCoords=True)

# 能量最小化优化结构（可选，但推荐）
AllChem.UFFOptimizeMolecule(mol)

# 保存为SDF文件（替换文件名）
filename = "compound15.sdf"
writer = Chem.SDWriter(filename)
writer.write(mol)
writer.close()

print(f"已生成3D结构并保存为 {filename}")
