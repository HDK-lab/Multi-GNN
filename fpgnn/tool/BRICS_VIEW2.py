from rdkit import Chem
from rdkit.Chem import Draw

# 获取分子结构图片，基于原子序号

# 1. 输入SMILES并创建分子对象
smiles = "CCCN1CCC(c2nc3cccc(C(N)=O)c3[nH]2)CC1"  # 可替换为任意SMILES
mol = Chem.MolFromSmiles(smiles)

# 检查分子是否有效
if mol is None:
    raise ValueError("无效的SMILES字符串")

# 2. 将原子标签设置为索引
for atom in mol.GetAtoms():
    atom.SetProp("_displayLabel", str(atom.GetIdx()))  # 强制显示索引

# 3. 生成高分辨率图像
img = Draw.MolToImage(
    mol,
    size=(800, 800),  # 可根据需要调整分辨率
    kekulize=True,     # 显示芳香结构
    wedgeBonds=True    # 显示立体化学
)

# 4. 保存图片（支持PNG/JPG等格式）
img.save("molecule_with_indices.png")
