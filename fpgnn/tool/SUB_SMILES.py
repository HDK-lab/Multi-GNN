import rdkit
from rdkit import Chem
from rdkit.Chem import rdmolops

# 获取分子片段

def get_fragment_smiles(smiles: str, fragment_indices: list) -> str:
    """
    根据分子SMILES和片段原子索引提取子结构SMILES

    参数:
        smiles: 完整分子的SMILES字符串
        fragment_indices: 片段包含的原子索引列表（0-based）

    返回:
        片段的SMILES表示（带连接点标记）
    """
    # 创建分子对象
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError("无效的SMILES字符串")

    # 构建邻接表（记录每个原子的连接原子及键类型）
    adj = [[] for _ in range(mol.GetNumAtoms())]
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()
        adj[a1].append((a2, bond_type))
        adj[a2].append((a1, bond_type))

    # 提取子结构
    submol = Chem.RWMol()
    atom_map = {idx: submol.AddAtom(mol.GetAtomWithIdx(idx)) for idx in fragment_indices}

    # 添加片段内的键
    for idx in fragment_indices:
        for neighbor, bond_type in adj[idx]:
            if neighbor in fragment_indices and idx < neighbor:  # 避免重复添加
                submol.AddBond(atom_map[idx], atom_map[neighbor], bond_type)

    # 确定连接点（片段原子与外界相连的原子）
    connection_points = []
    for idx in fragment_indices:
        external_neighbors = [n for n, _ in adj[idx] if n not in fragment_indices]
        if len(external_neighbors) > 0:
            connection_points.append(idx)

    # 生成子结构SMILES
    sub_smiles = Chem.MolToSmiles(submol, isomericSmiles=False, canonical=False)

    # 添加连接点标记（假设单连接点情况，示例中为单侧连接）
    if len(connection_points) == 1:
        # 确定连接点在子结构中的位置（假设原子顺序保留原分子中的顺序）
        # 简单处理：单连接点时添加单键连接符
        return sub_smiles  #f"-{sub_smiles}" if fragment_indices[0] == connection_points[0] else f"{sub_smiles}-"
    elif len(connection_points) == 0:
        # 无连接点（环状结构或独立片段）
        return sub_smiles
    else:
        # 多连接点暂按无标记处理（可扩展）
        return sub_smiles


# 示例测试
if __name__ == "__main__":
    test_smiles = "CCCN1CCC(c2nc3cccc(C(N)=O)c3[nH]2)CC1"
    test_fragment = [3,4,20,5,19,6]  # 对应原子索引2,1,0（从左到右的前三个C原子）

    try:
        fragment_smiles = get_fragment_smiles(test_smiles, test_fragment)
        print(f"片段SMILES: {fragment_smiles}")  # 输出: -CCC
    except Exception as e:
        print(f"处理错误: {str(e)}")
