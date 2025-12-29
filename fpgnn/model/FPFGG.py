import warnings

warnings.filterwarnings("ignore", message="DEPRECATION WARNING: please use GetValence")

from rdkit.Chem import AllChem, MACCSkeys
from fpgnn.data import GetPubChemFPs
from rdkit.Chem import BRICS, Draw
import torch_geometric.nn as pyg_nn
import numpy as np
from rdkit import Chem
from torch_geometric.data import DataLoader, Data
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


# 全局设备设置（统一管理）
def get_device(cuda=True):
    return torch.device('cuda:0' if cuda and torch.cuda.is_available() else 'cpu')


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


BT_MAPPING_INT = {
    Chem.rdchem.BondType.SINGLE: 1,
    Chem.rdchem.BondType.DOUBLE: 2,
    Chem.rdchem.BondType.TRIPLE: 3,
    Chem.rdchem.BondType.AROMATIC: 1.5,
}


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]


def atom_features(atom, use_chirality=True):
    results = one_of_k_encoding_unk(
        atom.GetSymbol(),
        ['B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At', 'other']) + \
              one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), [
                  Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                  Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                  Chem.rdchem.HybridizationType.SP3D2, 'other']) + [atom.GetIsAromatic()]
    results += one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    if use_chirality:
        try:
            results += one_of_k_encoding_unk(
                atom.GetProp('_CIPCode'), ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results += [False, False] + [atom.HasProp('_ChiralityPossible')]
    return np.array(results)


def generate_graph_features(mol):
    graph_features = [0] * 22
    node_features = {}
    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        atomic_norm = np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                                     ['B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br',
                                                      'Te', 'I', 'At', 'other']))
        implicit_substruct_valence = 0
        for neighbor in atom.GetNeighbors():
            j = neighbor.GetIdx()
            implicit_substruct_valence += BT_MAPPING_INT[mol.GetBondBetweenAtoms(i, j).GetBondType()]
        formal_charge = atom.GetFormalCharge()
        num_Hs = atom.GetTotalNumHs()
        valence = atom.GetExplicitValence() - 2 * implicit_substruct_valence
        is_aromatic = 1 if atom.GetIsAromatic() > 0 else 0
        mass = atom.GetMass()
        edges_sum = implicit_substruct_valence
        list_atom_feature = list(atomic_norm)
        list_bond_feature = [num_Hs * 0.1, valence, formal_charge, is_aromatic, mass * 0.01, edges_sum * 0.1]
        node_features[i] = list_atom_feature + list_bond_feature
        graph_features = np.sum([list_atom_feature + list_bond_feature, graph_features], axis=0).tolist()
    return node_features, graph_features


def generate_substructure_features(mol, substruct):
    atoms = [mol.GetAtomWithIdx(i) for i in substruct]
    substruct_atomic_encoding = np.array([one_of_k_encoding_unk(atom.GetSymbol(),
                                                                ['B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As',
                                                                 'Se', 'Br', 'Te', 'I', 'At', 'other'])
                                          for atom in atoms])
    substruct_atomic_sum = np.sum(substruct_atomic_encoding, axis=0).tolist()
    substruct_atomic_sum_norm = [0.1 * s for s in substruct_atomic_sum]
    implicit_substruct_valence = 0
    for i in range(len(atoms)):
        for j in range(i, len(atoms)):
            bond = mol.GetBondBetweenAtoms(substruct[i], substruct[j])
            if bond:
                implicit_substruct_valence += BT_MAPPING_INT[bond.GetBondType()]
    substruct_formal_charge = sum(atom.GetFormalCharge() for atom in atoms)
    substruct_num_Hs = sum(atom.GetTotalNumHs() for atom in atoms)
    substruct_valence = sum(atom.GetExplicitValence() for atom in atoms) - 2 * implicit_substruct_valence
    substruct_is_aromatic = 1 if sum(atom.GetIsAromatic() for atom in atoms) > 0 else 0
    substruct_mass = sum(atom.GetMass() for atom in atoms)
    substruct_edges_sum = implicit_substruct_valence
    features = substruct_atomic_sum_norm + [substruct_num_Hs * 0.1, substruct_valence, substruct_formal_charge,
                                            substruct_is_aromatic, substruct_mass * 0.01, substruct_edges_sum * 0.1]
    return features


def return_brics_leaf_structure(smiles):
    m = Chem.MolFromSmiles(smiles)
    res = list(BRICS.FindBRICSBonds(m))
    all_brics_bond = [set(r[0]) for r in res]
    all_brics_atom = list(set([a for b in all_brics_bond for a in b]))
    substrate_idx = {}
    if len(all_brics_atom) > 0:
        all_break_atom = {}
        for atom in all_brics_atom:
            all_break_atom[atom] = [a for b in all_brics_bond if atom in b for a in b if a != atom]
        used_atom = []
        for idx, break_atoms in all_break_atom.items():
            if idx not in used_atom:
                substrate_idx_i = [idx]
                begin_list = [idx]
                while begin_list:
                    new_begin = []
                    for i in begin_list:
                        for neighbor in m.GetAtomWithIdx(i).GetNeighbors():
                            n_idx = neighbor.GetIdx()
                            if n_idx not in substrate_idx_i and n_idx not in break_atoms:
                                substrate_idx_i.append(n_idx)
                                new_begin.append(n_idx)
                    begin_list = new_begin
                substrate_idx[idx] = substrate_idx_i
                used_atom += substrate_idx_i
    else:
        substrate_idx[0] = list(range(m.GetNumAtoms()))
    return {'substructure': substrate_idx, 'substructure_bond': all_brics_bond}


def reindex_substructure(substructure_dir):
    all_atom_list = list(set([a for b in substructure_dir['substructure_bond'] for a in b]))
    substructure_reindex = {i: v for i, (_, v) in enumerate(substructure_dir['substructure'].items())}
    substructure_dir['substructure_reindex'] = substructure_reindex
    new_sub_bond_dir = {}
    for atom in all_atom_list:
        for j, sub in substructure_reindex.items():
            if atom in sub:
                new_sub_bond_dir[atom] = j
                break
    ss_bond = [[new_sub_bond_dir[a] for a in b] for b in substructure_dir['substructure_bond']]
    substructure_dir['ss_bond'] = ss_bond
    return substructure_dir


def dataload_g(smiles, batch_size):
    dataset = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        atoms_feature, _ = generate_graph_features(mol)
        x = torch.tensor(list(atoms_feature.values()), dtype=torch.float)
        edge_index = torch.tensor([[b.GetBeginAtomIdx() for b in mol.GetBonds()],
                                   [b.GetEndAtomIdx() for b in mol.GetBonds()]], dtype=torch.long)
        dataset.append(Data(x=x, edge_index=edge_index))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def dataload_f(smiles, batch_size):
    dataset = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        sub_dir = return_brics_leaf_structure(s)
        sub_dir = reindex_substructure(sub_dir)
        s_features = [generate_substructure_features(mol, tuple(sub))
                      for sub in sub_dir['substructure_reindex'].values()]
        x = torch.tensor(s_features, dtype=torch.float)
        edge_index = torch.tensor([[b[0] for b in sub_dir['ss_bond']],
                                   [b[1] for b in sub_dir['ss_bond']]], dtype=torch.long)
        dataset.append(Data(x=x, edge_index=edge_index))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False), sub_dir['substructure_reindex']


def dataload_gf(smiles, batch_size):
    dataset_g, dataset_f = [], []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        # F数据
        sub_dir = return_brics_leaf_structure(s)
        sub_dir = reindex_substructure(sub_dir)
        s_features = [generate_substructure_features(mol, tuple(sub))
                      for sub in sub_dir['substructure_reindex'].values()]
        x_f = torch.tensor(s_features, dtype=torch.float)
        edge_f = torch.tensor([[b[0] for b in sub_dir['ss_bond']],
                               [b[1] for b in sub_dir['ss_bond']]], dtype=torch.long)
        dataset_f.append(Data(x=x_f, edge_index=edge_f))
        # G数据
        atoms_feature, _ = generate_graph_features(mol)
        x_g = torch.tensor(list(atoms_feature.values()), dtype=torch.float)
        edge_g = torch.tensor([[b.GetBeginAtomIdx() for b in mol.GetBonds()],
                               [b.GetEndAtomIdx() for b in mol.GetBonds()]], dtype=torch.long)
        dataset_g.append(Data(x=x_g, edge_index=edge_g))
    return DataLoader(dataset_g, batch_size=batch_size, shuffle=False), \
        DataLoader(dataset_f, batch_size=batch_size, shuffle=False)


class FGN(nn.Module):
    def __init__(self, args):
        super(FGN, self).__init__()
        self.hide_features = args.hide_features
        self.hidden_size = args.hidden_size
        self.batch_size = args.batch_size
        self.conv1 = GCNConv(22, self.hide_features)
        self.conv2 = GCNConv(self.hide_features, self.hide_features)
        self.out = nn.Linear(self.hide_features, self.hidden_size)

    def forward(self, smile):
        # 获取模型所在设备（避免重复设置）
        device = next(self.parameters()).device
        dataset_g, _ = dataload_gf(smile, self.batch_size)
        for data in dataset_g:
            data = data.to(device)  # 数据移到模型所在设备
            x = self.conv1(data.x, data.edge_index).relu()
            x = self.conv2(x, data.edge_index).relu()
            x = pyg_nn.global_max_pool(x, data.batch)
            x = F.dropout(x, training=self.training)
            out = self.out(x)
        return out


class GNN(nn.Module):
    def __init__(self, args):
        super(GNN, self).__init__()
        self.hide_features = args.hide_features
        self.hidden_size = args.hidden_size
        self.batch_size = args.batch_size
        self.conv1 = GCNConv(22, self.hide_features)
        self.conv2 = GCNConv(self.hide_features, self.hide_features)
        self.out = nn.Linear(self.hide_features, self.hidden_size)

    def forward(self, smile):
        device = next(self.parameters()).device
        _, dataset_f = dataload_gf(smile, self.batch_size)
        for data in dataset_f:
            data = data.to(device)
            x = self.conv1(data.x, data.edge_index).relu()
            x = self.conv2(x, data.edge_index).relu()
            x = pyg_nn.global_max_pool(x, data.batch)
            x = F.dropout(x, training=self.training)
            out = self.out(x)
        return out


class FPN(nn.Module):
    def __init__(self, args):
        super(FPN, self).__init__()
        self.hidden_dim = args.hidden_size
        self.fp_dim = 1489
        self.fp_2_dim = 512
        self.fp_changebit = getattr(args, 'fp_changebit', None)
        self.fc1 = nn.Linear(self.fp_dim, self.fp_2_dim)
        self.act_func = nn.ReLU()
        self.dropout = nn.Dropout(p=args.dropout)
        self.fc2 = nn.Linear(self.fp_2_dim, self.hidden_dim)

    def forward(self, smile):
        device = next(self.parameters()).device
        fp_list = []
        for s in smile:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                fp = [0] * self.fp_dim
            else:
                fp_maccs = list(AllChem.GetMACCSKeysFingerprint(mol))
                fp_erg = list(AllChem.GetErGFingerprint(mol, fuzzIncrement=0.3, maxPath=21, minPath=1))
                fp_pub = list(GetPubChemFPs(mol))
                fp = fp_maccs + fp_erg + fp_pub
            fp_list.append(fp)

        if self.fp_changebit is not None and self.fp_changebit != 0:
            fp_array = np.array(fp_list)
            fp_array[:, self.fp_changebit - 1] = 1
            fp_list = fp_array.tolist()

        fp_tensor = torch.tensor(fp_list, dtype=torch.float, device=device)  # 直接创建在目标设备
        out = self.fc1(fp_tensor)
        out = self.dropout(out)
        out = self.act_func(out)
        out = self.fc2(out)
        return out


class fpfgModel_att(nn.Module):
    def __init__(self, is_classif, cuda, dropout_fpn, args):
        super(fpfgModel_att, self).__init__()
        self.device = get_device(cuda)
        self.is_classif = is_classif
        self.dropout_fpn = dropout_fpn
        self.ablation_mode = args.ablation_mode

        # 创建子模块（自动移到设备）
        self.create_fgn(args)
        self.create_gnn(args)
        self.create_fpn(args)
        self.create_scale(args)

        # 门控注意力机制
        hidden = args.hidden_size
        self.gate_fgn = nn.Linear(hidden, 1).to(self.device)
        self.gate_gnn = nn.Linear(hidden, 1).to(self.device)
        self.gate_fpn = nn.Linear(hidden, 1).to(self.device)
        self.softmax = nn.Softmax(dim=1)

        # FFN层
        self.create_ffn(args)

        if self.is_classif:
            self.sigmoid = nn.Sigmoid().to(self.device)

        self.to(self.device)

    def create_fgn(self, args):
        self.encoder2 = FGN(args).to(self.device)

    def create_gnn(self, args):
        self.encoder3 = GNN(args).to(self.device)

    def create_fpn(self, args):
        self.encoder5 = FPN(args).to(self.device)

    def create_scale(self, args):
        linear_dim = args.hidden_size
        self.fc_fgn = nn.Linear(linear_dim, linear_dim).to(self.device)
        self.fc_gnn = nn.Linear(linear_dim, linear_dim).to(self.device)
        self.fc_fpn = nn.Linear(linear_dim, linear_dim).to(self.device)
        self.act_func = nn.ReLU().to(self.device)

    def create_ffn(self, args):
        linear_dim = args.hidden_size
        self.ffn = nn.Sequential(
            nn.Dropout(self.dropout_fpn),
            nn.Linear(linear_dim, linear_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_fpn),
            nn.Linear(linear_dim, args.task_num),
        ).to(self.device)

    def forward(self, smile, num=None):
        # SMILES是字符串列表，无需移到设备
        # 各分支特征提取
        fgn = self.act_func(self.fc_fgn(self.encoder2(smile)))
        gnn = self.act_func(self.fc_gnn(self.encoder3(smile)))
        fpn = self.act_func(self.fc_fpn(self.encoder5(smile)))

        # 收集有效分支
        gates = []
        reps = []
        names = []
        if self.ablation_mode != 4:  # 保留FGN
            gates.append(self.gate_fgn(fgn))
            reps.append(fgn)
            names.append("FGN")
        if self.ablation_mode != 3:  # 保留GNN
            gates.append(self.gate_gnn(gnn))
            reps.append(gnn)
            names.append("GNN")
        if self.ablation_mode != 2:  # 保留FPN
            gates.append(self.gate_fpn(fpn))
            reps.append(fpn)
            names.append("FPN")

        # 计算注意力权重
        gates = torch.cat(gates, dim=1)
        attn_weights = self.softmax(gates)

        # 加权融合
        rep_final = torch.zeros_like(reps[0], device=self.device)
        for i, r in enumerate(reps):
            rep_final += attn_weights[:, i:i + 1] * r

        # 最终输出
        output = self.ffn(rep_final)
        if self.is_classif and not self.training:
            output = self.sigmoid(output)

        # 注意力信息（用于查看）
        attn_info = {
            "weights": attn_weights,
            "names": names,
            "tensor": attn_weights
        }
        return output,attn_info


class fpfgModel(nn.Module):
    def __init__(self, is_classif, cuda, dropout_fpn, args):
        super(fpfgModel, self).__init__()
        self.device = get_device(cuda)
        self.is_classif = is_classif
        self.dropout_fpn = dropout_fpn
        self.ablation_mode = 1
        self.hidden_size = args.hidden_size

        # 创建子模块（自动移到设备）
        self.create_fgn(args)
        self.create_gnn(args)
        self.create_fpn(args)
        self.create_scale(args)


        self.concat_fc = nn.Linear(
            in_features=3 * self.hidden_size,
            out_features=self.hidden_size
        ).to(self.device)

        # FFN层（保持不变）
        self.create_ffn(args)

        if self.is_classif:
            self.sigmoid = nn.Sigmoid().to(self.device)

        self.to(self.device)


    def create_fgn(self, args):
        self.encoder2 = FGN(args).to(self.device)

    def create_gnn(self, args):
        self.encoder3 = GNN(args).to(self.device)

    def create_fpn(self, args):
        self.encoder5 = FPN(args).to(self.device)

    def create_scale(self, args):
        linear_dim = args.hidden_size
        self.fc_fgn = nn.Linear(linear_dim, linear_dim).to(self.device)
        self.fc_gnn = nn.Linear(linear_dim, linear_dim).to(self.device)
        self.fc_fpn = nn.Linear(linear_dim, linear_dim).to(self.device)
        self.act_func = nn.ReLU().to(self.device)

    def create_ffn(self, args):
        linear_dim = args.hidden_size
        self.ffn = nn.Sequential(
            nn.Dropout(self.dropout_fpn),
            nn.Linear(linear_dim, linear_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_fpn),
            nn.Linear(linear_dim, args.task_num),
        ).to(self.device)

    def forward(self, smile, num=None):
        # 各分支特征提取（保持不变）
        fgn = self.act_func(self.fc_fgn(self.encoder2(smile)))
        gnn = self.act_func(self.fc_gnn(self.encoder3(smile)))
        fpn = self.act_func(self.fc_fpn(self.encoder5(smile)))

        rep_list = [fgn, gnn, fpn]
        names = ["FGN", "GNN", "FPN"]

        # ========== 特征拼接（固定3分支拼接，逻辑简化） ==========
        # 拼接所有3个分支的特征 [batch_size, 3*hidden_size]
        rep_concat = torch.cat(rep_list, dim=1)
        # 降维回原hidden_size [batch_size, hidden_size]
        rep_final = self.concat_fc(rep_concat)
        rep_final = self.act_func(rep_final)

        # 最终输出（保持不变）
        output = self.ffn(rep_final)
        if self.is_classif and not self.training:
            output = self.sigmoid(output)

        return output, rep_final


def FPFG(args):
    is_classif = 1 if args.dataset_type == 'classification' else 0
    model = fpfgModel(is_classif=is_classif,
                      cuda=args.cuda,
                      dropout_fpn=args.dropout,
                      args=args)

    # 参数初始化
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)

    return model