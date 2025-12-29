from argparse import Namespace
from rdkit import Chem
import random
from torch.utils.data.dataset import Dataset
import torch
class MoleData:
    def __init__(self,line,args):
        self.args = args
        self.smile = line[0]
        self.mol = Chem.MolFromSmiles(self.smile)
        self.label = [float(x) if x != '' else None for x in line[1:]]
        
    def task_num(self):
        return len(self.label)

    def to(self, device):
        # 确保 self.mol 是一个 torch.Tensor 对象
        if hasattr(self.mol, 'to'):
            self.mol = self.mol.to(device)
        # 确保 self.label 是一个 torch.Tensor 对象
        if self.label is not None:
            self.label = torch.tensor(self.label, device=device)
        return self
    def change_label(self,label):
        self.label = label


class MoleDataSet(Dataset):
    def __init__(self,data):
        self.data = data
        if len(self.data) > 0:
            self.args = self.data[0].args
        else:
            self.args = None
        self.scaler = None
    
    def smile(self):
        smile_list = []
        for one in self.data:
            smile_list.append(one.smile)
        return smile_list


    def to(self, device):
        for item in self.data:
            if hasattr(item, 'to'):
                item.to(device)
        return self

    def mol(self):
        mol_list = []
        for one in self.data:
            mol_list.append(one.mol)
        return mol_list
    
    def label(self):
        label_list = []
        for one in self.data:
            label_list.append(one.label)
        return label_list
    
    def task_num(self):
        if len(self.data) > 0:
            return self.data[0].task_num()
        else:
            return None
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,key):
        return self.data[key]
    
    def random_data(self,seed):
        random.seed(seed)
        random.shuffle(self.data)
    
    def change_label(self,label):
        assert len(self.data) == len(label)
        for i in range(len(label)):
            self.data[i].change_label(label[i])