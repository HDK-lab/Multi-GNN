import csv
import numpy as np
from rdkit.Chem import MolToSmiles

from fpgnn.model.FPFGG import return_brics_leaf_structure, reindex_substructure
from fpgnn.tool import set_predict_argument, get_scaler, load_args, load_data, load_model
from fpgnn.tool.SUB_SMILES import get_fragment_smiles
from fpgnn.tool.search_pro import load_smiles_properties
from fpgnn.train import predict, sub_predict
from fpgnn.data import MoleDataSet
CUDA_LAUNCH_BLOCKING=1

def predicting1(args):
    args.MASK = 1
    print('Load args.')
    scaler = get_scaler(args.model_path)
    print('scaler', scaler)
    train_args = load_args(args.model_path)

    for key, value in vars(train_args).items():
        if not hasattr(args, key):
            setattr(args, key, value)

    print('Load data.')
    test_data = load_data(args.predict_path, args)
    print('Load model')
    model = load_model(args.model_path, args.cuda)

    test_smiles = test_data.smile()
    for idx in range(len(test_smiles)):
        all_brics_substructure_subset = return_brics_leaf_structure(test_smiles[idx])
        reindex = reindex_substructure(all_brics_substructure_subset)
        f_idx = reindex['substructure_reindex']
        if len(f_idx) <= 1:
            continue
        for num in range(len(f_idx)):

            fra1 = f_idx[num]
            test_pred = sub_predict(model, test_smiles[idx], args.batch_size, scaler,num)
            test_pred = np.array(test_pred)
            test_pred = test_pred.tolist()

            # 查询分子的属性值
            csv_path = args.property_dataset_path
            smiles_to_lookup = test_smiles[idx]

            properties = load_smiles_properties(csv_path)
            if properties:
                try:
                    value = properties[smiles_to_lookup]
                    # print(f"SMILES '{smiles_to_lookup}' 对应的属性值为：{value}")
                except KeyError:
                    print(f"未找到SMILES '{smiles_to_lookup}' 对应的属性值")
                    break

            sub_smiles = get_fragment_smiles(test_smiles[idx],fra1)
            mol_property = float(value)
            with open(args.result_path, 'a', newline='') as file:
                writer = csv.writer(file)
                # for i in range(len(test_data)):
                line = []
                line.append(test_smiles[idx])
                line.append(sub_smiles)
                sub_importence = round(mol_property - test_pred[0][0],5)
                sub_importence_list = []
                sub_importence_list.append(sub_importence)
                line.extend(sub_importence_list)
                # line.extend(test_pred[0])
                writer.writerow(line)


if __name__ == '__main__':
    args = set_predict_argument()
    predicting1(args)
