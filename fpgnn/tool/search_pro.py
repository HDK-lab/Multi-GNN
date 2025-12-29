import csv


def load_smiles_properties(csv_file):
    """
    从CSV文件加载SMILES到属性值的映射
    :param csv_file: CSV文件路径，第一列为SMILES，第二列为属性值
    :return: 字典，键为SMILES，值为属性值
    """
    smiles_dict = {}
    try:
        with open(csv_file, 'r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) >= 2:  # 确保行数据足够
                    smiles = row[0].strip()
                    property_value = row[1].strip()
                    smiles_dict[smiles] = property_value
        return smiles_dict
    except FileNotFoundError:
        print(f"错误：文件 {csv_file} 未找到")
        return None
    except Exception as e:
        print(f"错误：读取文件时发生异常 - {str(e)}")
        return None


if __name__ == "__main__":
    csv_path = 'D:\PycharmProjects\FP_GNN_20240901\FP_GNN\\fpgnn\dataset\save_dataset\RES_BBBP_drugbank_24.csv'
    smiles_to_lookup = 'N[C@@H](CC(=O)O)C(=O)O'

    properties = load_smiles_properties(csv_path)
    if properties:
        try:
            value = properties[smiles_to_lookup]
            print(f"SMILES '{smiles_to_lookup}' 对应的属性值为：{value}")
        except KeyError:
            print(f"未找到SMILES '{smiles_to_lookup}' 对应的属性值")
