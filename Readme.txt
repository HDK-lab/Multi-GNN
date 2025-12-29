0. Dependencies Installation
Requirement: Python 3.10.19Install the specified packages with the following versions:

python 3.10.19
torch 2.8.0
rdkit 2025.03.6
pandas 2.3.3
numpy 2.2.5

1. Data Preparation
Place your data file(s) in the directory: fpgnn/dataset/sirt_s/.The file must contain two columns: smiles and label.

2. Training Execution
Open the file train_main.py.
Set the two parameters: task_name and args.data_path.
Run train_main.py directly.
Configure all remaining parameters in fpgnn/tool/args.py.



