"""Legacy optional utility with dataset-specific paths; use get_data.py for the README workflow."""

import pandas as pd
from rdkit import Chem
import ast
df1 = pd.read_csv('./csvs/ld50_train.csv', header=None)
df2 = pd.read_csv('../data/RT/LD50/train.csv')
label_number=df1.iloc[:, 0].tolist()
labels=df2['expt']
smiles_all=df2['smiles']
smiles_no=[]
labels_no=[]
for i in range(len(labels)):
    if i not in label_number:
        smiles_no.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles_all[i]))) 
        labels_no.append(labels[i])


data_no_3d_all=[]
for i in range(len(smiles_no)):
    data_no_3d = ' '.join(smiles_no[i])    
    data_no_3d=data_no_3d.strip()
    data_no_3d= data_no_3d+' GEO' +' [SEP]'+' labels '+str(labels_no[i])
    data_no_3d_all.append(data_no_3d)
    

# Convert tokenized torsions to model input records.
second_column = df1.iloc[:,1]
print(len(second_column))
labels_2=df1.iloc[:,2]
result=[]
for i in range(len(second_column)):
    row=second_column[i]
    row=ast.literal_eval(row)
    row.append('labels')
    row.append(str(labels_2[i]))
    result_sequence = ' '.join(row)
    result.append(result_sequence)
print(len(result))

torsion_all=result + data_no_3d_all
df3=pd.DataFrame()
df3['smiles_label_torsion_all']=torsion_all
df3.to_csv('./csvs/RT/ld50_train_smiles_label_torsion_all.csv',index=False)
