# -*- coding:utf-8 -*-
"""Legacy optional utility with dataset-specific paths; use get_data.py for the README workflow."""
# @Author: yangyang
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from tqdm import tqdm
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
RDLogger.DisableLog('rdApp.*')


if __name__ == '__main__':
    all_smiles_new = []
    all_mols = []

    df = pd.read_csv('../data/RT/LD50/train.csv')
    all_smiles = df['smiles']

    inludeChirality=True
    for smiles in tqdm(all_smiles):
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles),isomericSmiles=inludeChirality)
        all_smiles_new.append(smiles)
    print("Successfully extracting SMILES... ")

    w1 = Chem.SDWriter('./sdfs/ld50_train_mols.sdf')
    for smiles in tqdm(all_smiles_new):
        numConfs=20
        mol=Chem.MolFromSmiles(smiles)
        try:
            mol = Chem.AddHs(mol)
            res = AllChem.EmbedMultipleConfs(mol, numConfs=numConfs)
            ### MMFF generates multiple conformations
            res = AllChem.MMFFOptimizeMoleculeConfs(mol)
            mol = Chem.RemoveHs(mol)
            index = np.argmin([x[1] for x in res])
        except :
            AllChem.Compute2DCoords(mol)
            index=0
        all_mols.append(Chem.RemoveHs(mol))
        w1.write(mol,confId=int(index))
    w1.close()
    print("Successfully extracting Coordinates... ")
