# -*- coding:utf-8 -*-
import copy
import logging
import random
from pathlib import Path

import tqdm
from rdkit import Chem, RDLogger
from tokenize_chirality_smiles import Vocabulary
from argparse import ArgumentParser

from rdkit.Chem import rdmolops, AllChem
from scipy.optimize import linear_sum_assignment
from utils.standardization import (get_torsion_angles, get_von_mises_rms,
                                   mmff_func, optimize_rotatable_bonds_)
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
RDLogger.DisableLog('rdApp.*')
logger = logging.getLogger(__name__)

parser = ArgumentParser()

parser.add_argument('--popsize', type=int, default=15, help='Population size for differential evolution')
parser.add_argument('--max_iter', type=int, default=15, help='Maximum number of iterations for differential evolution')
parser.add_argument('--confs_per_mol', type=int, default=30,
                    help='Maximum number of conformers to take for each molecule')
parser.add_argument('--mmff', action='store_true', default=False,
                    help='Whether to relax seed conformers with MMFF before matching')
parser.add_argument('--no_match', action='store_true', default=False, help='Whether to skip conformer matching')
parser.add_argument('--boltzmann', choices=['top', 'resample'], default=None,
                    help='If set, specifies a different conformer selection policy')

parser.add_argument('--csv_path', type=str, default='../deal_data/ex.csv',
                    help='Input CSV containing smiles and value columns')
parser.add_argument('--sdf_path', type=str, default="./sdfs/ex.sdf",
                    help='Legacy SDF path; unused by the tokenization workflow')
parser.add_argument('--save_path', type=str, default='./csvs/ex_smiles_label_torsion_all.csv',
                    help='Output CSV path')
args = parser.parse_args()


REMOVE_HS = lambda x: Chem.RemoveHs(x, sanitize=False)


def sort_confs(confs):
    return sorted(confs, key=lambda conf: -conf['boltzmannweight'])


def resample_confs(confs, max_confs=None):
    weights = [conf['boltzmannweight'] for conf in confs]
    weights = np.array(weights) / sum(weights)
    k = min(max_confs, len(confs)) if max_confs else len(confs)
    return random.choices(confs, weights, k=k)


def log_error(err):
    logger.info(err)
    return None


average_rsmd = []
long_term_log = []
long_term_rmsd_cache = []


def conformer_match(name, confs):
    n_confs = 1

    if args.boltzmann == 'top':
        confs = sort_confs(confs)


    if not confs:
        return log_error("no_clean_confs")

    if args.boltzmann == 'resample':
        confs = resample_confs(confs, max_confs=args.confs_per_mol)


    template_rdkit = Chem.MolFromSmiles(name)


    # Align atom ordering with the reference molecule.
    reindex = confs.GetSubstructMatch(template_rdkit)
    template_rdkit = copy.deepcopy(confs)
    try:
        template_rdkit = rdmolops.RenumberAtoms(template_rdkit, reindex)
    except:
        logger.info('atomCounts shorter than the number of atoms')

    template_rdkit.RemoveAllConformers()
    template_rdkit_list = template_rdkit


    conf_new = Chem.Conformer(confs.GetNumAtoms())
    positions = confs.GetConformer().GetPositions()
    for rdkit_ind, m_id in enumerate(reindex):
        conf_new.SetAtomPosition(rdkit_ind, positions[m_id])

    template_rdkit_list.AddConformer(conf_new)
    template_rdkit_list = [template_rdkit_list]

    mol_rdkit = copy.deepcopy(template_rdkit)

    rotable_bonds = get_torsion_angles(mol_rdkit)

    if not rotable_bonds:
        logger.info(name)
        return log_error("no_rotable_bonds")

    mol_rdkit.RemoveAllConformers()
    AllChem.EmbedMultipleConfs(mol_rdkit, numConfs=n_confs)
    if mol_rdkit.GetNumConformers() == 0:
        b = 0
        b = b + 1
        logger.info(f'wrong: {name}')
        logger.info(b)


    if args.mmff:
        try:
            mmff_func(mol_rdkit)
        except:
            return log_error("mmff_error")

    if not args.no_match:
        cost_matrix = [[get_von_mises_rms(template_rdkit_list[i], mol_rdkit, rotable_bonds, j) for j in range(n_confs)]
                       for
                       i in range(n_confs)]
        cost_matrix = np.asarray(cost_matrix)
        cost_matrix = cost_matrix.astype('float64')
        try:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
        except:
            logger.info('ValueError: matrix contains invalid numeric entries')

        average_rsmd.extend(cost_matrix.flatten().tolist())

    else:
        row_ind, col_ind = np.arange(len(confs)), np.arange(len(confs))

    iterable = tqdm.tqdm(enumerate(template_rdkit_list), total=n_confs, disable=True)

    geos = []
    for i, template_conf in iterable:
        mol = template_conf
        try:
            conf_id = int(col_ind[i])
        except:
            logger.info('UnboundLocalError')

        try:
            mol_rdkit_single = copy.deepcopy(mol_rdkit)
            [mol_rdkit_single.RemoveConformer(j) for j in range(n_confs) if j != conf_id]
            _, torsions = optimize_rotatable_bonds_(mol_rdkit_single, mol, rotable_bonds,
                                                    popsize=args.popsize, maxiter=args.max_iter)
            for tors in torsions:
                torsion_label_rnd = format(tors, ".2f")
                if torsion_label_rnd == '-0.00':
                    torsion_label_rnd = '0.00'
                geos.append(torsion_label_rnd)
        except Exception as e:
            logger.info(e)
            continue
    return geos


def get_best_torsion(mol, csv_path):
    index, confs = mol
    df = pd.read_csv(csv_path)
    labels = df['value']
    try:

        name = Chem.MolToSmiles(confs, isomericSmiles=True)

        voc = Vocabulary(init_from_file=str(Path(__file__).resolve().with_name("Voc_charity_SMILES")))

        smi = voc.charity_tokenizer(name)

        geos = conformer_match(name, confs)
        smi.extend(geos)
    except Exception as e:
        logger.info(f'{index}: {e}')
        return None

    return index, smi, labels[index]

def get_smiles(csv_path, sdf_path):
    all_smiles_new = []
    all_mols = []

    df = pd.read_csv(csv_path)
    all_smiles = df['smiles']

    inludeChirality = True
    for smiles in tqdm.tqdm(all_smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Cannot parse SMILES: {smiles}")
            continue
        smiles = Chem.MolToSmiles(mol, isomericSmiles=inludeChirality)
        all_smiles_new.append(smiles)
    print("Successfully extracting SMILES... ")

    w1 = Chem.SDWriter(sdf_path)
    for smiles in tqdm.tqdm(all_smiles_new):
        numConfs = 20
        mol = Chem.MolFromSmiles(smiles)
        try:
            mol = Chem.AddHs(mol)
            res = AllChem.EmbedMultipleConfs(mol, numConfs=numConfs)
            # MMFF generates multiple conformations.
            res = AllChem.MMFFOptimizeMoleculeConfs(mol)
            mol = Chem.RemoveHs(mol)
            index = np.argmin([x[1] for x in res])
        except:
            AllChem.Compute2DCoords(mol)
            index = 0
        all_mols.append(Chem.RemoveHs(mol))
        w1.write(mol, confId=int(index))
    w1.close()
    print("Successfully extracting Coordinates... ")

if __name__ == '__main__':

    csv_path = args.csv_path

    df = pd.read_csv(csv_path)
    if not {'smiles', 'value'}.issubset(df.columns):
        raise ValueError("Input CSV must contain smiles and value columns")
    labels = df['value']
    smiles_all = df['smiles']

    label_name = Path(csv_path).absolute().parent.name
    voc = Vocabulary(init_from_file=str(Path(__file__).resolve().with_name("Voc_charity_SMILES")))
    data_no_3d_all=[]
    for i in range(len(smiles_all)):
        smi = voc.charity_tokenizer(smiles_all[i])
        data_no_3d= ' '.join(smi)
        data_no_3d= data_no_3d.strip()
        data_no_3d= data_no_3d +' [UNK] ' + label_name + ' ' + str(labels[i])
        data_no_3d_all.append(data_no_3d)

    torsion_all = data_no_3d_all
    df = pd.DataFrame()
    df['smiles_label_torsion_all'] = torsion_all
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.save_path, index=False)
    print("Total records:", len(df))
