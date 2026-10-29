# -*- coding:utf-8 -*-
"""Legacy optional utility with dataset-specific paths; use get_data.py for the README workflow."""

import copy
import logging
import glob, os, pickle, random, tqdm
from tokenize_chirality_smiles import Vocabulary
from argparse import ArgumentParser

from rdkit.Chem import Draw, rdmolops
from scipy.optimize import linear_sum_assignment
import sys
sys.path.append("..")
from utils.standardization import *
from multiprocessing import Pool
import csv
import pandas as pd
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
    n_confs =1

    if args.boltzmann == 'top':
        confs = sort_confs(confs)


    if not confs:
        return log_error("no_clean_confs")

    if args.boltzmann == 'resample':
        confs = resample_confs(confs, max_confs=args.confs_per_mol)


    template_rdkit = Chem.MolFromSmiles(name)


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
    if mol_rdkit.GetNumConformers() ==0:
        b=0
        b=b+1
        logger.info(f'wrong: {name}')
        logger.info(b)
    

    if args.mmff:
        try:
            mmff_func(mol_rdkit)
        except:
            return log_error("mmff_error")

    if not args.no_match:
        cost_matrix = [[get_von_mises_rms(template_rdkit_list[i], mol_rdkit, rotable_bonds, j) for j in range(n_confs)] for
                       i in range(n_confs)]
        cost_matrix = np.asarray(cost_matrix)
        cost_matrix=cost_matrix.astype('float64')
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
            _,torsions = optimize_rotatable_bonds_(mol_rdkit_single, mol, rotable_bonds,
                                     popsize=args.popsize, maxiter=args.max_iter)

            for tors in torsions:
                torsion_label_rnd = format(tors, ".2f")
                if torsion_label_rnd=='-0.00':
                    torsion_label_rnd='0.00'
                geos.append(torsion_label_rnd)
        except Exception as e:
            logger.info(e)
            continue
    return geos


def get_best_torsion(mol):
        index,confs=mol
        df = pd.read_csv('../data/RT/Lipophilicity/train.csv')
        labels=df['expt']
        try:

            name = Chem.MolToSmiles(confs, isomericSmiles=True)

            voc = Vocabulary(init_from_file="../data/Voc_charity_SMILES")

            smi = voc.charity_tokenizer(name)
        
            geos = conformer_match(name, confs)
        
            smi.extend(geos)
        except Exception as e:
            logger.info(f'{index}: {e}')
            return None
    
        return index,smi ,labels[index]

if __name__ == '__main__':
    
    logging.basicConfig(level=logging.DEBUG,
                    filename='./logp_train_label_output.log',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    sdf_path ="./sdfs/logp_train_mols.sdf"
    supplier = Chem.SDMolSupplier(sdf_path,removeHs=True, sanitize=True)    
    mols = [(index,x) for (index,x) in enumerate(supplier) if x is not None]
    logger.info(len(mols))
    
    pool = Pool() 
    a = pool.map(get_best_torsion,mols)
    a = list(filter(lambda x: x is not None, a))
    pool.close()
    pool.join()
    csv_file_path = './csvs/logp_train.csv'
    with open(csv_file_path, mode='w', newline='') as file:
          writer = csv.writer(file)
          writer.writerows(a)

