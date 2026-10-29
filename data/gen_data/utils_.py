import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem


def Variable(tensor):
    """Wrapper for torch.autograd.Variable that also accepts
       numpy arrays directly and automatically assigns it to
       the GPU. Be aware in case some operations are better
       left to the CPU."""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if torch.cuda.is_available():
        return torch.autograd.Variable(tensor).to(device)
    return torch.autograd.Variable(tensor)


def decrease_learning_rate(optimizer, decrease_by=0.01):
    """Multiplies the learning rate of the optimizer by 1 - decrease_by"""
    for param_group in optimizer.param_groups:
        param_group['lr'] *= (1 - decrease_by)


def seq_to_smiles(seqs, voc):
    """Takes an output sequence from the RNN and returns the
       corresponding SMILES."""
    smiles = []
    for seq in seqs:
        smiles.append(voc.decode(seq))
    return smiles


def fraction_valid_smiles(smiles):
    """Takes a list of SMILES and returns fraction valid."""
    i = 0
    for smile in smiles:
        if Chem.MolFromSmiles(smile):
            i += 1
    return i / len(smiles)


def unique(arr):
    # Finds unique rows in arr and return their indices
    arr = arr.cpu().numpy()
    arr_ = np.ascontiguousarray(arr).view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))
    _, idxs = np.unique(arr_, return_index=True)
    if torch.cuda.is_available():
        return torch.LongTensor(np.sort(idxs)).cuda()
    return torch.LongTensor(np.sort(idxs))


def check_chemical_validity(mol):
    """
    Checks the chemical validity of the mol object. Existing mol object is
    not modified. Radicals pass this test.

    Args:
        mol: Rdkit mol object

    :rtype:
        :class:`bool`, True if chemically valid, False otherwise
    """

    m = Chem.MolFromSmiles(mol)
    if m:
        return True
    else:
        return False


def check_valency(mol):
    """
    Checks that no atoms in the mol have exceeded their possible
    valency.

    Args:
        mol: Rdkit mol object

    :rtype:
        :class:`bool`, True if no valency issues, False otherwise
    """

    try:
        s = Chem.MolToSmiles(mol, isomericSmiles=True)
        m = Chem.MolFromSmiles(s)
        Chem.SanitizeMol(m, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True
    except:
        return False


def denovo_decode(matrix, EOM):
    """Takes an array of indices and returns the corresponding SMILES"""
    geo_chars = []

    matrix = matrix.split('EOS')
    smiles = matrix[0][37:]

    smiles = smiles.replace("L", "Cl").replace("R", "Br").replace(' ', '')

    geo = matrix[1].split(' ')

    for i in geo:
        if i == EOM:
            break
        geo_chars.append(i)

    return smiles, geo_chars[1:]


def convert_to_mol(smiles, coords):

    mol = Chem.MolFromSmiles(smiles)
    coords = np.array(coords).reshape(-1, 3).astype(float)


    AllChem.EmbedMolecule(mol)

    conf = mol.GetConformer()

    # Set the conformer coordinates.
    for i, coord in enumerate(coords):
        conf.SetAtomPosition(i, coord)


    mol.AddConformer(conf)


    sdf_block = Chem.MolToMolBlock(mol)

    # Append the molecule to the SDF output.
    with open('output.sdf', 'a') as f:
        f.write(sdf_block)
        f.write('$$$$')
        f.write('\n')
