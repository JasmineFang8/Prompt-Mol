import copy
import csv
import re
import pickle
from typing import List, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import catboost as cb
import argparse
from collections import defaultdict
from tqdm.auto import tqdm

import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from torch.optim import AdamW, Adam
from transformers import get_scheduler, GPT2Config
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from torch import distributions
import torch.nn.functional as F

from GPT_meta import PromptGPT
from bert_tokenizer import ExpressionBertTokenizer
from maplight import get_fingerprints, scaler
from prompts_nor_1 import PropertyTokenizer, MoleculeDescriptorCalculator, SimilarityFinder, ExtendedSimilarityFinder, \
    ExtendedSimLabelFinder
from transformers.models.gpt2 import GPT2LMHeadModel

from early_stop.pytorchtools import EarlyStopping
from sklearn import metrics
from scipy.stats import spearmanr

from rdkit import Chem, DataStructs
from maplight import get_rdkit_features, get_chosen_descriptors_model, get_chosen_descriptors,rdkit_2d_normalized_features_generator
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from all_range import pp_diff_threshold, pp_diff, pp_des, pp_range

class MyDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __getitem__(self, index):
        input_ids = self.data_list[index]
        return input_ids

    def __len__(self):
        return len(self.data_list)


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='../RTM_torsion_countinue_v2_epoch7_400/', type=str, help='')
    parser.add_argument('--vocab_path', default='./data/torsion_voc_gc.csv', type=str, help='')
    parser.add_argument('--save_results', default="", type=str, help='')
    parser.add_argument('--train_raw_path', default='train_raw_data.txt', type=str, help='')
    parser.add_argument('--extra_raw_path', default='', type=str, help='')
    parser.add_argument('--test_raw_path', default='test_raw_data.txt', type=str, help='')

    parser.add_argument('--warmup_steps', default=20000, type=int, required=False, help='warm up steps')
    parser.add_argument('--lr', default=5e-3, type=float, required=False, help='learn rate/meta-level outer learning rate')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--log_step', default=10, type=int, required=False, help='print log steps')
    parser.add_argument('--extra_emb_dim', default=256, type=float, required=False, help='extra embedding model dim, for example protein')
    parser.add_argument('--k', default=5, type=int, required=False, help='n_support')
    return parser.parse_args()

def organize_mixed_dataset(data_list: List[str], mode: str="no_3d"):
    organized_smiles = []
    organized_label = []

    for entry in tqdm(data_list):
        try:
            parts = entry.split()
            if mode == "3d":
                smiles = " ".join(parts[:-2])
            else:
                smiles = " ".join(parts[:parts.index("GEO")]) 
            property_name = parts[-2] 
            property_value = float(parts[-1])  
            organized_smiles.append(smiles)
            organized_label.append((property_name, property_value))
        except (IndexError, ValueError):
            print(f"格式错误，跳过此条目: {entry}")
    return organized_smiles, organized_label


def pairs_propmt(finder_similar):
    result_similar = finder_similar.find_top_k_from_extra() 
    s_q_list = []
    for query, _, sim_list in result_similar:
        d_list = []
        if sim_list != []:
            for support in sim_list:
                s_property_name = support[1]
                s_label = property_tokenizer.normalize_to_input(support[2], pp_range[s_property_name][0], pp_range[s_property_name][1], 0,1)
                output = '<|beginoftext|> ' + support[0] + ' <' + s_property_name + '> ' + s_label + ' [SEP] <|mask:0|> <|mask:0|>'
                d_list.append(output)
        else:
            no_sim_list.append(query)
        s_q_list.append(d_list)
    return s_q_list, no_sim_list

def data_tokenizer(data_i):
    return tokenizer.encode(data_i, truncation=True, max_length=399, return_special_tokens_mask=True,
                            add_special_tokens=False)

def merge_2d_lists(list1, list2):
    return [row1 + row2 for row1, row2 in zip(list1, list2)]

def data_to_list(train_mol, train_label, extra_smiles, extra_labels):
    no_sim_count = {}
    mol_labels = [mol[1] for mol in train_label]

    finder_tanimoto = ExtendedSimilarityFinder(train_mol, mol_labels, extra_smiles, extra_labels , top_k=args.k,
                                              similarity_method = 'tanimoto')
    tanimoto_property, no_sim = pairs_propmt(finder_tanimoto)
    no_sim_count["no tanimoto"] = len(no_sim)
    finder_descriptor = ExtendedSimilarityFinder(train_mol, mol_labels, extra_smiles, extra_labels , top_k=1,
                                              similarity_method = 'descriptor', output_file="des.pkl")
    descriptor_property, no_sim = pairs_propmt(finder_descriptor)
    no_sim_count["no descriptor"] = len(no_sim)
    updated_list = merge_2d_lists(tanimoto_property, descriptor_property)

    print("!!!",len(updated_list))
    print("nan support：", no_sim_count)
    print(updated_list[7])
    
    return updated_list

def collate_fn(mix_batch):
    batch, des_batch = list(zip(*mix_batch))

    # Convert des_batch to tensor and expand dimensions
    des_batch_tensor = torch.tensor(des_batch, dtype=torch.float32)  # [batch_size, feature_dim]
    des_batch = des_batch_tensor.unsqueeze(1)  # [batch_size, 1, feature_dim]
    input_ids = []

    input_lens_list = [len(w) for w in batch]
    max_input_len = max(input_lens_list)

    for btc_idx in range(len(batch)):
        input_len = len(batch[btc_idx])
        input_ids.append(batch[btc_idx])
        input_ids[btc_idx].extend([tokenizer.pad_token_id] * (max_input_len - input_len))

    # Padding des_batch directly using PyTorch
    max_des_len = des_batch.size(1)
    des_ids = torch.zeros((len(des_batch), max_des_len, des_batch.size(2)), dtype=torch.float32)
    des_ids[:, :max_des_len, :] = des_batch

    return torch.tensor(input_ids, dtype=torch.long), des_ids

@torch.no_grad()
def predict(model, tokenizer, batch_size, single_pocket,
            text="<|beginoftext|> <|mask:0|> <|mask:0|>"):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # model, _ = load_model(args.save_model_path, args.vocab_path)
    protein_batch = single_pocket
    model.to(device)
    model.eval()
    max_length = 195
    input_ids = []
    input_ids.extend(tokenizer.encode(text, truncation=True, max_length=399, return_special_tokens_mask=True,
                            add_special_tokens=False))
    input_length = len(input_ids)

    input_tensor = torch.zeros(batch_size, input_length).long()
    input_tensor[:] = torch.tensor(input_ids)

    Seq_list = []

    # finished = torch.zeros(batch_size, 1).byte().to(device)
    finished = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
    eos_token_id = tokenizer.encode('<|endofmask|>', add_special_tokens=False)[0]

    protein_batch = torch.tensor(protein_batch, dtype=torch.float32)
    protein_batch = protein_batch.to(device)
    protein_batch = protein_batch.repeat(batch_size, 1, 1)
    for i in range(max_length):
        inputs = input_tensor.to(device)
        outputs = model(inputs, protein_batch)
        logits = outputs.logits
        logits = F.softmax(logits[:, -1, :], dim=1)

        last_token_id = torch.multinomial(logits, 1)
        EOS_sampled = (last_token_id.squeeze(-1) == eos_token_id).unsqueeze(-1)
        finished = finished | EOS_sampled  # element-wise OR
        if finished.all():
            print("End")
            break

        last_token = tokenizer.convert_ids_to_tokens(last_token_id)
        input_tensor = torch.cat((input_tensor, last_token_id.detach().to('cpu')), 1)

        Seq_list.append(last_token)
    Seq_list = np.array(Seq_list).T

    return Seq_list

def decode(matrix):
    chars = []
    for i in matrix:
        if i == '<|endofmask|>' or i == '<SA>' : break
        chars.append(i)
    seq = " ".join(chars)
    return seq


def read_data(path):
    data = []
    with open(path, 'rb') as f:
        while True:
            try:
                aa = pickle.load(f)
                data.extend(aa)
            except EOFError:
                break
    return data


if __name__ == '__main__':
    args = setup_args()
    print(args.model_path)
    column = 'smiles_label_torsion_all'
    tokenizer = ExpressionBertTokenizer.from_pretrained(args.vocab_path)
    property_tokenizer = PropertyTokenizer()
    tokenizer.sep_token_id = tokenizer.vocab['<|endofmask|>']
    # args.batch_size = 8

    protein_matrix = read_data('../data/protein/test_protein_represent.pkl')

    if args.extra_raw_path!='':
        data1 = pd.read_csv(args.train_raw_path)
        data2 = pd.read_csv(args.extra_raw_path)
        train_data = pd.concat([data1,data2], ignore_index=True, join="inner").values.flatten().tolist()
    else:
        train_data = pd.read_csv(args.train_raw_path, usecols=[column]).values.flatten().tolist()

    test_data = pd.read_csv(args.test_raw_path, usecols=[column]).values.flatten().tolist()
    train_smiles, train_label = organize_mixed_dataset(train_data) #[smiles,], [(str,label),]

    test_smiles, test_label = organize_mixed_dataset(test_data)
 
    test_data_list = data_to_list(test_smiles, test_label,train_smiles, train_label)
    
    Ada_config = GPT2Config(
        architectures=["GPT2LMHeadModel"],
        model_type="GPT2LMHeadModel",
        vocab_size=836,
        n_positions=400,
        n_ctx=400,  # max length
        n_embd=768,
        n_layer=12,
        n_head=8,

        task_specific_params={
            "text-generation": {
                "do_sample": True,
                "max_length": 400
            }
        }
    )
    model = PromptGPT(pretrain_path='../RTM_torsion_countinue_v2_epoch7_400/', config=Ada_config, emb_dim=args.extra_emb_dim)
    param_dict = {key.replace("module.", ""): value for key, value in torch.load(args.model_path, map_location='cuda').items()}
    model.load_state_dict(param_dict)
    pred_list = []
    var_list = []
    output_file = args.save_results

    all_output = []
    # Total number = range * batch size
    for idx, pocket in tqdm(enumerate(protein_matrix)):
        one_output = []
        Seq_all = []
        if test_data_list[idx] != []:
            for i in test_data_list[idx]:
                Seq_list = predict(model, tokenizer, single_pocket=pocket, batch_size=25, text=i)
                Seq_all.extend(Seq_list)
            for j in Seq_all:
                print(j)
                one_output.append(decode(j))
            all_output.append(one_output)

    print(f'Genetion finished! Raw seqs saves at {args.save_results}')
    output = pd.DataFrame(all_output)

    output.to_csv(args.save_results, index=False, header=False, mode='a')

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters {num_params} of type {type(model)}")
