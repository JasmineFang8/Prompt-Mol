import copy
import csv
import re
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
from prompts_nor import PropertyTokenizer, MoleculeDescriptorCalculator, SimilarityFinder, ExtendedSimilarityFinder, \
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
    parser.add_argument('--cat_model_path', default="../cat_models", type=str, help='')
    parser.add_argument('--train_raw_path', default='train_raw_data.txt', type=str, help='')
    parser.add_argument('--extra_raw_path', default='', type=str, help='')
    parser.add_argument('--test_raw_path', default='test_raw_data.txt', type=str, help='')

    parser.add_argument('--epochs', default=1001, type=int, required=False, help='epochs')
    parser.add_argument('--warmup_steps', default=20000, type=int, required=False, help='warm up steps')
    parser.add_argument('--lr', default=5e-3, type=float, required=False, help='learn rate/meta-level outer learning rate')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--log_step', default=10, type=int, required=False, help='print log steps')
    parser.add_argument('--extra_emb_dim', default=400, type=float, required=False, help='extra embedding model dim, for example protein')
    parser.add_argument('--property_name', default='esol', type=str, required=True,
                        help='property need to be predicted')
    parser.add_argument('--k', default=5, type=int, required=False, help='n_support')
    parser.add_argument('--mix_feature', default="rdkit", type=str, required=False, help='choose mix feature')
    return parser.parse_args()

def organize_mixed_dataset(data_list: List[str], mode: str="no_3d"):

    organized_smiles = []
    organized_label = []

    for entry in data_list:
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

def merge_dicts(dict1, dict2):
    merged_dict = {}

    for key, value in dict1.items():
        if key in dict2:
            merged_dict[key] = value + dict2[key]
        else:
            merged_dict[key] = value

    for key, value in dict2.items():
        if key not in merged_dict:
            merged_dict[key] = value

    return merged_dict


def pairs_propmt(finder_similar):
    result_similar = finder_similar.find_top_k_from_extra()
    no_sim_list = []
    s_q_dict = {}
    for query, sim_list in result_similar.items():
        d_list = []
        q_label = property_tokenizer.normalize_to_input(query[1], pp_range[args.property_name][0],
                                                        pp_range[args.property_name][1], 0, 1)
        if sim_list != []:
            for support in sim_list:
                s_property_name = support[1]
                s_label = property_tokenizer.normalize_to_input(support[2], pp_range[s_property_name][0], pp_range[s_property_name][1], 0,1)
                output = '<|beginoftext|> ' + support[0] + ' <' + s_property_name + '> ' + s_label + ' [SEP] ' + query[
                    0] + ' <' + args.property_name + '> <|mask:0|>'
                d_list.append(output)
        else:
            no_sim_list.append(query[0])
        s_q_dict[query[0]] = d_list
    return s_q_dict, no_sim_list

def data_tokenizer(data_i):
    return tokenizer.encode(data_i, truncation=True, max_length=399, return_special_tokens_mask=True,
                            add_special_tokens=False)


def get_diff_label(smiles_list:List[str], target_property: str):
    form_smiles = [''.join(smiles.split()) for smiles in smiles_list]
    train_path = './data/catTrain/' + target_property + '_scaffold/train.csv'
    val_path = './data/catTrain/' + target_property + '_scaffold/valid.csv'
    benchmark_config = {
        'caco2': ('regression', False),
        'lipo': ('regression', False),
        'logp': ('regression', False),
        'solubility': ('regression', False),
        'ppb': ('regression', False),
        'vdss': ('regression', True),
        'half_life': ('regression', True),
        'hum_mic_cl': ('regression', True),
        'ld50': ('regression', False),
        'esol': ('regression', False),
        'freesolv': ('regression', False),
        'water_sol': ('regression', False),
        'cm': ('regression', True),
        'ch': ('regression', True),
        'cyp_2c9': ('regression', True),
        'cyp_2d6': ('regression', True),
        'cyp_3a4': ('regression', True),
        'logd': ('regression', True),
        'mou_mic_cl': ('regression', True),
        'rat_mic_cl': ('regression', True),
    }
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.DataFrame({'smiles': form_smiles})

    train = pd.concat([df_train, df_val])
    test = df_test
    X_train = get_fingerprints(train['smiles'])
    X_test = get_fingerprints(test['smiles'])
    task, log_scale = benchmark_config[target_property]
    params = {
        'random_strength': 2,
        'verbose': 0,
    }

    Y_scaler = scaler(log=log_scale)
    Y_scaler.fit(train['expt'].values)
    train['Y_scale'] = Y_scaler.transform(train['expt'].values)

    params['loss_function'] = 'MAE'
    model = cb.CatBoostRegressor(**params)
    model.fit(X_train, train['Y_scale'].values)
    y_pred = model.predict(X_test)
    y_pred_test = Y_scaler.inverse_transform(y_pred).reshape(-1)
    print("cat pred over!")
    return y_pred_test

def load_model_and_predict(attribute_name, smiles_list:List[str], model_dir="../cat_models"):
    import os
    form_smiles = [''.join(smiles.split()) for smiles in smiles_list]
    df_test = pd.DataFrame({'smiles': form_smiles})
    test = df_test
    X_test = get_fingerprints(test['smiles'])
    # attribute_name = args.train_raw_path.split("/")[-2]
    # attribute_name = args.cat_model_path.split("/")[-1]
    attribute_name = args.property_name

    model_path = os.path.join(model_dir, f"{attribute_name}_model.cbm")
    scaler_path = os.path.join(model_dir, f"{attribute_name}_scaler.pkl")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Model or scaler for attribute '{attribute_name}' not found in {model_dir}")

    model = cb.CatBoostRegressor()
    model.load_model(model_path)
    Y_scaler = scaler.load(scaler_path)

    predictions_scaled = model.predict(X_test)
    predictions = Y_scaler.inverse_transform(predictions_scaled).reshape(-1)
    # print(f"Predictions for {attribute_name}: {predictions}")
    return predictions

def data_to_list(train_smiles, train_label, test_smiles, test_label):
    prompts = {}
    no_sim_count = {}
    re_indx = [] #[94, 124]
    for idx, query in enumerate(test_smiles):
        data_list = []
        if query in prompts:
            re_indx.append(idx)
            continue
        prompts[query] = data_list

    Target_name = args.property_name
    # TrainTargetLabel = get_diff_label(train_smiles, Target_name) #[float,float,, ]
    # TrainTargetLabel = [(Target_name, l) for l in TrainTargetLabel]
    # TestTargetLabel = get_diff_label(test_smiles, Target_name) #[float,float,, ]
    if train_label[0][0] == Target_name:
        TrainTargetLabel = train_label
    else:
        TrainTargetLabel = load_model_and_predict(Target_name, train_smiles, args.cat_model_path)
        TrainTargetLabel = [(Target_name, l) for l in TrainTargetLabel]
    TestTargetLabel = load_model_and_predict(Target_name, test_smiles, args.cat_model_path)

    finder_SimLabel = ExtendedSimLabelFinder(test_smiles, TestTargetLabel,train_smiles, TrainTargetLabel,
                                             top_k = args.k, label_diff_threshold = 0.1, mode = 'sim_property')
    simLabel_property, no_sim = pairs_propmt(finder_SimLabel)
    no_sim_count["no simLabel"] = len(no_sim)
    prompts = merge_dicts(prompts, simLabel_property)

    finder_tanimoto = ExtendedSimilarityFinder(test_smiles, TestTargetLabel, train_smiles, TrainTargetLabel , top_k=args.k,
                                              similarity_method = 'tanimoto')
    finder_descriptor = ExtendedSimilarityFinder(test_smiles, TestTargetLabel, train_smiles, TrainTargetLabel , top_k=args.k,
                                              similarity_method = 'descriptor')
    tanimoto_property, no_sim = pairs_propmt(finder_tanimoto)
    no_sim_count["no tanimoto"] = len(no_sim)
    prompts = merge_dicts(prompts, tanimoto_property)
    descriptor_property, no_sim = pairs_propmt(finder_descriptor)
    no_sim_count["no descriptor"] = len(no_sim)
    prompts = merge_dicts(prompts, descriptor_property)

    print("nan support：", no_sim_count)

    updated_prompts = {key: list(map(data_tokenizer, value)) for key, value in prompts.items()}
    print(next(iter(prompts.items())))
    return updated_prompts, prompts, re_indx

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

def predict(model, DataLoader, p_text, true_label):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()
    pred_label = []
    max_length = 1
    with torch.no_grad():
        for i, data in enumerate(DataLoader):
            input_ids, extra_input = data
            input_ids = input_ids.to(device)
            extra_input = extra_input.to(device)
            # inputs = {"input_ids": batch.to(device)}
            response = [] 
            out_prompt = p_text[i]
            t_label = property_tokenizer.normalize_to_input(true_label, pp_range[args.property_name][0], pp_range[args.property_name][1], 0, 1)
            for _ in range(max_length):
                i = 0
                outputs = model(input_ids, extra_input)
                logits = outputs.logits
                logits /= 1.2
                next_token_logits = logits[0][-1, :]
               
                next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
                _, next_token = F.softmax(next_token_logits).max(dim=0)
                
                if next_token == tokenizer.sep_token_id: 
                    if response != []:
                        break

                    input_ids = torch.cat((input_ids, torch.IntTensor([tokenizer.vocab['<'+args.property_name+'>']]).unsqueeze(1).to(device)),
                                          1)
                    continue
                if len(input_ids) == 400:
                    print("超出长度,第{}条".format(i + 1), tokenizer.decode(input_ids))
                    break
                response.append(next_token.item())
    
                next_token = next_token.unsqueeze(0).unsqueeze(1)
                input_ids = torch.cat((input_ids, next_token.detach()), 1)
            text = tokenizer.convert_ids_to_tokens(response)
            # print(text)
            if args.save_results != "":
                with open(output_file, mode='a', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow([query, out_prompt, text, t_label])
            pred_label.append(property_tokenizer.normalize_to_real(text, pp_range[args.property_name][0], pp_range[args.property_name][1]))
        print(pred_label)
        pred_label = list(filter(lambda x: x != -1, pred_label))
        if pred_label == []:
            pred_label = [-1]
        var = np.var(pred_label)
        if var > 0.4 and len(pred_label)>2:
            pred_label = remove_outliers_iqr(pred_label)
        return sum(pred_label) / len(pred_label), var

def remove_outliers_iqr(data):
    data_array = np.array(data)
    Q1 = np.percentile(data_array, 25)  
    Q3 = np.percentile(data_array, 75) 
    IQR = Q3 - Q1 
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = [x for x in data if lower_bound <= x <= upper_bound]
    return filtered_data

def evaluation(true_label, pred_label):
    rmse = np.sqrt(metrics.mean_squared_error(true_label, pred_label))
    mae = metrics.mean_absolute_error(true_label, pred_label)
    spearman = spearmanr(true_label, pred_label)[0]
    return rmse, mae, spearman

def remove_indices(original_list, indices_to_remove):
    result_list = original_list[:]
    indices_to_remove = sorted(set(indices_to_remove), reverse=True)
    for index in indices_to_remove:
        if 0 <= index < len(result_list):
            del result_list[index]

    return result_list

def mix_extra_embedding(init_data_list, t_data_list):
    new_list = []
    calculator = MolecularDescriptorCalculator(get_chosen_descriptors_model())
    for idx, data in enumerate(init_data_list):
        matchq = re.search(r'\[SEP\]\s(.*?)\s<', data)
        matchs = re.search(r'<\|beginoftext\|>\s(.*?)\s<', data)

        data_ = [t_data_list[idx]]

        if matchq and matchs:
            smiles_q = matchq.group(1)
            smiles_s = matchs.group(1)
            formatted_smiles_q = ''.join(smiles_q.split())
            formatted_smiles_s = ''.join(smiles_s.split())
            molecule_q = Chem.MolFromSmiles(formatted_smiles_q)
            molecule_s = Chem.MolFromSmiles(formatted_smiles_s)
            if args.mix_feature != "rdkit":
                extra_matrix_q = calculator.CalcDescriptors(molecule_q)
                extra_matrix_s = calculator.CalcDescriptors(molecule_s)
            else:
                extra_matrix_q = rdkit_2d_normalized_features_generator(molecule_q)
                extra_matrix_s = rdkit_2d_normalized_features_generator(molecule_s)
            # extra_matrix = np.concatenate((extra_matrix_s, extra_matrix_q),axis=0)
            extra_matrix = np.concatenate((extra_matrix_q, extra_matrix_s),axis=0)
        data_.append(list(extra_matrix))
        new_list.append(data_)
    return new_list

if __name__ == '__main__':
    args = setup_args()
    print(args.model_path)
    column = 'smiles_label_torsion_all'
    tokenizer = ExpressionBertTokenizer.from_pretrained(args.vocab_path)
    property_tokenizer = PropertyTokenizer()
    tokenizer.sep_token_id = tokenizer.vocab['<|endofmask|>']
    # args.batch_size = 8
    if args.extra_raw_path!='':
        data1 = pd.read_csv(args.train_raw_path)
        data2 = pd.read_csv(args.extra_raw_path)
        train_data = pd.concat([data1,data2], ignore_index=True, join="inner").values.flatten().tolist()
    else:
        train_data = pd.read_csv(args.train_raw_path, usecols=[column]).values.flatten().tolist()

    test_data = pd.read_csv(args.test_raw_path, usecols=[column]).values.flatten().tolist()
    train_smiles, train_label = organize_mixed_dataset(train_data) #[smiles,], [(str,label),]
    #print(train_smiles[0],train_label[0])
    test_smiles, test_label = organize_mixed_dataset(test_data)
    true_label = [value[1] for value in test_label]
    test_prompt_dict, prompts, re_indx = data_to_list(train_smiles, train_label, test_smiles, true_label)
    print(re_indx)
    true_label = remove_indices(true_label, re_indx)
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

    if args.save_results != '':
        with open(output_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Object', 'Prompt', 'Prediction'])
    count = 0
    print("len:", len(true_label), len(test_prompt_dict))
    for query, support in tqdm(test_prompt_dict.items()):
        test_data = mix_extra_embedding(prompts[query], support)
        test_dataset = MyDataset(test_data)
        testDataLoader = torch.utils.data.DataLoader(test_dataset, shuffle=False, pin_memory=False, collate_fn=collate_fn)
        s_pred_label, var = predict(model, testDataLoader, prompts[query], true_label[count])
        pred_list.append(s_pred_label)
        var_list.append(var)
        count = count+1

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters {num_params} of type {type(model)}")
    print(true_label, '\n', pred_list, '\n', var_list)
    rmse, mae, spearman = evaluation(true_label, pred_list)
    print(args.property_name)
    print("val_rmse: {},".format(rmse),
          "val_mae: {}.".format(mae),
          "val_spearman: {}.".format(spearman))