#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import math
import os
import pickle
import random
import re
import sys
from collections import defaultdict
from pathlib import Path


def setup_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--task', choices=['property', 'protein_generation'], default='property')
    parser.add_argument('--code_dir', default=str(Path(__file__).resolve().parent),
                        help='模块所在目录；默认本文件所在目录')
    parser.add_argument('--model_path', default=None, help='预训练 GPT 目录；默认 ../RTM_torsion_countinue_v2_epoch7_400')
    parser.add_argument('--vocab_path', default=None, help='默认 prompt_reg_voc.csv')
    parser.add_argument('--save_model_path', default='save_model_unified')
    parser.add_argument('--train_raw_path', required=True)
    parser.add_argument('--valid_raw_path', required=True)
    parser.add_argument('--extra_raw_path', default=None, help='蛋白任务的独立 support 分子库 CSV')
    parser.add_argument('--train_protein_path', default=None)
    parser.add_argument('--valid_protein_path', default=None)
    parser.add_argument('--column', default='smiles_label_torsion_all')
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--epochs', default=1001, type=int)
    parser.add_argument('--warmup_steps', default=20000, type=int)
    parser.add_argument('--lr', default=5e-3, type=float)
    parser.add_argument('--max_grad_norm', default=1.0, type=float)
    parser.add_argument('--log_step', default=10, type=int)
    parser.add_argument('--k', default=5, type=int)
    parser.add_argument('--threshold', default=0.0, type=float,
                        help='保留原词表 750..756 属性名 token 的附加监督权重')
    parser.add_argument('--early_patience', default=30, type=int)
    parser.add_argument('--extra_emb_dim', default=None, type=int, help='默认自动推断；仅调整维度，不选择特征类型')
    parser.add_argument('--mix_feature', default='rdkit', choices=['rdkit', 'des'], help='仅 property 使用')
    parser.add_argument('--lambda_attn', default=0.0, type=float, help='仅 property 使用')
    parser.add_argument('--lambda_alig', default=None, type=float,
                        help='property 默认 0.2；protein_generation 默认 0，且必须为 0')
    parser.add_argument('--max_length', default=399, type=int, help='文本长度上限；同时检查文本+额外特征不超过 400')
    parser.add_argument('--overlength', choices=['error', 'skip'], default='error',
                        help='过长样本报错或整条跳过；不会截断目标')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--prepare_only', action='store_true')
    opts = parser.parse_args(argv)
    if opts.lambda_alig is None:
        opts.lambda_alig = 0.2 if opts.task == 'property' else 0.0
    if opts.task == 'protein_generation':
        for name in ('extra_raw_path', 'train_protein_path', 'valid_protein_path'):
            if not getattr(opts, name):
                parser.error('--task protein_generation 必须指定 --' + name)
        if opts.lambda_attn != 0 or opts.lambda_alig != 0:
            parser.error('蛋白模式不使用属性预测专用的 attention/alignment loss，请将两项权重设为 0')
    for name in ('batch_size', 'epochs', 'log_step', 'k', 'early_patience', 'max_length'):
        if getattr(opts, name) <= 0:
            parser.error('--' + name + ' 必须大于 0')
    if opts.max_length > 399:
        parser.error('模型位置上限为 400，--max_length 必须 <= 399')
    if opts.extra_emb_dim is not None and opts.extra_emb_dim <= 0:
        parser.error('--extra_emb_dim 必须大于 0')
    if opts.lr <= 0 or opts.max_grad_norm <= 0 or opts.warmup_steps < 0:
        parser.error('lr/max_grad_norm 必须为正，warmup_steps 不能为负')
    if min(opts.threshold, opts.lambda_attn, opts.lambda_alig) < 0:
        parser.error('损失权重不能为负')
    root = Path(opts.code_dir).expanduser().resolve()
    opts.code_dir = str(root)
    opts.model_path = opts.model_path or str(root.parent / 'RTM_torsion_countinue_v2_epoch7_400')
    opts.vocab_path = opts.vocab_path or str(root / 'prompt_reg_voc.csv')
    return opts


def load_dependencies(code_dir):
    root = Path(code_dir)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root))
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from torch.optim import AdamW
    from transformers import GPT2Config, get_scheduler
    from rdkit import Chem
    from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
    from tqdm.auto import tqdm
    from GPT_meta import PromptGPT
    from bert_tokenizer import ExpressionBertTokenizer
    from prompts_nor import PropertyTokenizer, SimilarityFinder, SimLabelFinder
    from prompts_nor import ExtendedSimilarityFinder
    from maplight import get_chosen_descriptors_model, rdkit_2d_normalized_features_generator
    from all_range import pp_diff_threshold, pp_diff, pp_range
    globals().update({key: value for key, value in locals().items() if key not in ('root', 'name', 'code_dir')})


class MyDataset:
    def __init__(self, data_list):
        self.data_list = data_list

    def __getitem__(self, index):
        return self.data_list[index]

    def __len__(self):
        return len(self.data_list)


def _property_label(value, property_name):
    if property_name not in pp_range:
        raise ValueError(f'属性 {property_name!r} 未在 all_range.pp_range 中定义')
    low, high = pp_range[property_name]
    label = property_tokenizer.normalize_to_input(value, low, high, 0, 1)
    return label


def organize_mixed_dataset(data_list, mode='no_3d'):
    organized_data = defaultdict(lambda: ([], []))
    smiles_list, labels = [], []
    for index, entry in enumerate(data_list):
        try:
            parts = entry.split()
            smiles = ' '.join(parts[:-2] if mode == '3d' else parts[:parts.index('GEO')])
            property_name, value = parts[-2], float(parts[-1])

            _property_label(value, property_name)
        except (IndexError, ValueError):
            print(f"格式错误，跳过此条目: {entry}")
        if args.task == 'property':
            if property_name not in pp_diff or property_name not in pp_diff_threshold:
                raise ValueError(f'{property_name} 缺少原配对阈值配置')
            organized_data[property_name][0].append(smiles)
            organized_data[property_name][1].append(value)
        else:
            smiles_list.append(smiles)
            labels.append((property_name, value))
    return organized_data if args.task == 'property' else (smiles_list, labels)


def pairs_propmt(pp_data, property_name, mode, top_k=5):
    if mode == 'sim_property':
        finder = SimLabelFinder(pp_data[0], pp_data[1], top_k=top_k,
                                label_diff_threshold=pp_diff_threshold[property_name], mode=mode)
    else:
        finder = SimilarityFinder(pp_data[0], pp_data[1], top_k=top_k,
                                  label_diff_threshold=pp_diff[property_name], mode=mode)
    result_similar = finder.find_molecules()
    d_list, no_sim_list = [], []
    for query, sim_list in result_similar.items():
        q_label = _property_label(query[1], property_name)
        if not sim_list:
            no_sim_list.append(query[0])
        for support in sim_list:
            s_label = _property_label(support[1], property_name)
            d_list.append(f'<|beginoftext|> {support[0]} <{property_name}> {s_label} '
                          f'[SEP] {query[0]} <{property_name}> <|mask:0|> {q_label} <|endofmask|>')
    return d_list, no_sim_list


def pairs_propmt_protein(finder_similar, query_labels):

    result_similar = finder_similar.find_top_k_from_extra()
    s_q_list, no_sim_list = [], []
    for (query, _, sim_list), (property_name, query_value) in zip(result_similar, query_labels):
        q_label = _property_label(query_value, property_name)
        d_list = []
        for support, s_property, s_value in sim_list:
            s_label = _property_label(s_value, s_property)
            d_list.append(f'<|beginoftext|> {support} <{s_property}> {s_label} '
                          f'[SEP] <|mask:0|> <|mask:0|> {query} <{property_name}> '
                          f'{q_label} <|endofmask|>')
        if not d_list:
            no_sim_list.append(query)
        s_q_list.append(d_list)
    return s_q_list, no_sim_list


def data_tokenizer(data_i):
    return tokenizer.encode(data_i, truncation=True, max_length=399, return_special_tokens_mask=True,
                            add_special_tokens=False)


def data_to_list(data, train_label=None, extra_smiles=None, extra_labels=None, matrix_protein=None):

    if args.task == 'protein_generation':
        if len(data) != len(matrix_protein) or len(data) != len(train_label):
            raise ValueError('CSV 分子条数、标签条数和蛋白矩阵条数必须完全一致，并按行对应')
        finder = ExtendedSimilarityFinder(data, [value for _, value in train_label],
                                           extra_smiles, extra_labels, top_k=args.k,
                                           similarity_method='tanimoto')
        prompts, no_sim = pairs_propmt_protein(finder, train_label)
        data_list, skipped = [], 0
        for index, prompt_list in enumerate(prompts):
            protein = np.asarray(matrix_protein[index], dtype=np.float32)
            if protein.ndim != 2 or min(protein.shape) <= 0 or not np.isfinite(protein).all():
                raise ValueError(f'第 {index + 1} 条蛋白矩阵必须是非空、有限的 [L, D] 数组')
            for text in prompt_list:
                ids = data_tokenizer(text)
                if ids is None:
                    skipped += 1
                else:
                    data_list.append([ids, protein])
        print(f'protein 提示数 {len(data_list)}，无 support 的 query {len(no_sim)}，过长跳过 {skipped}')
        return data_list

    data_list, no_sim_count = [], {}
    for property_name, pp_data in data.items():
        for smiles, value in zip(*pp_data):
            label = _property_label(value, property_name)
            data_list.append(f'<|beginoftext|> {smiles} <{property_name}> {label} '
                             f'[SEP] {smiles} <{property_name}> <|mask:0|> {label} <|endofmask|>')
        for mode, top_k in [('sim_property', args.k), ('similar_property', args.k * 2),
                            ('dissimilar_property', args.k)]:
            prompts, no_sim = pairs_propmt(pp_data, property_name, mode, top_k)
            data_list.extend(prompts)
            no_sim_count[f'{property_name}/{mode}'] = len(no_sim)
    tokenized, texts = [], []
    for text in data_list:
        ids = data_tokenizer(text)
        if ids is not None:
            print(text)
            tokenized.append(ids)
            texts.append(text)
    print(f'property 提示数 {len(texts)}，过长跳过 {len(data_list) - len(texts)}；无 support：{no_sim_count}')
    return tokenized, texts


def mix_extra_embedding(init_data_list, t_data_list):
    calculator = MolecularDescriptorCalculator(get_chosen_descriptors_model())
    new_list, cache = [], {}
    for text, ids in zip(init_data_list, t_data_list):
        matchq = re.search(r'\[SEP\]\s(.*?)\s<', text)
        matchs = re.search(r'<\|beginoftext\|>\s(.*?)\s<', text)
        if not matchq or not matchs:
            raise ValueError('无法从 property 提示解析 query/support 分子')
        features = []
        for match in (matchq, matchs):
            smiles = ''.join(match.group(1).split())
            if smiles not in cache:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    raise ValueError(f'无效 SMILES：{smiles}')
                values = (rdkit_2d_normalized_features_generator(mol) if args.mix_feature == 'rdkit'
                          else calculator.CalcDescriptors(mol))
                cache[smiles] = np.asarray(values, dtype=np.float32)
            features.append(cache[smiles])
        extra = np.concatenate(features).copy()
        finite = extra[np.isfinite(extra)]
        extra[np.isinf(extra)] = np.max(finite) if finite.size else 0.0
        extra = np.nan_to_num(extra, nan=0.0)
        new_list.append([ids, extra])
    return new_list


def collate_fn(mix_batch):
    batch, extra_batch = zip(*mix_batch)
    max_input_len = max(map(len, batch))
    input_ids = [list(ids) + [tokenizer.pad_token_id] * (max_input_len - len(ids)) for ids in batch]
    matrices = [np.asarray(x, dtype=np.float32) for x in extra_batch]
    matrices = [x[None, :] if x.ndim == 1 else x for x in matrices]
    if any(x.ndim != 2 or x.shape[1] != args.extra_emb_dim for x in matrices):
        raise ValueError('batch 中额外特征维度与 extra_emb_dim 不匹配')
    lengths = [len(x) for x in matrices]
    extra_ids = np.zeros((len(batch), max(lengths), args.extra_emb_dim), dtype=np.float32)
    for index, matrix in enumerate(matrices):
        extra_ids[index, :len(matrix)] = matrix
    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(extra_ids), lengths


def forward_batch(model, batch, extra_batch, extra_lengths):

    from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
    if len(set(extra_lengths)) == 1:
        return model(batch, extra_batch[:, :extra_lengths[0]])

    groups = defaultdict(list)
    for index, length in enumerate(extra_lengths):
        groups[length].append(index)
    logits, indices = [], []
    for length, rows in groups.items():
        row_ids = torch.tensor(rows, device=batch.device)
        text = batch.index_select(0, row_ids)
        text_length = int(text.ne(tokenizer.pad_token_id).sum(dim=1).max().item())
        out = model(text[:, :text_length], extra_batch.index_select(0, row_ids)[:, :length])
        logits.append(F.pad(out.logits, (0, 0, 0, batch.size(1) - text_length)))
        indices.extend(rows)
    order = torch.tensor(indices, device=batch.device).argsort()
    return CausalLMOutputWithCrossAttentions(logits=torch.cat(logits).index_select(0, order))


def get_all_normal_dis_pdf(voc_len=None, confs_num=None):
 
    voc_len = len(tokenizer) if voc_len is None else voc_len
    pdf_array = torch.eye(voc_len)
    numeric_ids = sorted((idx for tok, idx in tokenizer.vocab.items() if re.fullmatch(r'p_\d+\.\d+', tok)),
                         key=lambda idx: float(tokenizer.convert_ids_to_tokens(idx)[2:]))
    if not numeric_ids:
        raise ValueError('词表缺少 p_ 数值 token')
    for index, token_id in enumerate(numeric_ids):
        weights = torch.exp(-0.5 * ((torch.arange(len(numeric_ids)).float() - index) / 2.0) ** 2)
        weights[index] *= 2
        pdf_array[token_id].zero_()
        pdf_array[token_id, numeric_ids] = weights / weights.sum()
    return pdf_array


def _loss_masks(labels, generate_molecule=False):
    shift_labels = labels[:, 1:]
    numeric = torch.zeros_like(shift_labels, dtype=torch.bool)
    for tok, token_id in tokenizer.vocab.items():
        if re.fullmatch(r'p_\d+\.\d+', tok):
            numeric |= shift_labels.eq(token_id)
    main_mask = numeric | shift_labels.eq(tokenizer.vocab['<|endofmask|>'])
    if generate_molecule:
        for row, ids in enumerate(labels):
            masks = ids.eq(tokenizer.vocab['<|mask:0|>']).nonzero(as_tuple=True)[0]
            ends = ids.eq(tokenizer.vocab['<|endofmask|>']).nonzero(as_tuple=True)[0]
            if len(masks) != 2 or len(ends) != 1 or masks[1] + 1 >= ends[0]:
                raise ValueError('蛋白生成提示必须包含两个 mask 和完整的目标分子/属性/结束标记')

            main_mask[row, int(masks[1]):int(ends[0])] = True
    extra_mask = (shift_labels >= 750) & (shift_labels <= 756)
    valid = shift_labels.ne(tokenizer.pad_token_id)
    return main_mask & valid, extra_mask & valid


def _calculate_loss(outputs, labels, pdf_array, generate_molecule):
    shift_logits = outputs.logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    main_mask, extra_mask = _loss_masks(labels, generate_molecule)
    active = main_mask | (extra_mask & (args.threshold > 0))
    count = active.sum()
    if count.item() == 0:
        raise ValueError('当前 batch 没有可监督的目标 token')
    log_probs = F.log_softmax(shift_logits, dim=-1)
    main_loss = -(pdf_array[shift_labels[main_mask]] * log_probs[main_mask]).sum()
    extra_loss = shift_logits.new_zeros(())
    if args.threshold > 0:
        extra_loss = -args.threshold * log_probs[extra_mask].gather(1, shift_labels[extra_mask].unsqueeze(-1)).sum()
    preds = shift_logits.argmax(dim=-1)
    accuracy = ((preds == shift_labels) & active).sum().float() / count
    return (main_loss + extra_loss) / count, accuracy, preds


def calculate_loss_and_accuracy_label(outputs, labels, pdf_array):
    return _calculate_loss(outputs, labels, pdf_array, generate_molecule=False)


def calculate_loss_and_accuracy_all(outputs, labels, pdf_array):
    return _calculate_loss(outputs, labels, pdf_array, generate_molecule=True)


def _property_positions(input_ids):
    sep = input_ids.eq(tokenizer.vocab['[SEP]'])
    end = input_ids.eq(tokenizer.vocab['<|endofmask|>'])
    if not (sep.sum(dim=1).eq(1).all() and end.sum(dim=1).eq(1).all()):
        raise ValueError('属性提示缺少唯一的 SEP 或结束标记')
    return sep.long().argmax(dim=1) - 1, end.long().argmax(dim=1) - 1


def compute_attention_supervision_loss(attentions, input_ids, tokenizer,
                                       pp_token='[SEP]', mask_token='<|endofmask|>'):

    support_pos, query_pos = _property_positions(input_ids)
    rows = torch.arange(input_ids.size(0), device=input_ids.device)
    scores = attentions[-1].mean(dim=1)[rows, query_pos - 1, :]
    positions = torch.arange(input_ids.size(1), device=input_ids.device)[None, :]
    valid = (positions <= (query_pos - 1)[:, None]) & input_ids.ne(tokenizer.pad_token_id)
    targets = torch.exp(-0.5 * ((positions - support_pos[:, None]).float() / 0.3) ** 2) * valid
    targets = targets / targets.sum(dim=1, keepdim=True)
    scores = scores * valid
    scores = scores / scores.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return F.kl_div(scores.clamp_min(1e-8).log(), targets, reduction='batchmean')


def prediction_distribution_alignment_loss(outputs, embed_batch, input_ids, T=1.0, mode='soft_weight'):
    support_pos, query_pos = _property_positions(input_ids)
    rows = torch.arange(input_ids.size(0), device=input_ids.device)
    logits_s = outputs.logits[rows, support_pos - 1]
    logits_q = outputs.logits[rows, query_pos - 1]
    kl = F.kl_div(F.log_softmax(logits_q / T, dim=-1), F.softmax(logits_s / T, dim=-1),
                  reduction='none').sum(dim=-1)  # 实际方向为 KL(s || q)，与原实现一致。
    if embed_batch.size(1) != 1 or embed_batch.size(-1) % 2:
        raise ValueError('属性对齐需要 query/support 两个等维描述符拼成一个额外 token')
    emb_q, emb_s = embed_batch[:, 0].chunk(2, dim=-1)
    sim_weight = (F.cosine_similarity(emb_q, emb_s, dim=-1) + 1) / 2  # [B]，不跨 batch 广播
    if mode == 'inverse_weight':
        sim_weight = 1 - sim_weight
    elif mode == 'hard_margin':
        sim_weight = sim_weight.gt(0.5).float()
    elif mode != 'soft_weight':
        raise ValueError('Unsupported alignment mode')
    return (sim_weight * kl).mean()


class EarlyStopping:
    def __init__(self, patience=30, verbose=False):
        self.patience, self.verbose = patience, verbose
        self.best_score, self.counter, self.early_stop = float('inf'), 0, False

    def __call__(self, val_loss, model, model_name):
        if not math.isfinite(val_loss):
            raise FloatingPointError('验证损失不是有限值，不保存该模型')
        if val_loss < self.best_score:
            self.best_score, self.counter = val_loss, 0
            torch.save(model.state_dict(), str(Path(model_name) / 'pytorch_model.pt'))
            print(f'保存最佳模型，val_loss={val_loss:.6f}')
        else:
            self.counter += 1
            self.early_stop = self.counter >= self.patience
            print(f'EarlyStopping counter: {self.counter}/{self.patience}')


def _check_model_runtime():
    from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
    if not all(hasattr(GPT2Attention, key) for key in ('_split_heads', '_merge_heads', '_attn')):
        raise RuntimeError('GPT_meta 使用旧 GPT2Attention 接口；请在兼容环境运行（原配置 transformers==4.24.0）')


def train(train_dataset, eval_dataset, args):
    _check_model_runtime()
    Ada_config = GPT2Config(architectures=['GPT2LMHeadModel'], model_type='GPT2LMHeadModel',
                           vocab_size=len(tokenizer), n_positions=400, n_ctx=400, n_embd=768,
                           n_layer=12, n_head=8,
                           task_specific_params={'text-generation': {'do_sample': True, 'max_length': 400}})
    model = PromptGPT(pretrain_path=args.model_path, config=Ada_config, emb_dim=args.extra_emb_dim)

    model.to(device)
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                 pin_memory=device.type == 'cuda', collate_fn=collate_fn)
    evalDataLoader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False,
                                                collate_fn=collate_fn)
    num_training_steps = args.epochs * len(trainDataLoader)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_scheduler('linear', optimizer=optimizer, num_warmup_steps=args.warmup_steps,
                                 num_training_steps=num_training_steps)
    early_stopping = EarlyStopping(patience=args.early_patience)
    pdf_array = get_all_normal_dis_pdf().to(device)
    loss_fn = calculate_loss_and_accuracy_label if args.task == 'property' else calculate_loss_and_accuracy_all
    os.makedirs(args.save_model_path, exist_ok=True)
    with open(Path(args.save_model_path) / 'training_args.json', 'w', encoding='utf-8') as handle:
        json.dump(vars(args), handle, ensure_ascii=False, indent=2)
    batch_steps = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch, extra_batch, extra_lengths in trainDataLoader:
            batch, extra_batch = batch.to(device), extra_batch.to(device)
            optimizer.zero_grad()
            outputs = forward_batch(model, batch, extra_batch, extra_lengths)
            loss, acc, _ = loss_fn(outputs, batch, pdf_array)
            if args.task == 'property':
                if args.lambda_attn > 0:
                    loss = loss + args.lambda_attn * compute_attention_supervision_loss(outputs.attentions, batch, tokenizer)
                if args.lambda_alig > 0:
                    loss = loss + args.lambda_alig * prediction_distribution_alignment_loss(outputs, extra_batch, batch)
            if not torch.isfinite(loss).item():
                raise FloatingPointError('训练损失出现 NaN/Inf，请检查输入特征及学习率')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            batch_steps += 1
            total_loss += loss.item()
            if batch_steps % args.log_step == 0:
                print(f'train epoch {epoch + 1}/{args.epochs}, batch {batch_steps}/{num_training_steps}, '
                      f'loss {loss.item():.6f}, accuracy {acc.item():.6f}')
        print('average epoch loss:', total_loss / len(trainDataLoader))
        eval_loss, eval_acc, _ = evaluate(model, evalDataLoader)
        early_stopping(eval_loss, model, args.save_model_path)
        if early_stopping.early_stop:
            print('Early stopping')
            break
    print(f'最佳验证损失 {early_stopping.best_score:.6f}；模型：{args.save_model_path}/pytorch_model.pt')


def evaluate(model, dataloader):
    was_training = model.training
    model.eval()
    loss_sum, correct_sum, target_count = 0.0, 0.0, 0
    pdf_array = get_all_normal_dis_pdf().to(device)
    generate = args.task == 'protein_generation'
    loss_fn = calculate_loss_and_accuracy_all if generate else calculate_loss_and_accuracy_label
    outputs = None
    with torch.no_grad():
        for batch, extra_batch, extra_lengths in tqdm(dataloader):
            batch, extra_batch = batch.to(device), extra_batch.to(device)
            outputs = forward_batch(model, batch, extra_batch, extra_lengths)
            loss, acc, _ = loss_fn(outputs, batch, pdf_array)
            main_mask, extra_mask = _loss_masks(batch, generate)
            count = int((main_mask | (extra_mask & (args.threshold > 0))).sum().item())
            loss_sum += loss.item() * count
            correct_sum += acc.item() * count
            target_count += count
    model.train(was_training)
    if target_count == 0:
        raise ValueError('验证集没有可监督目标')
    epoch_loss, epoch_accuracy = loss_sum / target_count, correct_sum / target_count
    print(f'val_loss: {epoch_loss:.6f}, val_accuracy: {epoch_accuracy:.6f}')
    return epoch_loss, epoch_accuracy, outputs


def read_data(path):
    data = []
    with open(path, 'rb') as handle:
        while True:
            try:
                chunk = pickle.load(handle)
            except EOFError:
                break
            if not isinstance(chunk, (list, tuple, np.ndarray)):
                raise ValueError(f'{path} 必须存储蛋白矩阵列表，不能是字典或其他对象')
            if isinstance(chunk, np.ndarray) and chunk.ndim == 2:
                raise ValueError(f'{path} 中单个矩阵应包装成列表 [matrix] 后保存')
            data.extend(chunk)
    return data


def _read_csv(path):
    data = pd.read_csv(path, usecols=[args.column])[args.column].tolist()
    if not data:
        raise ValueError(f'数据文件为空：{path}')
    return data


def main(argv=None):
    global args, tokenizer, property_tokenizer, device
    args = setup_args(argv)
    load_dependencies(args.code_dir)
    if not args.prepare_only:
        _check_model_runtime()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    requested = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else args.device
    device = torch.device('cpu' if requested == 'auto' else requested)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('指定了 cuda，但当前环境没有可用 CUDA 设备')
    tokenizer = ExpressionBertTokenizer.from_pretrained(args.vocab_path)
    property_tokenizer = PropertyTokenizer()

    train_data = organize_mixed_dataset(_read_csv(args.train_raw_path))
    valid_data = organize_mixed_dataset(_read_csv(args.valid_raw_path))
    if args.task == 'property':

        train_tokens, train_texts = data_to_list(train_data)
        valid_tokens, valid_texts = data_to_list(valid_data)
        train_list = mix_extra_embedding(train_texts, train_tokens)
        eval_list = mix_extra_embedding(valid_texts, valid_tokens)
    else:
        extra_smiles, extra_labels = organize_mixed_dataset(_read_csv(args.extra_raw_path))
        protein_matrix = read_data(args.train_protein_path)
        train_list = data_to_list(*train_data, extra_smiles, extra_labels, protein_matrix)
        del protein_matrix
        eval_protein = read_data(args.valid_protein_path)
        eval_list = data_to_list(*valid_data, extra_smiles, extra_labels, eval_protein)
        del eval_protein


    train_dataset, eval_dataset = MyDataset(train_list), MyDataset(eval_list)
    print(f'task={args.task}, train={len(train_dataset)}, valid={len(eval_dataset)}, '
          f'extra_emb_dim={args.extra_emb_dim}, device={device}')
    if args.prepare_only:
        print('数据准备和维度校验完成；未启动训练、未写模型文件。')
        return train_dataset, eval_dataset
    train(train_dataset, eval_dataset, args)


if __name__ == '__main__':
    main()
