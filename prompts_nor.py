import random
import os
import pickle
import gc
from rdkit import Chem, DataStructs
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from rdkit.Chem import AllChem
from typing import List, Tuple, Dict

from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm.auto import tqdm
import psutil
from joblib import Parallel, delayed


pp_des = {'HallKierAlpha':[-18, 8], 'MolWt':[16, 2600], 'NumAromaticRings':[0, 20], 'NumHAcceptors':[0, 50],
        'NumHDonors':[0, 50], 'NumRotatableBonds':[0, 150], 'TPSA':[0,966], 'MolLogP':[-20, 70], 'qed':[0, 1]}

def print_memory_usage(tag=""):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024  # MB
    print(f"[{tag}] 当前内存: {mem:.2f} MB")

def _generate_fp_from_smiles(smiles):
    try:
        # 在子进程中将 smiles 转为 mol，避免跨进程传递 mol 对象
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return AllChem.GetMorganFingerprintAsBitVect(mol, 2)
    except Exception:
        pass
    return None

class PropertyTokenizer:
    """Run a property tokenization."""
    def __init__(self) -> None:
        """Constructs a PropertyTokenizer."""

    def normalize_to_real(self, norm_label, min, max):
        try:
            # print("generate label number",len(norm_label))
            norm_label = norm_label[0]
            norm_label = float(norm_label.split('p_')[1])
            real_label = (norm_label) * (max - min) + min
            # eal_label=round(real_label,2)
        except ValueError:
            real_label = -1
        except IndexError:  # FANG
            real_label = -1
        return real_label

    def normalize_to_input(self, values, current_min, current_max, new_min, new_max):
        # 线性映射到指定范围
        normalized_values = (values - current_min) / (current_max - current_min) * (new_max - new_min) + new_min
        # 保留两位小数
        normalized_values = str("{:.2f}".format(normalized_values))
        normalized_values = "p_" + normalized_values
        return normalized_values


class MoleculeDescriptorCalculator:
    def __init__(self):
        self.chosen_descriptors = ['HallKierAlpha', 'MolWt', 'NumAromaticRings', 'NumHAcceptors',
                                   'NumHDonors', 'NumRotatableBonds', 'TPSA', 'MolLogP', 'qed']
        self.chosen_des_voc = ['<HallKierAlpha>', '<molwt>', '<NumAromaticRings>','<NumHAcceptors>',
                               '<NumHDonors>', '<NumRotatableBonds>', '<tpsa>', '<logp>', '<qed>']
        self.calculator = MolecularDescriptorCalculator(self.chosen_descriptors)

    def compute_descriptors(self, smiles):
        # 移除空格还原为标准的SMILES字符串
        property_tokenizer = PropertyTokenizer()
        formatted_smiles = ''.join(smiles.split())
        molecule = Chem.MolFromSmiles(formatted_smiles)
        if molecule is None:
            return f"Invalid SMILES: {smiles}"
        descriptors = self.calculator.CalcDescriptors(molecule)
        descriptors_tokens = [property_tokenizer.normalize_to_input(value, pp_des[name][0], pp_des[name][1], 0, 1) for name, value in zip(self.chosen_descriptors, descriptors)]
        # 构建输出字符串
        descriptor_str = ' '.join(f"{label} {value}" for label, value in zip(self.chosen_des_voc, descriptors_tokens))
        # 返回格式化后的结果
        return f"{smiles} {descriptor_str}"

class SimilarityFinder:
    def __init__(self, smiles_list: List[str], labels: List[float], top_k: int = 5,
                 label_diff_threshold: float = 0.1, similarity_threshold: float = 0.3,
                 mode: str = 'similar_property'):
        """
        参数:
            smiles_list (List[str]): 以空格分隔的SMILES字符串列表
            labels (List[float]): 每个分子的属性值列表
            top_k (int): 返回的分子个数
            label_diff_threshold (float): 属性值差异的阈值
            similarity_threshold (float): 相似度阈值
            mode (str): 'similar_property'寻找属性值接近的分子，'dissimilar_property'寻找属性值差异较大的分子
        """
        self.smiles_list = smiles_list
        self.labels = labels
        self.top_k = top_k
        self.label_diff_threshold = label_diff_threshold
        self.similarity_threshold = similarity_threshold
        self.mode = mode
        self.molecules = [Chem.MolFromSmiles(''.join(smiles.split())) for smiles in smiles_list]
        # self.fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, 2) for mol in self.molecules]
        self.n_jobs = max(1, 18)
        self.fingerprints = Parallel(n_jobs=self.n_jobs, batch_size=16)(
            delayed(_generate_fp_from_smiles)(''.join(smi.split())) for smi in smiles_list
        )
        
    def calculate_similarity(self, fp1, fp2):
        """计算两个分子指纹之间的Tanimoto相似性。"""
        return DataStructs.TanimotoSimilarity(fp1, fp2)

    def find_molecules(self) -> Dict[Tuple[str, float], List[Tuple[str, float]]]:
        """
        对每个分子根据模式找到满足属性和结构相似性的分子。

        返回:
            dict: 每个分子的(原始SMILES, 属性值)元组为键，值是包含满足条件的top-k分子列表。
        """
        result = {}

        for i, (smiles, label) in enumerate(zip(self.smiles_list, self.labels)):
            eligible_molecules = []
            for j, (other_smiles, other_label) in enumerate(zip(self.smiles_list, self.labels)):
                if i != j:
                    label_diff = abs(label - other_label)
                    sim = self.calculate_similarity(self.fingerprints[i], self.fingerprints[j])

                    # 判断当前分子是否符合筛选条件
                    if self.mode == 'similar_property':
                        if label_diff <= self.label_diff_threshold and sim >= self.similarity_threshold:
                            eligible_molecules.append((other_smiles, other_label, sim))
                    elif self.mode == 'dissimilar_property':
                        # if label_diff >= self.label_diff_threshold and sim >= self.similarity_threshold:
                        if label_diff >= self.label_diff_threshold:
                            eligible_molecules.append((other_smiles, other_label, sim))

            # 按相似度排序并选择top-k
            if self.mode == 'similar_property':
                top_k_similars = sorted(eligible_molecules, key=lambda x: -x[2])[:self.top_k]
            elif self.mode == 'dissimilar_property':
                top_k_similars = sorted(eligible_molecules, key=lambda x: x[2])[:self.top_k]

            result[(smiles, label)] = [(sm, lb) for sm, lb, _ in top_k_similars]

        return result


class ExtendedSimilarityFinder(SimilarityFinder):
    def __init__(self, smiles_list: List[str], labels: List[float], extra_smiles: List[str],
                 extra_labels: List[Tuple[str, float]], top_k: int = 5, descriptors_threshold: float = 0.0,
                 similarity_method: str = 'tanimoto', output_file: str = "extra_mol.pkl"):
        """
        参数:
            smiles_list (List[str]): 主分子的SMILES字符串列表
            labels (List[float]): 主分子的属性值列表
            extra_smiles (List[str]): 额外分子的SMILES字符串列表
            extra_labels (List[Tuple[str, float]]): 每个额外分子的属性名和属性值列表
            top_k (int): 返回的相似分子个数
            similarity_method (str): 'tanimoto' 或 'descriptor'，指定相似性计算方法
        """
        super().__init__(smiles_list, labels, top_k=top_k)
        self.extra_smiles = extra_smiles
        self.extra_labels = extra_labels
        self.similarity_method = similarity_method
        self.extra_molecules = [Chem.MolFromSmiles(''.join(smiles.split())) for smiles in extra_smiles]
        # self.output_file = output_file
        # if os.path.exists(self.output_file):
        #     os.remove(self.output_file)
        # self.extra_info = self.load_fp_and_descriptors_list(self.output_file)
         

        # 使用 joblib 进行多核并行计算
        if self.similarity_method == 'tanimoto':
            # self.extra_fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, 2) for mol in self.extra_molecules]
            self.extra_fingerprints = Parallel(n_jobs=self.n_jobs, batch_size=16)(
            delayed(_generate_fp_from_smiles)(''.join(smi.split())) for smi in extra_smiles
        )
        else:
            self.descriptors = self.get_descriptors_for_all_molecules(self.molecules)
            self.descriptor_threshold = descriptors_threshold
            self.extra_descriptors = self.get_descriptors_for_all_molecules(self.extra_molecules)
            
    def process_and_save(self, smiles, batch_size=5000):
        buffer = []
        with open(self.output_file, 'ab') as f:
            for smi in tqdm(smiles):
                mol = Chem.MolFromSmiles(''.join(smi.split()))
                if mol is None:
                    continue  # 跳过非法分子
                # 计算指纹
                if self.similarity_method=="tanimoto":
                    res = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
                    buffer.append(res)
                else:
                    buffer.append(mol)
                
                # 每个样本单独写入
                if len(buffer) >= batch_size:
                    if self.similarity_method!="tanimoto": 
                        buffer = self.get_descriptors_for_all_molecules(buffer)
                    pickle.dump(buffer, f)
                    buffer = []

            if buffer:  # flush last batch
                if self.similarity_method!="tanimoto": 
                    buffer = self.get_descriptors_for_all_molecules(buffer)
                pickle.dump(buffer, f)           

    def load_fp_and_descriptors_list(self, pkl_file):
        self.process_and_save(self.extra_smiles)
        data = []
        print_memory_usage("load begin.....")
        with open(pkl_file, 'rb') as f:
            while True:
                try:
                    item = pickle.load(f)
                    data.extend(item)
                except EOFError:
                    break
        return data  # 返回 list[fingerprint_arr] 

    def get_chosen_descriptors(self):
        chosen_descriptors = ['BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1',
                              'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v',
                              'EState_VSA1', 'EState_VSA10', 'EState_VSA11', 'EState_VSA2', 'EState_VSA3',
                              'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8',
                              'EState_VSA9', 'ExactMolWt', 'FpDensityMorgan1', 'FpDensityMorgan2',
                              'FpDensityMorgan3', 'FractionCSP3', 'HallKierAlpha', 'HeavyAtomCount',
                              'HeavyAtomMolWt', 'Ipc', 'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA',
                              'MaxAbsEStateIndex', 'MaxAbsPartialCharge', 'MaxEStateIndex', 'MaxPartialCharge',
                              'MinAbsEStateIndex', 'MinAbsPartialCharge', 'MinEStateIndex', 'MinPartialCharge',
                              'MolLogP', 'MolMR', 'MolWt', 'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles',
                              'NumAliphaticHeterocycles', 'NumAliphaticRings', 'NumAromaticCarbocycles',
                              'NumAromaticHeterocycles', 'NumAromaticRings', 'NumHAcceptors', 'NumHDonors',
                              'NumHeteroatoms', 'NumRadicalElectrons', 'NumRotatableBonds',
                              'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 'NumSaturatedRings',
                              'NumValenceElectrons', 'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12',
                              'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5',
                              'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'RingCount', 'SMR_VSA1',
                              'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7',
                              'SMR_VSA8', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA11', 'SlogP_VSA12',
                              'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7',
                              'SlogP_VSA8', 'SlogP_VSA9', 'TPSA', 'VSA_EState1', 'VSA_EState10', 'VSA_EState2',
                              'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7',
                              'VSA_EState8', 'VSA_EState9', 'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN',
                              # 'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O',
                              # 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0', 'fr_NH1', 'fr_NH2',
                              'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_Nhpyrrole', 'fr_SH',
                              'fr_aldehyde', 'fr_alkyl_carbamate', 'fr_alkyl_halide', 'fr_allylic_oxid',
                              'fr_amide', 'fr_amidine', 'fr_aniline', 'fr_aryl_methyl', 'fr_azide', 'fr_azo',
                              'fr_barbitur', 'fr_benzene', 'fr_benzodiazepine', 'fr_bicyclic', 'fr_diazo',
                              'fr_dihydropyridine', 'fr_epoxide', 'fr_ester', 'fr_ether', 'fr_furan', 'fr_guanido',
                              'fr_halogen', 'fr_hdrzine', 'fr_hdrzone', 'fr_imidazole', 'fr_imide', 'fr_isocyan',
                              'fr_isothiocyan', 'fr_ketone', 'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone',
                              'fr_methoxy', 'fr_morpholine', 'fr_nitrile', 'fr_nitro', 'fr_nitro_arom',
                              'fr_nitro_arom_nonortho', 'fr_nitroso', 'fr_oxazole', 'fr_oxime',
                              'fr_para_hydroxylation', 'fr_phenol', 'fr_phenol_noOrthoHbond', 'fr_phos_acid',
                              'fr_phos_ester', 'fr_piperdine', 'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd',
                              'fr_pyridine', 'fr_quatN', 'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone',
                              'fr_term_acetylene', 'fr_tetrazole', 'fr_thiazole', 'fr_thiocyan', 'fr_thiophene',
                              'fr_unbrch_alkane', 'fr_urea', 'qed']
        return chosen_descriptors

    def compute_descriptors(self, molecule):
        """
        计算给定小分子的描述符特征。
        这个函数应该返回一个长度为1015的描述符列表。
        """
        calculator = MolecularDescriptorCalculator(self.get_chosen_descriptors())
        # print(len(get_chosen_descriptors()))
        # print(calculator.GetDescriptorSummaries())
        X_rdkit = calculator.CalcDescriptors(molecule)
        return X_rdkit

    # def get_descriptors_for_all_molecules(self, molecules):
    #     """
    #     计算所有小分子的描述符特征。
    #     """
    #     with ThreadPoolExecutor() as executor:
    #         descriptors = list(executor.map(self.compute_descriptors, molecules))
    #     return np.nan_to_num(np.array(descriptors),nan=0)
    
    def get_descriptors_for_all_molecules(self, molecules):
        """
        工业级优化版本（接口不变）
        支持输入 mol 或 smiles
        """

        # 🔹 统一转 SMILES（避免跨进程传 mol）
        smiles_list = []
        for m in molecules:
            if isinstance(m, str):
                smiles_list.append(''.join(m.split()))
            else:
                smiles_list.append(Chem.MolToSmiles(m) if m is not None else None)

        # 🔹 分块
        chunk_size = 256
        chunks = [
            smiles_list[i:i + chunk_size]
            for i in range(0, len(smiles_list), chunk_size)
        ]

        descriptor_names = self.get_chosen_descriptors()

        # 🔹 batch worker（每个进程只初始化一次 calculator）
        def _worker(chunk):
            calc = MolecularDescriptorCalculator(descriptor_names)
            out = []
            for smi in chunk:
                if smi is None:
                    out.append(None)
                    continue
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    out.append(None)
                else:
                    try:
                        out.append(calc.CalcDescriptors(mol))
                    except Exception:
                        out.append(None)
            return out

        # 🔹 并行（限制核数，防炸）
        results = Parallel(
            n_jobs=min(self.n_jobs, 4),
            batch_size=1,   # 已经手动分chunk
            backend="loky"
        )(
            delayed(_worker)(chunk) for chunk in chunks
        )

        # 🔹 展平
        descriptors = [d for batch in results for d in batch]

        # 🔹 填充
        dim = len(descriptor_names)
        descriptors = [
            d if d is not None else np.zeros(dim)
            for d in descriptors
        ]

        return np.nan_to_num(np.array(descriptors), nan=0)

    def find_top_k_from_extra(self) -> List[Tuple[str, float, List[Tuple[str, str, float]]]]:
        """
        在extra_smiles列表中找到与smiles_list中每个分子最相似的top-k分子。

        返回:
            List[Tuple[str, float, List[Tuple[str, str, float]]]]: 每条数据包含主分子信息及其top-k相似分子列表
        """
        result = []
        for i, (smiles, label) in tqdm(enumerate(zip(self.smiles_list, self.labels))):
            eligible_molecules = []
            for j, (extra_smiles, (attr_name, extra_label)) in enumerate(zip(self.extra_smiles, self.extra_labels)):
                if self.similarity_method == 'tanimoto':
                    target_fp = self.fingerprints[i]
                    similarity = self.calculate_similarity(target_fp, self.extra_fingerprints[j])
                    if similarity >= self.similarity_threshold:
                        eligible_molecules.append((extra_smiles, attr_name, extra_label, similarity))
                elif self.similarity_method == 'descriptor':
                    # target_desc = self.get_descriptors_for_all_molecules([self.molecules[i]])[0]
                    target_desc = self.descriptors[i]
                    similarity = cosine_similarity(
                        target_desc.reshape(1, -1),
                        self.extra_descriptors[j].reshape(1, -1)
                    )[0, 0]
                    if similarity >= self.descriptor_threshold:
                        eligible_molecules.append((extra_smiles, attr_name, extra_label, similarity))
            top_k_similars = sorted(eligible_molecules, key=lambda x: -x[3])[:self.top_k]
            top_k_cleaned = [(sm, attr, lb) for sm, attr, lb, _ in top_k_similars]
            result.append((smiles, label, top_k_cleaned))
        return result


        # def find_top_k_from_extra(self) -> Dict[Tuple[str, float], List[Tuple[str, str, float]]]:
        #     """
        #     在extra_smiles列表中找到与smiles_list中每个分子最相似的top-k分子。

        #     返回:
        #         dict: 键是主分子的(原始SMILES, 属性值)，值是包含top-k分子的(smiles, 属性名, 属性值)的列表
        #     """
        #     result = {}
        #     for i, (smiles, label) in tqdm(enumerate(zip(self.smiles_list, self.labels))):
        #         target_fp = AllChem.GetMorganFingerprintAsBitVect(self.molecules[i], 2)
        #         eligible_molecules = []
        #         for j, (extra_smiles, (attr_name, extra_label)) in enumerate(zip(self.extra_smiles, self.extra_labels)):
        #             if self.similarity_method == 'tanimoto':
        #                 # 使用Tanimoto相似度
        #                 similarity = self.calculate_similarity(target_fp, self.extra_fingerprints[j])
        #                 if similarity >= self.similarity_threshold:
        #                     eligible_molecules.append((extra_smiles, attr_name, extra_label, similarity))
        #             elif self.similarity_method == 'descriptor':
        #                 # 使用描述符相似度（余弦相似度）
        #                 target_desc = self.get_descriptors_for_all_molecules([self.molecules[i]])[0]
        #                 similarity = cosine_similarity(target_desc.reshape(1, -1), self.extra_descriptors[j].reshape(1, -1))[0, 0]
        #                 if similarity >= self.descriptor_threshold:
        #                     eligible_molecules.append((extra_smiles, attr_name, extra_label, similarity))
        #             gc.collect()
        #         # 按相似度排序并选择top-k
        #         top_k_similars = sorted(eligible_molecules, key=lambda x: -x[3])[:self.top_k]
        #         result[(smiles, label)] = [(sm, attr, lb) for sm, attr, lb, _ in top_k_similars]
        #     return result

class SimLabelFinder:
    def __init__(self, smiles_list: List[str], labels: List[float], top_k: int = 5,
                 label_diff_threshold: float = 0.1,
                 mode: str = 'sim_property'):
        """
        参数:
            smiles_list (List[str]): 以空格分隔的SMILES字符串列表
            labels (List[float]): 每个分子的属性值列表
            top_k (int): 返回的分子个数
            label_diff_threshold (float): 属性值差异的阈值
            mode (str): 'sim_property'寻找属性值接近的分子，'dissim_property'寻找属性值差异较大的分子
        """
        self.smiles_list = smiles_list
        self.labels = labels
        self.top_k = top_k
        self.label_diff_threshold = label_diff_threshold
        self.mode = mode

    def find_molecules(self) -> Dict[Tuple[str, float], List[Tuple[str, float]]]:
        """
        对每个分子根据模式找到满足属性接近或差距大的分子。
        返回:
            dict: 每个分子的(原始SMILES, 属性值)元组为键，值是包含满足条件的top-k分子列表。
        """
        result = {}
        for i, (smiles, label) in enumerate(zip(self.smiles_list, self.labels)):
            eligible_molecules = []
            for j, (other_smiles, other_label) in enumerate(zip(self.smiles_list, self.labels)):
                if i != j:
                    label_diff = abs(label - other_label)
                    # 判断当前分子是否符合筛选条件
                    if self.mode == 'sim_property':
                        if label_diff <= self.label_diff_threshold :
                            eligible_molecules.append((other_smiles, other_label, label_diff))
                    elif self.mode == 'dissim_property':
                        if label_diff >= self.label_diff_threshold:
                            eligible_molecules.append((other_smiles, other_label, label_diff))
            # 按属性差距排序并选择top-k
            if self.mode == 'sim_property':
                top_k_smiles = sorted(eligible_molecules, key=lambda x: -x[2], reverse = True)[:self.top_k]
            elif self.mode == 'dissim_property':
                random.shuffle(eligible_molecules)
                top_k_smiles = eligible_molecules[:self.top_k]

            result[(smiles, label)] = [(sm, lb) for sm, lb, _ in top_k_smiles]
        return result


class ExtendedSimLabelFinder(SimLabelFinder):
    def __init__(self, smiles_list: List[str], labels: List[float],
                 extra_smiles_list: List[str], extra_labels: List[Tuple[str, float]],
                 top_k: int = 5, label_diff_threshold: float = 0.1,
                 similarity_threshold: float = 0.3, mode: str = 'sim_property'):
        """
        参数:
            extra_smiles_list (List[str]): 额外的SMILES字符串列表
            extra_labels (List[Tuple[str, float]]): 每个额外分子的属性名和属性值列表
            similarity_threshold (float): 分子结构相似性阈值
            其他参数继承自SimLabelFinder
        """
        super().__init__(smiles_list, labels, top_k, label_diff_threshold, mode)
        self.extra_smiles_list = extra_smiles_list
        self.extra_labels = extra_labels
        self.similarity_threshold = similarity_threshold
        # self.property_name = extra_labels[0][0]

    @staticmethod
    def calculate_similarity(smiles1: str, smiles2: str) -> float:
        """
        计算两个分子的Tanimoto相似性
        """
        mol1 = Chem.MolFromSmiles(''.join(smiles1.split()))
        mol2 = Chem.MolFromSmiles(''.join(smiles2.split()))
        if mol1 and mol2:
            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=2048)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=2048)
            return DataStructs.TanimotoSimilarity(fp1, fp2)
        return 0.0

    def find_top_k_from_extra(self) -> Dict[Tuple[str, float], List[Tuple[str, float]]]:
        """
        对查询列表中的每个分子，在额外列表中找到与其结构相似性满足阈值，且属性值最接近的top-k分子。
        返回:
            dict: 查询分子的(原始SMILES, 属性名, 属性值)元组为键，值是包含满足条件的top-k分子列表。
        """
        result = {}
        for query_smiles, query_label in zip(self.smiles_list, self.labels):
            eligible_molecules = []
            nosim_list = []
            for extra_smiles, (attr_name, extra_label) in zip(self.extra_smiles_list, self.extra_labels):
                similarity = self.calculate_similarity(query_smiles, extra_smiles)
                label_diff = abs(query_label - extra_label)
                if similarity >= self.similarity_threshold:
                    if self.mode == 'sim_property' and label_diff <= self.label_diff_threshold:
                        eligible_molecules.append((extra_smiles, extra_label, similarity, label_diff, attr_name))
                    elif self.mode == 'dissim_property' and label_diff >= self.label_diff_threshold:
                        eligible_molecules.append((extra_smiles, extra_label, similarity, label_diff, attr_name))
                else:
                    if self.mode == 'sim_property' and label_diff <= self.label_diff_threshold:
                        nosim_list.append((extra_smiles, extra_label, similarity, label_diff, attr_name))
                    elif self.mode == 'dissim_property' and label_diff >= self.label_diff_threshold:
                        nosim_list.append((extra_smiles, extra_label, similarity, label_diff, attr_name))
            if eligible_molecules==[]:
                eligible_molecules = nosim_list
            # 按属性值差距排序并选择top-k
            top_k_smiles = sorted(eligible_molecules, key=lambda x: (x[3], -x[2]))[:self.top_k]
            result[(query_smiles, query_label)] = [(sm, att, lb) for sm, lb, _, _ , att in top_k_smiles]
        return result
