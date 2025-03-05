import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel


# 用预训练模型获取初始特征

# # PubChem10M_SMILES_BPE_450k
# # ChemBERTa_zinc250k_v2_40k
# model_name = 'ChemBERTa_zinc250k_v2_40k'
# tokenizer_ = AutoTokenizer.from_pretrained("seyonec/{}".format(model_name))
# model_ = AutoModel.from_pretrained("seyonec/{}".format(model_name))
#
# drug = pd.read_csv("database/NP_Herb.csv")['SMILES'].values.tolist()
# with open("database/NP_Feature_.txt", 'w') as file:
#     for smiles in tqdm.tqdm(drug):
#         outputs = model_(**tokenizer_(smiles, return_tensors='pt'))
#         seq_feature = outputs.pooler_output[0].detach().numpy().tolist()
#         seq_feature = [str(f) for f in seq_feature]
#         for l in seq_feature:
#             line = ','.join(seq_feature)
#         file.write(line + '\n')

# # 计算分子指纹
# def compute_fingerprint(smiles):
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is not None:
#         # 计算 Morgan 指纹（半径为2，2048位）
#         fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
#         return np.array(fingerprint)
#     else:
#         return np.zeros(2048)
#
#
# smiles_list = pd.read_csv("database/NP_Herb.csv")['SMILES'].values.tolist()
#
# # 获取所有分子的指纹
# fingerprints = [compute_fingerprint(smiles) for smiles in smiles_list]
# fingerprints = np.array([fp for fp in fingerprints if fp is not None])
#
# index = pd.read_csv("database/NP_Herb.csv")['np_id'].values.tolist()
# data = pd.DataFrame(fingerprints)     # 1424x768
# data.index = index
# data.to_csv("database/NP_Feature_Finger.csv")
# print(data)

# index = pd.read_csv("database/NP_Herb.csv")['np_id'].values.tolist()
# data = pd.read_csv("database/NP_Feature_.txt", header=None)     # 1424x768
# data.index = index
# data.to_csv("dataset/NP_Feature_.csv")
# print(data)

data = pd.read_csv("database/NP_Herb.csv")
Herb_feature = pd.read_csv("dataset/NP_Feature_.csv", index_col=0)

# 标准化 herb_feature 特征矩阵
scaler = StandardScaler()
herb_feature_matrix = scaler.fit_transform(Herb_feature.values)

# 使用KMeans进行聚类
kmeans = KMeans(n_clusters=768, random_state=42)
data['Cluster'] = kmeans.fit_predict(herb_feature_matrix)
# data[['np_id', 'FHDI_Herb_ID', 'SMILES', 'Cluster']].to_csv("dataset/NP_Herb_Finger_Clustered.csv", index=False)
data[['np_id', 'FHDI_Herb_ID', 'SMILES', 'Cluster']].to_csv("dataset/NP_Herb_.csv", index=False)

# data = pd.read_csv("dataset/NP_Herb_Finger_Clustered.csv")
data = pd.read_csv("dataset/NP_Herb_.csv")
herb_feature = {}
one_hot = []
for np_id, FHDI_Herb_ID, SMILES, Cluster in data.values:
    one_hot = [0] * 768
    one_hot[int(Cluster)] = 1
    herb_feature[str(FHDI_Herb_ID)] = one_hot
herb_feature = pd.DataFrame(herb_feature).T
# herb_feature.to_csv("dataset/Herb_Finger_feature.csv")
herb_feature.to_csv("dataset/Herb_feature_.csv")
