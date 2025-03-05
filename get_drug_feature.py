import tqdm
import pandas as pd
from transformers import AutoTokenizer, AutoModel
# 用预训练模型获取初始特征

# PubChem10M_SMILES_BPE_450k
model_name = 'ChemBERTa-zinc-base-v1'
tokenizer_ = AutoTokenizer.from_pretrained("seyonec/{}".format(model_name))
model_ = AutoModel.from_pretrained("seyonec/{}".format(model_name))

drug = pd.read_csv("database/Drug_SMILES.csv")['Canonical_SMILES'].values.tolist()
for smiles in tqdm.tqdm(drug):
    outputs = model_(**tokenizer_(smiles, return_tensors='pt'))
    seq_feature = outputs.pooler_output[0].detach().numpy().tolist()
    seq_feature = [str(f) for f in seq_feature]
    with open("database/Drug_feature.txt", 'a+') as file:
        for l in seq_feature:
            line = ','.join(seq_feature)
        file.write(line + '\n')

index = pd.read_csv("database/Drug_SMILES.csv")['FHDI_Drug_ID'].values.tolist()
data = pd.read_csv("database/Drug_feature.txt", header=None)     # 1424x768
data.index = index
data.to_csv("dataset/Drug_feature.csv")
print(data)
