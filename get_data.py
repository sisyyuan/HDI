import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data
import torch_geometric.transforms as T

# 读取数据
drug_feature = pd.read_csv("dataset/Drug_feature.csv", index_col=0)  # [1424 rows x 768 columns]
herb_feature = pd.read_csv("dataset/Herb_feature.csv", index_col=0)  # [681 rows x 768 columns]
interaction = pd.read_csv("dataset/DHI.csv")  # [10033 rows x 4 columns]

drug_features_tensor = torch.tensor(drug_feature.values, dtype=torch.float)
herb_features_tensor = torch.tensor(herb_feature.values, dtype=torch.float)

drug_index_map = {drug_id: idx for idx, drug_id in enumerate(drug_feature.index)}
herb_index_map = {herb_id: idx for idx, herb_id in enumerate(herb_feature.index)}

edges = []
for _, row in interaction.iterrows():
    drug_id = row['Drug_ID']
    herb_id = row['Food_Herb_ID']

    if drug_id in drug_index_map and herb_id in herb_index_map:
        drug_idx = drug_index_map[drug_id]
        herb_idx = herb_index_map[herb_id] + len(drug_feature)
        edges.append([drug_idx, herb_idx])

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
features = torch.cat([drug_features_tensor, herb_features_tensor], dim=0)


def edge_to_tensor(edge_list, add_reverse=True):
    output = []
    for edge in edge_list:
        a, b = edge
        output.append([a, b])
        if add_reverse:
            output.append([b, a])
    return torch.LongTensor(output).T


def split_data(node_feature, A_node_num, B_node_num, ratio=1, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    all_possible_edges = set((A, B + A_node_num) for A in range(A_node_num) for B in range(B_node_num))
    positive_edges = set(map(tuple, edges))
    negative_edges = list(all_possible_edges - positive_edges)
    positive_edges = list(positive_edges)
    negative_edges = list(negative_edges)
    np.random.shuffle(positive_edges)
    np.random.shuffle(negative_edges)

    num_edge = len(positive_edges)
    train_num = int(num_edge*train_ratio)
    val_num = int(num_edge*val_ratio)
    test_num = int(num_edge*test_ratio)

    train_pos_edge = positive_edges[:train_num]
    train_neg_edge = negative_edges[:train_num*ratio]

    val_pos_edge = positive_edges[train_num:train_num+val_num]
    val_neg_edge = negative_edges[train_num*ratio:(train_num + val_num)*ratio]

    test_pos_edge = positive_edges[train_num+val_num:train_num+val_num+test_num]
    test_neg_edge = negative_edges[(train_num + val_num)*ratio:(train_num + val_num + test_num)*ratio]

    train_data = Data(x=node_feature, edge_index=edge_to_tensor(train_pos_edge, add_reverse=True),
                      pos_edge_label_index=edge_to_tensor(train_pos_edge, add_reverse=False),
                      neg_edge_label_index=edge_to_tensor(train_neg_edge, add_reverse=False))
    val_data = Data(x=node_feature, edge_index=edge_to_tensor(train_pos_edge, add_reverse=True),
                    pos_edge_label_index=edge_to_tensor(val_pos_edge, add_reverse=False),
                    neg_edge_label_index=edge_to_tensor(val_neg_edge, add_reverse=False))
    test_data = Data(x=node_feature, edge_index=torch.cat((edge_to_tensor(train_pos_edge, add_reverse=True),
                                                           edge_to_tensor(train_pos_edge, add_reverse=True)), dim=-1),
                     pos_edge_label_index=edge_to_tensor(test_pos_edge, add_reverse=False),
                     neg_edge_label_index=edge_to_tensor(test_neg_edge, add_reverse=False))
    return train_data, val_data, test_data


train_data, val_data, test_data = split_data(features, len(drug_feature), len(herb_feature))
print(train_data)
print(val_data)
print(test_data)

# Data(x=[2105, 768], edge_index=[2, 12146], pos_edge_label=[6073], pos_edge_label_index=[2, 6073])
# Data(x=[2105, 768], edge_index=[2, 12146], pos_edge_label=[759], pos_edge_label_index=[2, 759], neg_edge_label=[759], neg_edge_label_index=[2, 759])
# Data(x=[2105, 768], edge_index=[2, 13664], pos_edge_label=[759], pos_edge_label_index=[2, 759], neg_edge_label=[759], neg_edge_label_index=[2, 759])

