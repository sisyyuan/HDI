import torch
import random
import argparse
import numpy as np
import pandas as pd
import torch_geometric.transforms as T
from model import DVGAE
from torch_geometric.data import Data

seed = 2024
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='DVGAE')
parser.add_argument('--channels', type=int, default=256)
parser.add_argument('--scaling_factor', type=float, default=1.8)
parser.add_argument('--epochs', type=int, default=1000)
args = parser.parse_args()

drug_feature = pd.read_csv("database/Drug_feature.csv", index_col=0)  # [1424 rows x 768 columns]
herb_feature = pd.read_csv("database/Herb_feature.csv", index_col=0)  # [681 rows x 768 columns]
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


def split_data(node_feature, edges, A_node_num, B_node_num, ratio=1, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
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


train_data, val_data, test_data = split_data(features, edges, len(drug_feature), len(herb_feature))
train_data, val_data, test_data = train_data.cuda(), val_data.cuda(), test_data.cuda()

network_input = torch.eye(2105).cuda()
model = DVGAE(args, 768, 2105).cuda()
model.load_state_dict(torch.load("checkpoint.pt"))
model.eval()
print(model)

test_pos, test_neg = test_data.pos_edge_label_index, test_data.neg_edge_label_index
z1 = model.encode1(test_data.x, test_data.edge_index)
z2 = model.encode2(network_input, test_data.edge_index)
output = model.test(z1, z2, 1.0, test_pos, test_neg)
print(output)

