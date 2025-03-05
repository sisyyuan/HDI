import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import APPNP, GCNConv
from torch_geometric.nn.inits import reset
from torch_geometric.utils import (add_self_loops, negative_sampling, remove_self_loops, dense_to_sparse)
from sklearn.metrics import average_precision_score, roc_auc_score

EPS = 1e-15
MAX_LOGSTD = 10
sc = 0.8
device = torch.device('cuda')


def calculate_metrics(y_true, y_pred):
    TP = sum((y_true[i] == 1 and y_pred[i] == 1) for i in range(len(y_true)))
    TN = sum((y_true[i] == 0 and y_pred[i] == 0) for i in range(len(y_true)))
    FP = sum((y_true[i] == 0 and y_pred[i] == 1) for i in range(len(y_true)))
    FN = sum((y_true[i] == 1 and y_pred[i] == 0) for i in range(len(y_true)))

    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-10)
    sensitivity = TP / (TP + FN + 1e-10)
    precision = TP / (TP + FP + 1e-10)
    specificity = TN / (TN + FP + 1e-10)
    mcc = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    F1_score = 2 * (precision * sensitivity) / (precision + sensitivity + 1e-10)
    return accuracy, sensitivity, precision, specificity, F1_score, mcc


class Encoder(torch.nn.Module):
    def __init__(self, args, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(in_channels, out_channels)
        self.linear2 = nn.Linear(in_channels, out_channels)
        # self.propagate = APPNP(K=1, alpha=0)
        self.propagate = GCNConv(out_channels, out_channels)
        self.scaling_factor = args.scaling_factor

    def forward(self, x, edge_index):
        x_ = self.linear1(x)
        x_ = self.propagate(x_, edge_index)

        x = self.linear2(x)
        x = F.normalize(x, p=2, dim=1) * self.scaling_factor

        x = self.propagate(x, edge_index)
        return x, x_


class Encoder2(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder2, self).__init__()
        self.linear1 = nn.Linear(in_channels, out_channels, bias=False)
        self.linear2 = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x, edge_index):
        x_ = self.linear1(x)

        x = self.linear2(x)
        x = F.normalize(x, p=2, dim=1) * sc
        return x, x_


class DVGAE(torch.nn.Module):
    def __init__(self, args, input_dim, N):
        super(DVGAE, self).__init__()
        self.encoder1 = Encoder(args, input_dim, args.channels)
        self.encoder2 = Encoder2(N, 2)
        self.decoder = InnerProductDecoder2()
        DVGAE.reset_parameters(self)

    def reset_parameters(self):
        reset(self.encoder1)
        reset(self.encoder2)
        reset(self.decoder)

    def encode1(self, *args, **kwargs):
        self.__mu1__, self.__logstd1__ = self.encoder1(*args, **kwargs)
        self.__logstd1__ = self.__logstd1__.clamp(max=MAX_LOGSTD)
        z = self.reparametrize(self.__mu1__, self.__logstd1__)
        return z

    def encode2(self, *args, **kwargs):
        self.__mu2__, self.__logstd2__ = self.encoder2(*args, **kwargs)
        self.__logstd2__ = self.__logstd2__.clamp(max=MAX_LOGSTD)
        z = self.reparametrize(self.__mu2__, self.__logstd2__)
        return z

    def reparametrize(self, mu, logstd):
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def test(self, z1, z2, temp, pos_edge_index, neg_edge_index):

        pos_y = z1.new_ones(pos_edge_index.size(1))
        neg_y = z1.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(z1, z2, temp, pos_edge_index, training=False)
        neg_pred = self.decoder(z1, z2, temp, neg_edge_index, training=False)

        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        AUC = roc_auc_score(y, pred)
        AP = average_precision_score(y, pred)

        temp = torch.tensor(pred)
        temp[temp >= 0.5] = 1
        temp[temp < 0.5] = 0
        accuracy, sensitivity, precision, specificity, F1_score, mcc = calculate_metrics(y, temp.cpu())

        return ['AUC:{:.4f}'.format(AUC), 'AP:{:.4f}'.format(AP),
                'acc:{:.4f}'.format(accuracy.item()), 'sen:{:.4f}'.format(sensitivity.item()),
                'pre:{:.4f}'.format(precision.item()), 'spe:{:.4f}'.format(specificity.item()),
                'f1:{:.4f}'.format(F1_score.item()), 'mcc:{:.4f}'.format(mcc.item())]

    def recon_loss(self, z1, z2, temp, pos_edge_index, neg_edge_index=None):

        decode_p = self.decoder(z1, z2, temp, pos_edge_index, training=True)
        pos_loss = -torch.log(decode_p + EPS).sum()

        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z1.size(0))

        decode_n1 = self.decoder(z1, z2, temp, neg_edge_index, training=True)
        neg_loss = -torch.log(1 - decode_n1 + EPS).sum()

        return pos_loss + neg_loss

    def kl_loss1(self, mu=None, logstd=None):
        mu = self.__mu1__ if mu is None else mu
        logstd = self.__logstd1__ if logstd is None else logstd.clamp(max=MAX_LOGSTD)
        return -0.5 * torch.mean(torch.sum(1 + 2 * logstd - mu ** 2 - logstd.exp() ** 2, dim=1))

    def kl_loss2(self, mu=None, logstd=None):
        mu = self.__mu2__ if mu is None else mu
        logstd = self.__logstd2__ if logstd is None else logstd.clamp(max=MAX_LOGSTD)
        return -0.5 * torch.mean(torch.sum(1 + 2 * logstd - mu ** 2 - logstd.exp() ** 2, dim=1))


class InnerProductDecoder2(torch.nn.Module):
    def __init__(self):
        super(InnerProductDecoder2, self).__init__()

    def forward(self, z1, z2, temp, edge_index, training=True):

        if training:
            z11 = z1.detach().clone()
            vf = (z11[edge_index[0]] * z11[edge_index[1]]).sum(dim=1)
            la = torch.cat((torch.unsqueeze(vf, 1), torch.zeros(torch.unsqueeze(vf, 1).shape).to(device)), 1)
            la_ra = la
            a = F.gumbel_softmax(la_ra, tau=temp, hard=True)[:, :1]
            value_feature = (z1[edge_index[0]] * z1[edge_index[1]]).sum(dim=1)
            value_network = z2[edge_index[0], [0]] + z2[edge_index[1], [0]]
            feature_flag = torch.flatten(a)
            return feature_flag * torch.sigmoid(value_feature) +\
                   (1 - feature_flag) * torch.sigmoid(value_network)

        else:
            value_feature = (z1[edge_index[0]] * z1[edge_index[1]]).sum(dim=1)
            value_network = z2[edge_index[0], [0]] + z2[edge_index[1], [0]]
            return torch.sigmoid(value_feature) * torch.sigmoid(value_feature) +\
                   (1 - torch.sigmoid(value_feature)) * torch.sigmoid(value_network)

    def forward_all(self, z):
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj)
