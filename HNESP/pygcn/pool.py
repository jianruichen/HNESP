import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn


def pooling_method1(node_feature, pos, neg, simp_num):
    for i in range(simp_num):
        ppos = np.array(pos)
        nneg = np.array(neg)
        pos_vector = torch.cat(([node_feature[int(ppos[i, j])].reshape(1, node_feature.shape[1]) for j in range(len(pos[1]))]), axis=1)
        neg_vector = torch.cat(([node_feature[int(nneg[i, j])].reshape(1, node_feature.shape[1]) for j in range(len(neg[1]))]), axis=1)
        if i == 0:
            pos_feature = pos_vector
            neg_feature = neg_vector
        else:
            pos_feature = torch.cat((pos_feature, pos_vector), axis=0)
            neg_feature = torch.cat((neg_feature, neg_vector), axis=0)
    labels = torch.cat((torch.ones(len(pos)).reshape(1, len(pos)), torch.zeros(len(neg)).reshape(1, len(pos))), 1)
    return torch.cat((pos_feature, neg_feature), axis=0), labels.T


def perturb_feature(feature_matrix, labels):
    new_raw = np.random.permutation(feature_matrix.shape[0])
    final_features = feature_matrix[new_raw, :]
    final_labels = labels[new_raw, :]
    return final_features, final_labels