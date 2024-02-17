import numpy as np
import torch
import math
from collections import Counter
from itertools import combinations
import itertools


def pool_mutual(output, pos_sample, neg_sample, edge_number):
    NMI_neg = []
    NMI_pos = []
    for i in range(edge_number):
        high_size = len(pos_sample[i])
        pos_vector_list = torch.round(10 * (output[pos_sample[i], :])) / 10
        neg_vector_list = torch.round(10 * (output[neg_sample[i], :])) / 10
        train_H = []
        neg_H = []
        list2 = list(range(0, high_size))
        for j in range(1, high_size+1):
            entro_2list = list(combinations(list2, j))
            for l in range(len(entro_2list)):
                entro_list = torch.LongTensor(entro_2list[l])
                t_X = torch.index_select(pos_vector_list, 0, entro_list)
                t_X = t_X.detach().numpy().tolist()
                t_X = list(zip(*itertools.chain(t_X)))
                n_X = torch.index_select(neg_vector_list, 0, entro_list)
                n_X = n_X.detach().numpy().tolist()
                n_X = list(zip(*itertools.chain(n_X)))
                train_H.append(((-1)**(j-1))*Entropy(t_X))
                neg_H.append(((-1)**(j-1))*Entropy(n_X))
        p_mi = np.sum(train_H)
        n_mi = np.sum(neg_H)
        train_H.append(high_size*n_mi/np.sum(train_H[0:high_size]))
        neg_H.append(high_size*p_mi/np.sum(neg_H[0:high_size]))
        NMI_neg.append(train_H)
        NMI_pos.append(neg_H)
    return (torch.tensor(NMI_pos + NMI_neg))


def Entropy(DataList):
    counts = len(DataList)
    counter = Counter(DataList)
    prob = {i[0]: i[1] / counts for i in counter.items()}
    H = - sum([i[1] * math.log2(i[1]) for i in prob.items()])
    return H

