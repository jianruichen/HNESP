from collections import Counter
import numpy as np
import random
from scipy import sparse


def preprocess_time(data, high_data):  #data是三列信息，节点，节点，时间
    new_time_list = data[:, -1]
    high_time_list = high_data[:, -1]
    time_unique = np.unique(new_time_list)
    per_scale = np.round((len(time_unique)*7)/10)#训练集的时间大小
    up_raw = time_unique[int(per_scale)]
    big_num = np.where(new_time_list > up_raw)
    small_num = np.where(new_time_list <= up_raw)
    high_big_num = np.where(high_time_list > up_raw)
    high_small_num = np.where(high_time_list <= up_raw)
    second_train_list = data[small_num[0], 0:2]
    second_test_list = data[big_num[0], 0:2]
    high_train_list = high_data[high_small_num[0], 0:high_data.shape[1]]
    high_test_list = high_data[high_big_num[0], 0:high_data.shape[1]]
    return second_train_list, second_test_list, high_train_list, high_test_list


def compare_pre_time(window_num, data, node_number, high_data):  #data是三列信息，节点，节点，时间
    new_time_list = data[:, -1]
    high_new_time_list = high_data[:, -1]
    time_unique = np.unique(new_time_list) #101个
    per_scale = np.round(len(time_unique)/window_num)#每个窗口的时间间隔
    adj_list = []
    for i in range(window_num):
        up_raw = time_unique[int(per_scale * i)]
        if i != window_num-1:
            down_raw = time_unique[int(per_scale * (i + 1) - 1)]
            raw_num = np.where((new_time_list >= up_raw) & (new_time_list <= down_raw))
            edge_index_list = data[raw_num[0], 0:2]
        else:
            raw_num = np.where(new_time_list >= up_raw)
            edge_index_list = data[raw_num[0], 0:2]
        adj_list_single = sparse.csr_matrix(contruct_adj(edge_index_list, node_number))
        adj_list.append(adj_list_single)
    return adj_list


def contruct_adj(edges, node_num):
    adj = np.zeros(shape=(node_num, node_num))
    for i in range(edges.shape[0]):
        adj[int(edges[i, 0] - 1), int(edges[i, 1] - 1)] = 1
        adj[int(edges[i, 1] - 1), int(edges[i, 0] - 1)] = 1
    return adj


def min_max_process(time_data, num):
    min_time = np.min(time_data)
    max_time = np.max(time_data)
    new_time_list = np.round(((time_data - min_time) / (max_time - min_time)), num)
    return new_time_list


# 构造训练集负样本
def neg_sampling(edge_num, true_sample, big_matrix):
    degree_vector = np.sum(big_matrix, axis=1)
    pos_sample = []
    neg_sample = []
    for i in range(edge_num):
        pos_ID = (true_sample[i, 0:true_sample.shape[1]])
        min_ID = np.where((degree_vector[pos_ID.astype(np.int)]) == np.min(degree_vector[pos_ID.astype(np.int)]))
        pos_ID = np.array(pos_ID)
        mind_node = pos_ID[min_ID]
        # random.seed(1)
        if len(mind_node) >= 2:
            p = mind_node[0]
            not_interact = np.where(big_matrix[p.astype(np.int), :] == 0)
            neg = random.sample(list(not_interact[0]), true_sample.shape[1] - 1)
            neg_sample.append([p] + neg)
            pos_sample.append(pos_ID)
        else:
            not_interact = np.where(big_matrix[mind_node.astype(np.int), :] == 0)
            neg = random.sample(list(not_interact[1]), true_sample.shape[1] - 1)
            neg_sample.append(list(mind_node) + neg)
            pos_sample.append(list(pos_ID))
    return pos_sample, neg_sample


def neg_sampling_random(edge_num, true_sample, adj):
    node_num = adj.shape[0]
    pos_sample = []
    neg_sample = []
    all_node = [i for i in range(node_num)]
    edge_number_diff = (true_sample.shape[1]) * (true_sample.shape[1] - 1) / 2
    a = 0
    for i in range(edge_num):
        pos_ID = list(true_sample[i, 0:true_sample.shape[1]])
        neg = random.sample((set(all_node) - set(pos_ID)), true_sample.shape[1] - 1)
        neg_neg = [pos_ID[0]] + neg
        adj_ab = 0
        for j in range(true_sample.shape[1]):
            for k in range(j + 1, true_sample.shape[1]):
                adj_ab = adj_ab + adj[int(neg_neg[j]), int(neg_neg[k])]
        if adj_ab == edge_number_diff:
            a = a + 1
            print("负样本存在")
        neg_sample.append(neg_neg)
        pos_sample.append(pos_ID)
    print(a)
    return pos_sample, neg_sample


def remove_sim(win_num, pos_samples, neg_samples):
    new_pos = []
    new_neg = []
    for i in range(win_num):
        pos_list = pos_samples[i]
        neg_list = neg_samples[i]
        for j in range(len(pos_list)):
            pos_set = set(pos_list[j])
            for k in range(len(pos_list)):
                if j != k and set(pos_list[k]) == pos_set and pos_set != []:
                    pos_list[k] = []
                    neg_list[k] = []
        new_pos.append(list(filter(None, pos_list)))
        new_neg.append(list(filter(None, neg_list)))
    return new_pos, new_neg