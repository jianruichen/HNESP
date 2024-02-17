from __future__ import division
from __future__ import print_function
import time
import argparse
import numpy as np
import torch
from sympy import *
import torch.nn.functional as F
from pygcn.pool import *
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, average_precision_score
from pygcn.utils import load_data
from pygcn.models import GCN
from pygcn.mutual_information import pool_mutual
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import warnings
warnings.filterwarnings(action='ignore')

# Training settings======================NdeGCN
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='DAWN', help="available datasets: "
       "[congress-bills, email-Eu, tags-math-sx, contact-primary-school, DAWN, NDC-substances, "
       "threads-ask-ubuntu, email-Enron, NDC-classes, tags-ask-ubuntu]")
parser.add_argument('--node_number', type=int, default=2290, help='number of nodes.')
parser.add_argument('--pre_order', type=str, default='three', help="predict order")
parser.add_argument('--no-cuda', action='store_true', default=True, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=201, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout1', type=float, default=0.3, help='Dropout rate (1 - keep probability).')
parser.add_argument('--initial_dim', type=int, default=32,  help='Dimensions of initial features.')
parser.add_argument('--hidden_dim1', type=int, default=64, help='Dimensions of hidden units.')
parser.add_argument('--output_dim', type=int, default=128, help='Dimensions of output layer.')
parser.add_argument('--mlp_hidden_dim1', type=int, default=128, help='Dimensions of output layer.')
parser.add_argument('--mlp_hidden_dim2', type=int, default=32, help='Dimensions of output layer.')
parser.add_argument('--mlp_output_dim', type=int, default=1, help='Dimensions of output layer.')
parser.add_argument('--pool_mutual', action='store_true', default=True, help='pool mutual information or not.')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj_matrix, lap_matrix_p, lap_matrix_n, incidence_matrix_preoder, node_number,  \
neg_train_sample, pos_train_sample, neg_test_sample, pos_test_sample = load_data(args.node_number, args.dataset, args.pre_order)

model = GCN(nfeat=args.initial_dim,
            nhid1=args.hidden_dim1,
            noutput=args.output_dim,
            dropout1=args.dropout1,
            node_num=node_number,
            n_feature=args.output_dim * len(pos_train_sample[0]) + 2 ** len(pos_train_sample[0]),
            n_hidden1=args.mlp_hidden_dim1,
            n_hidden2=args.mlp_hidden_dim2,
            n_output=args.mlp_output_dim)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)#, eps=1e-1


def train(epoch):
    model.train()
    optimizer.zero_grad()
    output = model.forward1(lap_matrix_p, lap_matrix_n, incidence_matrix_preoder)
    feature_pooling, order_pooling = pooling_method1(output, pos_train_sample, neg_train_sample, len(pos_train_sample))
    if args.pool_mutual:
        mutual_vector = pool_mutual(output, pos_train_sample, neg_train_sample, len(pos_train_sample))
        feature_pooling = torch.cat((feature_pooling, mutual_vector), dim=1)
    order_feature, order_labels = perturb_feature(feature_pooling, order_pooling)
    MLP_output = model.forward2(order_feature.to(torch.float32))
    loss_train = compute_loss(MLP_output, order_labels)
    AUC_train, AUC_PR_train, AP_train = metrics(MLP_output.detach(), order_labels)
    loss_train.backward()
    optimizer.step()
    print(f'EPOCH[{epoch}/{args.epochs}]')
    print(f"[train_lOSS{epoch}] {loss_train}",
          f"[train_AUC{epoch}] {AUC_train, AUC_PR_train, AP_train}")
    if epoch % 10 == 0:
        loss_test, AUC_test, AUC_PR_test, AP_test = test(output)
        print('==' * 27)
        print(f"[test_lOSS{epoch}] {loss_test}",
              f"[test_AUC{epoch}] {AUC_test, AUC_PR_test, AP_test}"
              )
        print('==' * 27)
    return loss_train, AUC_train, AUC_PR_train, AP_train


def test(final_feature_matrix=None):
    model.eval()
    final_feature_matrix = final_feature_matrix.detach()
    with torch.no_grad():
        feature_pooling, order_pooling = pooling_method1(final_feature_matrix, pos_test_sample, neg_test_sample, len(pos_test_sample))
        if args.pool_mutual:
            mutual_vector = pool_mutual(final_feature_matrix, pos_test_sample, neg_test_sample, len(pos_test_sample))
            feature_pooling = torch.cat((feature_pooling, mutual_vector), dim=1)
        order_feature, order_labels = perturb_feature(feature_pooling, order_pooling)
        MLP_output = model.forward2(order_feature.to(torch.float32))
        loss_test = compute_loss(MLP_output, order_labels)
        AUC_test, AUC_PR_test, AP_test = metrics(MLP_output.detach(), order_labels)
    return loss_test, AUC_test, AUC_PR_test, AP_test


def compute_loss(MLP_output, order_labels):
    loss1 = F.binary_cross_entropy_with_logits(MLP_output, order_labels)
    reg_loss = (1 / 2) * ((model.gc1.weight.norm(2)) + (model.gc2.weight.norm(2)))#+ (model.gc3.weight.norm(2))
    reg_loss = reg_loss * args.weight_decay
    loss = loss1 + reg_loss
    return loss


def metrics(MLP_output, order_labels):
    AUC_value = roc_auc_score(order_labels, MLP_output)
    precision, recall, thresholds = precision_recall_curve(order_labels, MLP_output)
    AUC_PR_value = auc(recall, precision)
    AP_value = average_precision_score(order_labels, MLP_output)
    return AUC_value, AUC_PR_value, AP_value


# Train model
loss_vector = []
AUC_PR_vector = []
AUC_vector = []
AP_vector = []
start = time.time()
for epoch in range(args.epochs):
    loss_train, AUC_train, AUC_PR_train, AP_train = train(epoch)
    loss_vector.append(loss_train.item())
    AUC_PR_vector.append(AUC_PR_train)
    AUC_vector.append(AUC_train)
    AP_vector.append(AP_train)
print('======================')
print("Optimization Finished!")
t_total = time.time()
final_time = t_total - start
print(final_time)



