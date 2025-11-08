import copy
import os
import random
import time

import nni
import numpy as np
import torch

from modules.distiller import get_PGD_inputs
from modules.logreg import LogReg
from params import set_params
from model import DualGCL
from utils.dataset import load_dataset
from utils.data_utils import eval_acc, class_rand_splits, load_fixed_splits, rand_train_test_idx, eval_rocauc
from utils.eval import *
from torch_geometric.utils import spmm


def fix_seed(seed=1024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


_kd = False
args = set_params()
fix_seed(args.seed)
print(args)

if torch.cuda.is_available() and args.gpu != -1:
    device = torch.device("cuda:" + str(args.gpu))
    torch.cuda.set_device(args.gpu)
else:
    device = torch.device("cpu")

### Load and preprocess data ###
dataset = load_dataset(args)

if args.rand_split:
    split_idx_lst = [rand_train_test_idx(label=dataset.y, train_prop=args.train_ratio, valid_prop=args.valid_ratio)
                     for _ in range(args.runs)]
elif args.rand_split_class:
    split_idx_lst = [class_rand_splits(
        dataset.y, args.label_num_per_class, args.valid_num, args.test_num)]
else:
    split_idx_lst = load_fixed_splits(name=args.dataset_name)
dataset.y = F.one_hot(dataset.y, dataset.y.max() + 1).squeeze(1)

dataset = dataset.to(device)

aucs = []
times = 0

model = DualGCL(dataset.x.size(1), args.hidden_channels, args.alpha, args.temperature,
                args.layer_norm, args.batch_norm).to(device)
gcl_optimizer = torch.optim.Adam(model.gcl_model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
kd_optimizer = torch.optim.Adam(model.kd_model.parameters(), weight_decay=args.kd_weight_decay, lr=args.kd_lr)

rsd_criterion = torch.nn.MSELoss(reduction='none')
criterion = torch.nn.BCEWithLogitsLoss()

eval_func = eval_rocauc
model.train()

## Training GCL Model
time_start = time.time()
print('Start training GCL model...')
for epoch in range(int(args.epochs) + 1):
    gcl_optimizer.zero_grad()
    loss = model.gcl_model(dataset.x, dataset.node_to_par, dataset.P, dataset.A_P)
    loss.backward()
    gcl_optimizer.step()
    print(loss)
time_end = time.time()
times = time_end - time_start

model.gcl_model.eval()
dataset.graph = dataset.graph.to(device)
if _kd:
    with torch.no_grad():
        adj_emb = model.adj_embed(dataset.x, dataset.graph, args.k_hop)
        gcl_emb = model.gcl_embed(dataset.x)
        cluster_emb = torch.spmm(dataset.P.T, gcl_emb)
        gcl_enc = copy.deepcopy(model.gcl_model.enc)

    ## Training distillation model
    print('Start training Distiller...')
    for epoch in range(args.kd_epochs):
        total_loss = 0
        num_batches = max(1, dataset.x.shape[0] // args.batch_size)
        idx_batch = torch.randperm(dataset.x.shape[0])[: num_batches * args.batch_size]
        if num_batches == 1:
            idx_batch = idx_batch.view(1, -1)
        else:
            idx_batch = idx_batch.view(num_batches, args.batch_size)

        for idx in idx_batch:
            kd_optimizer.zero_grad()
            kd_out = model.kd_model(gcl_emb[idx])
            loss = rsd_criterion(kd_out, adj_emb[idx]).mean() #F.pairwise_distance
            loss.backward()
            kd_optimizer.step()
            total_loss += loss
        if epoch % 50 == 0:
            print(total_loss / num_batches)

print('Start testing...')
model.eval()

with torch.no_grad():
    if _kd:
        emb = model.kd_embed(dataset.x)
    else:
        emb = model.adj_embed(dataset.x, dataset.graph, args.k_hop)

for i in range(args.runs):
    if args.dataset_name in ('cora', 'citeseer', 'pubmed'):
        split_idx = split_idx_lst[0]
    else:
        split_idx = split_idx_lst[i]
    train_idx = split_idx['train'].to(device)
    valid_idx = split_idx['valid'].to(device)
    test_idx = split_idx['test'].to(device)

    best_auc_val = 0
    best_loss_val = 1e9
    final_test = 0
    logreg = LogReg(args.hidden_channels, dataset.y.max()+1).to(device)
    optimizer = torch.optim.Adam(logreg.parameters(), lr=args.cls_lr, weight_decay=args.cls_weight_decay)

    for _ in range(args.cls_epochs):
        logreg.train()
        optimizer.zero_grad()
        out = logreg(emb[train_idx])
        loss = criterion(out, dataset.y.squeeze(1)[train_idx].to(torch.float))
        loss.backward()
        optimizer.step()

        logreg.eval()
        prob = logreg(emb)
        loss_val = criterion(prob[valid_idx], dataset.y.squeeze(1)[valid_idx].to(torch.float))
        auc_val = eval_func(prob[valid_idx], dataset.y[valid_idx])
        auc_test = eval_func(prob[test_idx], dataset.y[test_idx])

        if auc_val >= best_auc_val and best_loss_val >= loss_val:
            #print("better classification!")
            best_auc_val = max(auc_val, best_auc_val)
            best_loss_val = loss_val
            final_test = max(auc_test, final_test)

    aucs.append(final_test.item())
    print(f'Run: {i:02d}, ' f'Test Accuracy: {final_test*100:.2f}')

print(f'Test ROC: {np.mean(aucs)*100:.2f} ± {np.std(aucs)*100:.2f}')
print(f'Time per Epoch: {times/args.epochs:.4f}s, ' f'Total Time: {times:.4f}s')

os._exit(0)