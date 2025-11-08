import os

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from ogb.nodeproppred import NodePropPredDataset
from torch_geometric.datasets import HeterophilousGraphDataset, WikiCS, Actor, WebKB
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
import torch_sparse
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
import scipy.sparse as sp

from utils.WikipedaNetwork import WikipediaNetwork


def rand_train_test_idx(label, train_prop=0.5, valid_prop=0.25, ignore_negative=True):
    """randomly splits label into train/valid/test splits"""
    if ignore_negative:
        labeled_nodes = torch.where(label != -1)[0]
    else:
        labeled_nodes = label

    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]
    val_indices = perm[train_num : train_num + valid_num]
    test_indices = perm[train_num + valid_num :]

    if not ignore_negative:
        return train_indices, val_indices, test_indices

    train_idx = labeled_nodes[train_indices]
    valid_idx = labeled_nodes[val_indices]
    test_idx = labeled_nodes[test_indices]
    split_idx = {"train": train_idx, "valid": valid_idx, "test": test_idx}
    return split_idx

def class_rand_splits(label, label_num_per_class, valid_num=500, test_num=1000):
    """use all remaining data points as test data, so test_num will not be used"""
    train_idx, non_train_idx = [], []
    idx = torch.arange(label.shape[0])
    class_list = label.squeeze().unique()
    for i in range(class_list.shape[0]):
        c_i = class_list[i]
        idx_i = idx[label.squeeze() == c_i]
        n_i = idx_i.shape[0]
        rand_idx = idx_i[torch.randperm(n_i)]
        train_idx += rand_idx[:label_num_per_class].tolist()
        non_train_idx += rand_idx[label_num_per_class:].tolist()
    train_idx = torch.as_tensor(train_idx)
    non_train_idx = torch.as_tensor(non_train_idx)
    non_train_idx = non_train_idx[torch.randperm(non_train_idx.shape[0])]
    valid_idx, test_idx = (
        non_train_idx[:valid_num],
        non_train_idx[valid_num : valid_num + test_num],
    )
    print(f"train:{train_idx.shape}, valid:{valid_idx.shape}, test:{test_idx.shape}")
    split_idx = {"train": train_idx, "valid": valid_idx, "test": test_idx}
    return split_idx



def load_fixed_splits(name):
    data_dir = './data/'
    splits_lst = []
    if name in ['roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions']:
        dataset = HeterophilousGraphDataset(name=name.capitalize(), root=data_dir)
        data = dataset[0]
        for i in range(data.train_mask.shape[1]):
            splits = {}
            splits['train'] = torch.where(data.train_mask[:,i])[0]
            splits['valid'] = torch.where(data.val_mask[:,i])[0]
            splits['test'] = torch.where(data.test_mask[:,i])[0]
            print(f"train:{splits['train'].shape}, valid:{ splits['valid'].shape}, test:{splits['test'].shape}")
            splits_lst.append(splits)
    elif name in ["cornell", "texas", "wisconsin"]:
        dataset = WebKB(root=f'{data_dir}/webkb/', name=name)
        data = dataset[0]
        for i in range(data.train_mask.shape[1]):
            splits = {}
            splits['train'] = torch.where(data.train_mask[:,i])[0]
            splits['valid'] = torch.where(data.val_mask[:,i])[0]
            splits['test'] = torch.where(data.test_mask[:,i])[0]
            splits_lst.append(splits)
    elif name in ["film"]:
        dataset = Actor(root=f'{data_dir}/film/', transform=T.NormalizeFeatures())
        data = dataset[0]
        for i in range(data.train_mask.shape[1]):
            splits = {}
            splits['train'] = torch.where(data.train_mask[:,i])[0]
            splits['valid'] = torch.where(data.val_mask[:,i])[0]
            splits['test'] = torch.where(data.test_mask[:,i])[0]
            splits_lst.append(splits)
    elif name in ['crocodile']:
        dataset = WikipediaNetwork(root=f'{data_dir}/Crocodile', name=name, geom_gcn_preprocess=False)
        data = dataset[0]
        for i in range(data.train_mask.shape[1]):
            splits = {}
            splits['train'] = torch.where(data.train_mask[:,i])[0]
            splits['valid'] = torch.where(data.val_mask[:,i])[0]
            splits['test'] = torch.where(data.test_mask[:,i])[0]
            print(splits['train'].size(0), splits['valid'].size(0), splits['test'].size(0))
            splits_lst.append(splits)
    elif name in ['ogbn-arxiv', 'ogbn-products']:
        dataset = NodePropPredDataset(root=f'{data_dir}/ogb/', name=name)
        for i in range(10):
            splits = {}
            split_idx = dataset.get_idx_split()
            splits['train'] = torch.as_tensor(split_idx['train'])
            splits['valid'] = torch.as_tensor(split_idx['valid'])
            splits['test'] = torch.as_tensor(split_idx['test'])
            splits_lst.append(splits)
    elif name in ['pokec']:
        split = np.load(f'{data_dir}/{name}/{name}-splits.npy', allow_pickle=True)
        for i in range(split.shape[0]):
            splits = {}
            splits['train'] = torch.from_numpy(np.asarray(split[i]['train']))
            splits['valid'] = torch.from_numpy(np.asarray(split[i]['valid']))
            splits['test'] = torch.from_numpy(np.asarray(split[i]['test']))
            splits_lst.append(splits)
    elif name in ["chameleon", "squirrel"]:
        file_path = f"{data_dir}/geom-gcn/{name}/{name}_filtered.npz"
        data = np.load(file_path)
        train_masks = data["train_masks"]  # (10, N), 10 splits
        val_masks = data["val_masks"]
        test_masks = data["test_masks"]
        N = train_masks.shape[1]

        node_idx = np.arange(N)
        for i in range(10):
            splits = {}
            splits["train"] = torch.as_tensor(node_idx[train_masks[i]])
            splits["valid"] = torch.as_tensor(node_idx[val_masks[i]])
            splits["test"] = torch.as_tensor(node_idx[test_masks[i]])
            splits_lst.append(splits)
    else:
        raise NotImplementedError

    return splits_lst


def eval_f1(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        f1 = f1_score(y_true, y_pred, average='micro')
        acc_list.append(f1)

    return sum(acc_list)/len(acc_list)


def eval_acc(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def eval_rocauc(y_pred, y_true):
    rocauc_list = []
    y_true = y_true.detach().cpu().numpy()
    if y_true.shape[1] == 1:
        # use the predicted class for single-class classification
        y_pred = F.softmax(y_pred, dim=-1)[:,1].unsqueeze(1).cpu().numpy()
    else:
        y_pred = y_pred.detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            score = roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])

            rocauc_list.append(score)

    if len(rocauc_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute ROC-AUC.')

    return sum(rocauc_list)/len(rocauc_list)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def sparse_mx_to_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a sparse tensor.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    rows = torch.from_numpy(sparse_mx.row).long()
    cols = torch.from_numpy(sparse_mx.col).long()
    values = torch.from_numpy(sparse_mx.data)
    return torch_sparse.SparseTensor(row=rows, col=cols, value=values, sparse_sizes=torch.tensor(sparse_mx.shape))


def asym_adj(edge_index, num_nodes):
    edge_index = to_undirected(edge_index)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    adj = sp.csr_matrix(([1]*edge_index.size(1), (edge_index[0].numpy(), edge_index[1].numpy())),
                        shape=(num_nodes, num_nodes), dtype=np.float32)
    return sparse_mx_to_sparse_tensor(adj)


def sym_adj(edge_index, num_nodes):
    edge_index = to_undirected(edge_index)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    adj = sp.csr_matrix(([1]*edge_index.size(1), (edge_index[0].numpy(), edge_index[1].numpy())),
                        shape=(num_nodes, num_nodes), dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def get_partition(graph, edge_index, nodes, clusters):
    edge_index = to_undirected(edge_index)
    edge_index, _ = remove_self_loops(edge_index)
    adj = torch_sparse.SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(nodes, nodes))
    _, partptr, perm = adj.partition(clusters, recursive=False)
    partptr = partptr.tolist()
    perm = perm.tolist()
    partptr = list(set(partptr))
    partptr.sort()

    clusters = len(partptr) - 1
    node_to_par = torch.zeros(nodes, dtype=torch.long)
    par_value = torch.ones(nodes)
    for i in range(clusters):
        start_idx = partptr[i]
        end_idx = partptr[i + 1]
        nodes_size = end_idx - start_idx
        node_to_par[perm[start_idx:end_idx]] = i
        par_value[perm[start_idx:end_idx]] = 1 / nodes_size
    par_indices = torch.stack([torch.arange(nodes), node_to_par])
    partition = torch.sparse.FloatTensor(par_indices, par_value, (nodes, clusters))

    P_A = torch.spmm(partition.T, graph)
    A_P = torch.spmm(P_A, partition)

    return node_to_par, partition, A_P.to_dense()

def get_partition_big(graph, edge_index, nodes, clusters, data_dir, data_name):
    edge_index = to_undirected(edge_index)
    edge_index, _ = remove_self_loops(edge_index)
    adj = torch_sparse.SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(nodes, nodes))

    par_dir = f'{data_dir}partition/' + data_name + '/' + str(clusters)
    if (os.path.exists(par_dir +"/partition.pt")):
        partition = torch.load(par_dir +"/partition.pt")
        node_to_par = partition.coalesce().indices()[1].clone()
    else:
        _, partptr, perm = adj.partition(clusters, recursive=False)
        partptr = partptr.tolist()
        perm = perm.tolist()
        partptr = list(set(partptr))
        partptr.sort()

        clusters = len(partptr) - 1
        node_to_par = torch.zeros(nodes, dtype=torch.long)
        par_value = torch.ones(nodes)
        for i in range(clusters):
            start_idx = partptr[i]
            end_idx = partptr[i + 1]
            nodes_size = end_idx - start_idx
            node_to_par[perm[start_idx:end_idx]] = i
            par_value[perm[start_idx:end_idx]] = 1 / nodes_size
        par_indices = torch.stack([torch.arange(nodes), node_to_par])
        partition = torch.sparse.FloatTensor(par_indices, par_value, (nodes, clusters))
        par_path = os.path.join(par_dir, 'partition.pt')
        os.makedirs(os.path.dirname(par_path), exist_ok=True)
        torch.save(partition, par_dir +"/partition.pt")

    P_A = torch.spmm(partition.T, graph)
    A_P = torch.spmm(P_A, partition)

    return node_to_par, partition, A_P.to_dense()

dataset_drive_url = {
    'snap-patents' : '1ldh23TSY1PwXia6dU0MYcpyEgX-w3Hia',
    'pokec' : '1dNs5E7BrWJbgcHeQ_zuy5Ozp2tRCWG0y',
    'yelp-chi': '1fAXtTVQS4CfEk4asqrFw9EPmlUPGbGtJ',
}

