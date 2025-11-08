import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Amazon, Coauthor, HeterophilousGraphDataset, WikiCS, WebKB, Actor
from ogb.nodeproppred import NodePropPredDataset

import numpy as np
import scipy.sparse as sp
from os import path
#from google_drive_downloader import GoogleDriveDownloader as gdd
#import gdown
import scipy
from torch_geometric.datasets import Planetoid

from .WikipedaNetwork import WikipediaNetwork
from .data_utils import dataset_drive_url, rand_train_test_idx, sym_adj, get_partition, get_partition_big


class Dataset(object):
    def __init__(self, name, graph, x, y, node_to_par, P, A_P):
        self.name = name  # original name, e.g., ogbn-proteins
        self.graph = graph
        self.x = x
        self.y = y
        self.node_to_par = node_to_par
        self.P = P
        self.A_P = A_P

    def to(self, device):
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        self.node_to_par = self.node_to_par.to(device)
        #self.graph = self.graph.to(device)
        self.P = self.P.to(device)
        self.A_P = self.A_P.to(device)
        return self


def load_dataset(args):
    data_name = args.dataset_name
    print("Loading {} dataset...".format(data_name))
    data_dir = './data/'

    if data_name in ('amazon-photo', 'amazon-computer'):
        dataset = load_amazon_dataset(data_dir, data_name)
    elif data_name in ('coauthor-cs', 'coauthor-physics'):
        dataset = load_coauthor_dataset(data_dir, data_name)
    elif data_name in ('roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions'): #rating\roman:nt
        dataset = HeterophilousGraphDataset(name=data_name.capitalize(), root=data_dir)
    elif data_name == 'wikics':
        dataset = WikiCS(root=f'{data_dir}/wikics/')
    elif data_name in ('ogbn-arxiv', 'ogbn-products'):
        dataset = load_ogb_dataset(data_dir, data_name)
    elif data_name == 'pokec':
        dataset = load_pokec_mat(data_dir)
    elif data_name in ('cora', 'citeseer', 'pubmed'):
        dataset = Planetoid(root=f'{data_dir}/Planetoid', name=data_name, transform=T.NormalizeFeatures())
    elif data_name in ["cornell", "texas", "wisconsin"]:
        dataset = WebKB(root=f'{data_dir}/webkb/', name=data_name)
    elif data_name == 'film':
        dataset = Actor(root=f'{data_dir}/film/', transform=T.NormalizeFeatures())
    elif data_name in ['crocodile']:
        dataset = WikipediaNetwork(root=f'{data_dir}/Crocodile', name=data_name,
                                   geom_gcn_preprocess=False, transform=T.NormalizeFeatures())
    else:
        raise ValueError('Invalid dataname')
    data = dataset[0]

    graph = sym_adj(data.edge_index, data.num_nodes)
    if args.big:
        node_to_par, P, A_P = get_partition_big(graph, data.edge_index, data.num_nodes,
                                            round(args.cluster * data.x.size(0)), data_dir, data_name)
    else:
        node_to_par, P, A_P = get_partition(graph, data.edge_index, data.num_nodes, round(args.cluster * data.x.size(0)))
    return Dataset(data_name, graph, data.x, data.y, node_to_par, P, A_P)


def load_large_dataset(args):
    data_name = args.dataset_name
    print("Loading {} dataset...".format(data_name))
    data_dir = './data/'

    if data_name in ['ogbn-arxiv', 'ogbn-products']:
        dataset = NodePropPredDataset(root=f'{data_dir}/ogb/',name=data_name)
    else:
        raise ValueError('Invalid dataname')

    data, y = dataset[0]
    num_nodes = data['num_nodes']
    data['edge_index'] = torch.as_tensor(data['edge_index'])
    y = torch.squeeze(torch.as_tensor(y).reshape(-1, 1), dim=1)
    x = torch.as_tensor(data['node_feat'])

    graph = sym_adj(data['edge_index'], num_nodes)
    node_to_par, P, A_P = get_partition_big(graph, data['edge_index'], num_nodes,
                                            round(args.cluster * x.size(0)), data_dir, data_name)
    return Dataset(data_name, graph, x, y, node_to_par, P, A_P)


def load_amazon_dataset(data_dir, name):
    transform = T.NormalizeFeatures()
    if name == 'amazon-photo':
        dataset = Amazon(root=f'{data_dir}Amazon', name='Photo', transform=transform)
        return dataset
    elif name == 'amazon-computer':
        dataset = Amazon(root=f'{data_dir}Amazon', name='Computers', transform=transform)
        return dataset


def load_coauthor_dataset(data_dir, name):
    transform = T.NormalizeFeatures()
    if name == 'coauthor-cs':
        dataset = Coauthor(root=f'{data_dir}Coauthor', name='CS', transform=transform)
        return dataset
    elif name == 'coauthor-physics':
        dataset = Coauthor(root=f'{data_dir}Coauthor', name='Physics', transform=transform)
        return dataset


def load_ogb_dataset(data_dir, name):
    dataset = Dataset(name)
    ogb_dataset = NodePropPredDataset(name=name, root=f'{data_dir}/ogb')
    dataset.graph = ogb_dataset.graph
    dataset.graph['edge_index'] = torch.as_tensor(dataset.graph['edge_index'])
    dataset.graph['node_feat'] = torch.as_tensor(dataset.graph['node_feat'])

    def ogb_idx_to_tensor():
        split_idx = ogb_dataset.get_idx_split()
        tensor_split_idx = {key: torch.as_tensor(
            split_idx[key]) for key in split_idx}
        return tensor_split_idx
    dataset.load_fixed_splits = ogb_idx_to_tensor  # ogb_dataset.get_idx_split
    dataset.label = torch.as_tensor(ogb_dataset.labels).reshape(-1, 1)
    return dataset


def load_pokec_mat(data_dir):
    """ requires pokec.mat """
    if not path.exists(f'{data_dir}/pokec/pokec.mat'):
        drive_id = '1575QYJwJlj7AWuOKMlwVmMz8FcslUncu'
        gdown.download(id=drive_id, output="data/pokec/")
        #import sys; sys.exit()
        #gdd.download_file_from_google_drive(
        #    file_id= drive_id, \
        #    dest_path=f'{data_dir}/pokec/pokec.mat', showsize=True)

    fulldata = scipy.io.loadmat(f'{data_dir}/pokec/pokec.mat')

    dataset = Dataset('pokec')
    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    node_feat = torch.tensor(fulldata['node_feat']).float()
    num_nodes = int(fulldata['num_nodes'])
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}

    label = fulldata['label'].flatten()
    dataset.label = torch.tensor(label, dtype=torch.long)
    return dataset
