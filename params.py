import argparse
import sys

argv = sys.argv
dataset = argv[1]

def cora_params():###################
    parser = argparse.ArgumentParser()
    #####################################
    ## basic info
    parser.add_argument('--dataset_name', type=str, default=dataset)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=123)

    parser.add_argument('--rand_split_class', type=bool, default=True,
                        help='use random splits with a fixed number of labeled nodes for each class')
    parser.add_argument('--rand_split', type=bool, default=False)
    parser.add_argument('--label_num_per_class', type=int, default=20,
                        help='labeled nodes per class(randomly selected)')
    parser.add_argument('--valid_num', type=int, default=500,
                        help='Total number of validation')
    parser.add_argument('--test_num', type=int, default=1000,
                        help='Total number of test')
    parser.add_argument('--train_ratio', type=float, default=.1,
                        help='training label proportion')
    parser.add_argument('--valid_ratio', type=float, default=.1,
                        help='validation label proportion')

    ## model
    parser.add_argument('--k_hop', type=int, default=10)
    parser.add_argument('--cluster', type=float, default=0.09)
    parser.add_argument('--hidden_channels', type=int, default=1024)#1024
    parser.add_argument('--temperature', type=float, default=0.09)
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--batch_norm', type=bool, default=False)
    parser.add_argument('--layer_norm', type=bool, default=True)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=4096)

    ## optimizer
    parser.add_argument('--epochs', type=int, default=15) #5
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=0.)

    parser.add_argument('--kd_epochs', type=int, default=500)
    parser.add_argument('--kd_lr', type=float, default=0.01)
    parser.add_argument('--kd_weight_decay', type=float, default=0.00005)

    parser.add_argument('--cls_epochs', type=int, default=600)
    parser.add_argument('--cls_lr', type=float, default=0.1)
    parser.add_argument('--cls_weight_decay', type=float, default=0.05)
    #####################################
    args, _ = parser.parse_known_args()
    return args


def citeseer_params():
    parser = argparse.ArgumentParser()
    #####################################
    ## basic info
    parser.add_argument('--dataset_name', type=str, default=dataset)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=123)

    parser.add_argument('--rand_split_class', type=bool, default=True,
                        help='use random splits with a fixed number of labeled nodes for each class')
    parser.add_argument('--rand_split', type=bool, default=False)
    parser.add_argument('--label_num_per_class', type=int, default=20,
                        help='labeled nodes per class(randomly selected)')
    parser.add_argument('--valid_num', type=int, default=500,
                        help='Total number of validation')
    parser.add_argument('--test_num', type=int, default=1000,
                        help='Total number of test')
    parser.add_argument('--train_ratio', type=float, default=.1,
                        help='training label proportion')
    parser.add_argument('--valid_ratio', type=float, default=.1,
                        help='validation label proportion')

    ## model
    parser.add_argument('--k_hop', type=int, default=3)
    parser.add_argument('--cluster', type=float, default=0.07)
    parser.add_argument('--hidden_channels', type=int, default=2048)
    parser.add_argument('--temperature', type=float, default=0.04)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--batch_norm', type=bool, default=False)
    parser.add_argument('--layer_norm', type=bool, default=False)
    parser.add_argument('--dropout', type=float, default=0.6)#0.15 #0.6
    parser.add_argument('--batch_size', type=int, default=4096)

    ## optimizer
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--weight_decay', type=float, default=0.00001)

    parser.add_argument('--kd_epochs', type=int, default=1200)
    parser.add_argument('--kd_lr', type=float, default=0.0001)
    parser.add_argument('--kd_weight_decay', type=float, default=0)

    parser.add_argument('--cls_epochs', type=int, default=1000)
    parser.add_argument('--cls_lr', type=float, default=0.0001)
    parser.add_argument('--cls_weight_decay', type=float, default=0.000005)
    #####################################
    args, _ = parser.parse_known_args()
    return args


def pubmed_params():
    parser = argparse.ArgumentParser()
    #####################################
    ## basic info
    parser.add_argument('--dataset_name', type=str, default=dataset)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1024)

    parser.add_argument('--rand_split_class', type=bool, default=True,
                        help='use random splits with a fixed number of labeled nodes for each class')
    parser.add_argument('--rand_split', type=bool, default=False)
    parser.add_argument('--label_num_per_class', type=int, default=20,
                        help='labeled nodes per class(randomly selected)')
    parser.add_argument('--valid_num', type=int, default=500,
                        help='Total number of validation')
    parser.add_argument('--test_num', type=int, default=1000,
                        help='Total number of test')
    parser.add_argument('--train_ratio', type=float, default=.1,
                        help='training label proportion')
    parser.add_argument('--valid_ratio', type=float, default=.1,
                        help='validation label proportion')

    ## model
    parser.add_argument('--k_hop', type=int, default=2)
    parser.add_argument('--cluster', type=float, default=0.01)
    parser.add_argument('--hidden_channels', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=0.08)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--batch_norm', type=bool, default=False)
    parser.add_argument('--layer_norm', type=bool, default=True)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=4096)

    ## optimizer
    parser.add_argument('--epochs', type=int, default=75)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--weight_decay', type=float, default=0.05)

    parser.add_argument('--kd_epochs', type=int, default=800)
    parser.add_argument('--kd_lr', type=float, default=0.0005)
    parser.add_argument('--kd_weight_decay', type=float, default=0.00001)

    parser.add_argument('--cls_epochs', type=int, default=600)
    parser.add_argument('--cls_lr', type=float, default=0.001)
    parser.add_argument('--cls_weight_decay', type=float, default=0.005)
    #####################################
    args, _ = parser.parse_known_args()
    return args


def wikics_params():
    parser = argparse.ArgumentParser()
    #####################################
    ## basic info
    parser.add_argument('--dataset_name', type=str, default=dataset)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--rand_split_class', type=bool, default=True,
                        help='use random splits with a fixed number of labeled nodes for each class')
    parser.add_argument('--rand_split', type=bool, default=False)
    parser.add_argument('--label_num_per_class', type=int, default=20,
                        help='labeled nodes per class(randomly selected)')
    parser.add_argument('--valid_num', type=int, default=500,
                        help='Total number of validation')
    parser.add_argument('--test_num', type=int, default=1000,
                        help='Total number of test')
    parser.add_argument('--train_ratio', type=float, default=.1,
                        help='training label proportion')
    parser.add_argument('--valid_ratio', type=float, default=.1,
                        help='validation label proportion')

    ## model
    parser.add_argument('--k_hop', type=int, default=3)
    parser.add_argument('--cluster', type=float, default=0.02)
    parser.add_argument('--hidden_channels', type=int, default=1024) #1024
    parser.add_argument('--temperature', type=float, default=0.08)
    parser.add_argument('--alpha', type=float, default=0.7)
    parser.add_argument('--batch_norm', type=bool, default=False)
    parser.add_argument('--layer_norm', type=bool, default=True)
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--batch_size', type=int, default=4096)

    ## optimizer
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--weight_decay', type=float, default=0.00001)

    parser.add_argument('--kd_epochs', type=int, default=1200)
    parser.add_argument('--kd_lr', type=float, default=0.0005)
    parser.add_argument('--kd_weight_decay', type=float, default=0.)

    parser.add_argument('--cls_epochs', type=int, default=1000)
    parser.add_argument('--cls_lr', type=float, default=0.05)
    parser.add_argument('--cls_weight_decay', type=float, default=0.005)
    #####################################
    args, _ = parser.parse_known_args()
    return args


def amazon_computer_params():
    parser = argparse.ArgumentParser()
    #####################################
    ## basic info
    parser.add_argument('--dataset_name', type=str, default=dataset)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=123) #1024

    parser.add_argument('--rand_split_class', type=bool, default=True,
                        help='use random splits with a fixed number of labeled nodes for each class')
    parser.add_argument('--rand_split', type=bool, default=False)
    parser.add_argument('--label_num_per_class', type=int, default=20,
                        help='labeled nodes per class(randomly selected)')
    parser.add_argument('--valid_num', type=int, default=500,
                        help='Total number of validation')
    parser.add_argument('--test_num', type=int, default=1000,
                        help='Total number of test')
    parser.add_argument('--train_ratio', type=float, default=.1,
                        help='training label proportion')
    parser.add_argument('--valid_ratio', type=float, default=.1,
                        help='validation label proportion')

    ## model
    parser.add_argument('--k_hop', type=int, default=3)
    parser.add_argument('--cluster', type=float, default=0.03)
    parser.add_argument('--hidden_channels', type=int, default=1024)
    parser.add_argument('--temperature', type=float, default=0.06)
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--batch_norm', type=bool, default=False)
    parser.add_argument('--layer_norm', type=bool, default=True)
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--batch_size', type=int, default=4096)

    ## optimizer
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=0.00001)

    parser.add_argument('--kd_epochs', type=int, default=1000)
    parser.add_argument('--kd_lr', type=float, default=0.0005)
    parser.add_argument('--kd_weight_decay', type=float, default=0.)

    parser.add_argument('--cls_epochs', type=int, default=1000)
    parser.add_argument('--cls_lr', type=float, default=0.1)
    parser.add_argument('--cls_weight_decay', type=float, default=0.001)
    #####################################
    args, _ = parser.parse_known_args()
    return args


def amazon_photo_params():
    parser = argparse.ArgumentParser()
    #####################################
    ## basic info
    parser.add_argument('--dataset_name', type=str, default=dataset)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--rand_split_class', type=bool, default=True,
                        help='use random splits with a fixed number of labeled nodes for each class')
    parser.add_argument('--rand_split', type=bool, default=False)
    parser.add_argument('--label_num_per_class', type=int, default=20,
                        help='labeled nodes per class(randomly selected)')
    parser.add_argument('--valid_num', type=int, default=500,
                        help='Total number of validation')
    parser.add_argument('--test_num', type=int, default=1000,
                        help='Total number of test')
    parser.add_argument('--train_ratio', type=float, default=.1,
                        help='training label proportion')
    parser.add_argument('--valid_ratio', type=float, default=.1,
                        help='validation label proportion')

    ## model
    parser.add_argument('--k_hop', type=int, default=5)
    parser.add_argument('--cluster', type=float, default=0.03)
    parser.add_argument('--hidden_channels', type=int, default=1024)
    parser.add_argument('--temperature', type=float, default=0.05) #0.08
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--batch_norm', type=bool, default=False)
    parser.add_argument('--layer_norm', type=bool, default=True)
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--batch_size', type=int, default=4096)

    ## optimizer
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.000005)

    parser.add_argument('--kd_epochs', type=int, default=1000)
    parser.add_argument('--kd_lr', type=float, default=0.0005)
    parser.add_argument('--kd_weight_decay', type=float, default=0.)

    parser.add_argument('--cls_epochs', type=int, default=1000)
    parser.add_argument('--cls_lr', type=float, default=0.001)
    parser.add_argument('--cls_weight_decay', type=float, default=0.0005)
    #####################################
    args, _ = parser.parse_known_args()
    return args


def coauthor_cs_params():
    parser = argparse.ArgumentParser()
    #####################################
    ## basic info
    parser.add_argument('--dataset_name', type=str, default=dataset)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--rand_split_class', type=bool, default=True,
                        help='use random splits with a fixed number of labeled nodes for each class')
    parser.add_argument('--rand_split', type=bool, default=False)
    parser.add_argument('--label_num_per_class', type=int, default=20,
                        help='labeled nodes per class(randomly selected)')
    parser.add_argument('--valid_num', type=int, default=500,
                        help='Total number of validation')
    parser.add_argument('--test_num', type=int, default=1000,
                        help='Total number of test')
    parser.add_argument('--train_ratio', type=float, default=.1,
                        help='training label proportion')
    parser.add_argument('--valid_ratio', type=float, default=.1,
                        help='validation label proportion')

    ## model
    parser.add_argument('--k_hop', type=int, default=1)
    parser.add_argument('--cluster', type=float, default=0.09)
    parser.add_argument('--hidden_channels', type=int, default=1024)
    parser.add_argument('--temperature', type=float, default=0.08)
    parser.add_argument('--alpha', type=float, default=0.8)
    parser.add_argument('--batch_norm', type=bool, default=False)
    parser.add_argument('--layer_norm', type=bool, default=True)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=4096)

    ## optimizer
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=0.001)

    parser.add_argument('--kd_epochs', type=int, default=600)
    parser.add_argument('--kd_lr', type=float, default=0.001)
    parser.add_argument('--kd_weight_decay', type=float, default=0.)

    parser.add_argument('--cls_epochs', type=int, default=1000)
    parser.add_argument('--cls_lr', type=float, default=0.001)
    parser.add_argument('--cls_weight_decay', type=float, default=0.001)
    #####################################
    args, _ = parser.parse_known_args()
    return args


def coauthor_physics_params():
    parser = argparse.ArgumentParser()
    #####################################
    ## basic info
    parser.add_argument('--dataset_name', type=str, default=dataset)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1024) #42

    parser.add_argument('--rand_split_class', type=bool, default=True,
                        help='use random splits with a fixed number of labeled nodes for each class')
    parser.add_argument('--rand_split', type=bool, default=False)
    parser.add_argument('--label_num_per_class', type=int, default=20,
                        help='labeled nodes per class(randomly selected)')
    parser.add_argument('--valid_num', type=int, default=500,
                        help='Total number of validation')
    parser.add_argument('--test_num', type=int, default=1000,
                        help='Total number of test')
    parser.add_argument('--train_ratio', type=float, default=.1,
                        help='training label proportion')
    parser.add_argument('--valid_ratio', type=float, default=.1,
                        help='validation label proportion')

    ## model
    parser.add_argument('--k_hop', type=int, default=1)
    parser.add_argument('--cluster', type=float, default=0.04)
    parser.add_argument('--hidden_channels', type=int, default=2048) #2048
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.7)
    parser.add_argument('--batch_norm', type=bool, default=False)
    parser.add_argument('--layer_norm', type=bool, default=True)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--batch_size', type=int, default=4096)

    ## optimizer
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--weight_decay', type=float, default=0.00005)

    parser.add_argument('--kd_epochs', type=int, default=500)
    parser.add_argument('--kd_lr', type=float, default=0.0005)
    parser.add_argument('--kd_weight_decay', type=float, default=0)

    parser.add_argument('--cls_epochs', type=int, default=1000)
    parser.add_argument('--cls_lr', type=float, default=0.01)
    parser.add_argument('--cls_weight_decay', type=float, default=0.005)
    #####################################
    args, _ = parser.parse_known_args()
    return args


def cornell_params(): #no_normalized
    parser = argparse.ArgumentParser()
    #####################################
    ## basic info
    parser.add_argument('--dataset_name', type=str, default=dataset)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1024)

    parser.add_argument('--rand_split_class', type=bool, default=True,
                        help='use random splits with a fixed number of labeled nodes for each class')
    parser.add_argument('--rand_split', type=bool, default=False)
    parser.add_argument('--label_num_per_class', type=int, default=20,
                        help='labeled nodes per class(randomly selected)')
    parser.add_argument('--valid_num', type=int, default=500,
                        help='Total number of validation')
    parser.add_argument('--test_num', type=int, default=1000,
                        help='Total number of test')
    parser.add_argument('--train_ratio', type=float, default=.1,
                        help='training label proportion')
    parser.add_argument('--valid_ratio', type=float, default=.1,
                        help='validation label proportion')

    ## model
    parser.add_argument('--k_hop', type=int, default=0)
    parser.add_argument('--cluster', type=float, default=0.2) #0.02
    parser.add_argument('--hidden_channels', type=int, default=8192)
    parser.add_argument('--temperature', type=float, default=0.03)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--batch_norm', type=bool, default=False)
    parser.add_argument('--layer_norm', type=bool, default=False)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=4096)

    ## optimizer
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--weight_decay', type=float, default=0.00005)

    parser.add_argument('--kd_epochs', type=int, default=600)
    parser.add_argument('--kd_lr', type=float, default=0.01)
    parser.add_argument('--kd_weight_decay', type=float, default=0.000001)

    parser.add_argument('--cls_epochs', type=int, default=1200)
    parser.add_argument('--cls_lr', type=float, default=0.05)
    parser.add_argument('--cls_weight_decay', type=float, default=0.005)
    #####################################
    args, _ = parser.parse_known_args()
    return args


def texas_params():
    parser = argparse.ArgumentParser()
    #####################################
    ## basic info
    parser.add_argument('--dataset_name', type=str, default=dataset)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1024)

    parser.add_argument('--rand_split_class', type=bool, default=True,
                        help='use random splits with a fixed number of labeled nodes for each class')
    parser.add_argument('--rand_split', type=bool, default=False)
    parser.add_argument('--label_num_per_class', type=int, default=20,
                        help='labeled nodes per class(randomly selected)')
    parser.add_argument('--valid_num', type=int, default=500,
                        help='Total number of validation')
    parser.add_argument('--test_num', type=int, default=1000,
                        help='Total number of test')
    parser.add_argument('--train_ratio', type=float, default=.1,
                        help='training label proportion')
    parser.add_argument('--valid_ratio', type=float, default=.1,
                        help='validation label proportion')

    ## model
    parser.add_argument('--k_hop', type=int, default=0)
    parser.add_argument('--cluster', type=float, default=0.05)
    parser.add_argument('--hidden_channels', type=int, default=8192)
    parser.add_argument('--temperature', type=float, default=0.04)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--batch_norm', type=bool, default=False)
    parser.add_argument('--layer_norm', type=bool, default=False)
    parser.add_argument('--dropout', type=int, default=0.5) #0.6
    parser.add_argument('--batch_size', type=int, default=4096)

    ## optimizer
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.001)

    parser.add_argument('--kd_epochs', type=int, default=200)
    parser.add_argument('--kd_lr', type=float, default=0.001)
    parser.add_argument('--kd_weight_decay', type=float, default=0.0001)

    parser.add_argument('--cls_epochs', type=int, default=1000)
    parser.add_argument('--cls_lr', type=float, default=0.05)
    parser.add_argument('--cls_weight_decay', type=float, default=0.01)
    #####################################
    args, _ = parser.parse_known_args()
    return args


def wisconsin_params():
    parser = argparse.ArgumentParser()
    #####################################
    ## basic info
    parser.add_argument('--dataset_name', type=str, default=dataset)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--rand_split_class', type=bool, default=True,
                        help='use random splits with a fixed number of labeled nodes for each class')
    parser.add_argument('--rand_split', type=bool, default=False)
    parser.add_argument('--label_num_per_class', type=int, default=20,
                        help='labeled nodes per class(randomly selected)')
    parser.add_argument('--valid_num', type=int, default=500,
                        help='Total number of validation')
    parser.add_argument('--test_num', type=int, default=1000,
                        help='Total number of test')
    parser.add_argument('--train_ratio', type=float, default=.1,
                        help='training label proportion')
    parser.add_argument('--valid_ratio', type=float, default=.1,
                        help='validation label proportion')

    ## model
    parser.add_argument('--k_hop', type=int, default=0)
    parser.add_argument('--cluster', type=float, default=0.09) #0.1
    parser.add_argument('--hidden_channels', type=int, default=4096)
    parser.add_argument('--temperature', type=float, default=0.06)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--batch_norm', type=bool, default=False)
    parser.add_argument('--layer_norm', type=bool, default=True)
    parser.add_argument('--dropout', type=float, default=0.55)
    parser.add_argument('--batch_size', type=int, default=4096)

    ## optimizer
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=0.005)

    parser.add_argument('--kd_epochs', type=int, default=800)
    parser.add_argument('--kd_lr', type=float, default=0.0005)
    parser.add_argument('--kd_weight_decay', type=float, default=0.00001)

    parser.add_argument('--cls_epochs', type=int, default=1000)
    parser.add_argument('--cls_lr', type=float, default=0.05)
    parser.add_argument('--cls_weight_decay', type=float, default=0.0005)
    #####################################
    args, _ = parser.parse_known_args()
    return args


def actor_params():
    parser = argparse.ArgumentParser()
    #####################################
    ## basic info
    parser.add_argument('--dataset_name', type=str, default=dataset)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--rand_split_class', type=bool, default=True,
                        help='use random splits with a fixed number of labeled nodes for each class')
    parser.add_argument('--rand_split', type=bool, default=False)
    parser.add_argument('--label_num_per_class', type=int, default=20,
                        help='labeled nodes per class(randomly selected)')
    parser.add_argument('--valid_num', type=int, default=500,
                        help='Total number of validation')
    parser.add_argument('--test_num', type=int, default=1000,
                        help='Total number of test')
    parser.add_argument('--train_ratio', type=float, default=.1,
                        help='training label proportion')
    parser.add_argument('--valid_ratio', type=float, default=.1,
                        help='validation label proportion')

    ## model
    parser.add_argument('--k_hop', type=int, default=0)
    parser.add_argument('--cluster', type=float, default=0.09)#0.1
    parser.add_argument('--hidden_channels', type=int, default=2048)
    parser.add_argument('--temperature', type=float, default=0.03)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--batch_norm', type=bool, default=False)
    parser.add_argument('--layer_norm', type=bool, default=True)
    parser.add_argument('--dropout', type=float, default=0.55)
    parser.add_argument('--batch_size', type=int, default=4096)

    ## optimizer
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.01)

    parser.add_argument('--kd_epochs', type=int, default=800)
    parser.add_argument('--kd_lr', type=float, default=0.001)
    parser.add_argument('--kd_weight_decay', type=float, default=0.00001)

    parser.add_argument('--cls_epochs', type=int, default=1000)
    parser.add_argument('--cls_lr', type=float, default=0.0001)
    parser.add_argument('--cls_weight_decay', type=float, default=0.000005)
    #####################################
    args, _ = parser.parse_known_args()
    return args


def crocodile_params():
    parser = argparse.ArgumentParser()
    #####################################
    ## basic info
    parser.add_argument('--dataset_name', type=str, default=dataset)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--rand_split_class', type=bool, default=True,
                        help='use random splits with a fixed number of labeled nodes for each class')
    parser.add_argument('--rand_split', type=bool, default=False)
    parser.add_argument('--label_num_per_class', type=int, default=20,
                        help='labeled nodes per class(randomly selected)')
    parser.add_argument('--valid_num', type=int, default=500,
                        help='Total number of validation')
    parser.add_argument('--test_num', type=int, default=1000,
                        help='Total number of test')
    parser.add_argument('--train_ratio', type=float, default=.48,
                        help='training label proportion')
    parser.add_argument('--valid_ratio', type=float, default=.32,
                        help='validation label proportion')

    ## model
    parser.add_argument('--k_hop', type=int, default=0) #2
    parser.add_argument('--cluster', type=float, default=0.02)
    parser.add_argument('--hidden_channels', type=int, default=8192)
    parser.add_argument('--temperature', type=float, default=0.09)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--batch_norm', type=bool, default=False)
    parser.add_argument('--layer_norm', type=bool, default=True)
    parser.add_argument('--dropout', type=float, default=0.5) #0.4
    parser.add_argument('--batch_size', type=int, default=4096)

    ## optimizer
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--weight_decay', type=float, default=0.0001)

    parser.add_argument('--kd_epochs', type=int, default=600)
    parser.add_argument('--kd_lr', type=float, default=0.01)
    parser.add_argument('--kd_weight_decay', type=float, default=0.000001)

    parser.add_argument('--cls_epochs', type=int, default=1000)
    parser.add_argument('--cls_lr', type=float, default=0.0005)
    parser.add_argument('--cls_weight_decay', type=float, default=0.0005)
    #####################################
    args, _ = parser.parse_known_args()
    return args


def roman_empire_params():
    parser = argparse.ArgumentParser()
    #####################################
    ## basic info
    parser.add_argument('--dataset_name', type=str, default=dataset)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--rand_split_class', type=bool, default=True,
                        help='use random splits with a fixed number of labeled nodes for each class')
    parser.add_argument('--rand_split', type=bool, default=False)
    parser.add_argument('--label_num_per_class', type=int, default=20,
                        help='labeled nodes per class(randomly selected)')
    parser.add_argument('--valid_num', type=int, default=500,
                        help='Total number of validation')
    parser.add_argument('--test_num', type=int, default=1000,
                        help='Total number of test')
    parser.add_argument('--train_ratio', type=float, default=.1,
                        help='training label proportion')
    parser.add_argument('--valid_ratio', type=float, default=.1,
                        help='validation label proportion')

    ## model
    parser.add_argument('--k_hop', type=int, default=1)
    parser.add_argument('--cluster', type=float, default=0.01)
    parser.add_argument('--hidden_channels', type=int, default=8192)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--batch_norm', type=bool, default=False)
    parser.add_argument('--layer_norm', type=bool, default=True)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--batch_size', type=int, default=4096)

    ## optimizer
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.001)

    parser.add_argument('--kd_epochs', type=int, default=600)
    parser.add_argument('--kd_lr', type=float, default=0.01)
    parser.add_argument('--kd_weight_decay', type=float, default=0.000001)

    parser.add_argument('--cls_epochs', type=int, default=800)
    parser.add_argument('--cls_lr', type=float, default=0.005)
    parser.add_argument('--cls_weight_decay', type=float, default=0.001)
    #####################################
    args, _ = parser.parse_known_args()
    return args


def amazon_rating_params():
    parser = argparse.ArgumentParser()
    #####################################
    ## basic info
    parser.add_argument('--dataset_name', type=str, default=dataset)
    parser.add_argument('--runs', type=int, default=10) #10
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--rand_split_class', type=bool, default=True,
                        help='use random splits with a fixed number of labeled nodes for each class')
    parser.add_argument('--rand_split', type=bool, default=False)
    parser.add_argument('--label_num_per_class', type=int, default=20,
                        help='labeled nodes per class(randomly selected)')
    parser.add_argument('--valid_num', type=int, default=500,
                        help='Total number of validation')
    parser.add_argument('--test_num', type=int, default=1000,
                        help='Total number of test')
    parser.add_argument('--train_ratio', type=float, default=.1,
                        help='training label proportion')
    parser.add_argument('--valid_ratio', type=float, default=.1,
                        help='validation label proportion')

    ## model
    parser.add_argument('--k_hop', type=int, default=2) #6
    parser.add_argument('--cluster', type=float, default=0.06)
    parser.add_argument('--hidden_channels', type=int, default=8192)
    parser.add_argument('--temperature', type=float, default=0.09)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--batch_norm', type=bool, default=False)
    parser.add_argument('--layer_norm', type=bool, default=False)
    parser.add_argument('--dropout', type=float, default=0.55)
    parser.add_argument('--batch_size', type=int, default=4096)

    ## optimizer
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.00001)

    parser.add_argument('--kd_epochs', type=int, default=300) #800
    parser.add_argument('--kd_lr', type=float, default=0.001)
    parser.add_argument('--kd_weight_decay', type=float, default=0.)

    parser.add_argument('--cls_epochs', type=int, default=1000)
    parser.add_argument('--cls_lr', type=float, default=0.05)
    parser.add_argument('--cls_weight_decay', type=float, default=0.00001)
    #####################################
    args, _ = parser.parse_known_args()
    return args


def questions_params():
    parser = argparse.ArgumentParser()
    #####################################
    ## basic info
    parser.add_argument('--dataset_name', type=str, default=dataset)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--rand_split_class', type=bool, default=True,
                        help='use random splits with a fixed number of labeled nodes for each class')
    parser.add_argument('--rand_split', type=bool, default=False)
    parser.add_argument('--label_num_per_class', type=int, default=20,
                        help='labeled nodes per class(randomly selected)')
    parser.add_argument('--valid_num', type=int, default=500,
                        help='Total number of validation')
    parser.add_argument('--test_num', type=int, default=1000,
                        help='Total number of test')
    parser.add_argument('--train_ratio', type=float, default=.1,
                        help='training label proportion')
    parser.add_argument('--valid_ratio', type=float, default=.1,
                        help='validation label proportion')

    ## model
    parser.add_argument('--k_hop', type=int, default=5)
    parser.add_argument('--cluster', type=float, default=0.007) #0.05
    parser.add_argument('--hidden_channels', type=int, default=8192)
    parser.add_argument('--temperature', type=float, default=0.05)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--batch_norm', type=bool, default=False)
    parser.add_argument('--layer_norm', type=bool, default=False)
    parser.add_argument('--dropout', type=float, default=0.55) #0.55
    parser.add_argument('--batch_size', type=int, default=4096)

    ## optimizer
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=0.000001)

    parser.add_argument('--kd_epochs', type=int, default=600)
    parser.add_argument('--kd_lr', type=float, default=0.01)
    parser.add_argument('--kd_weight_decay', type=float, default=0.000001)

    parser.add_argument('--cls_epochs', type=int, default=1200)
    parser.add_argument('--cls_lr', type=float, default=0.01)
    parser.add_argument('--cls_weight_decay', type=float, default=0.)
    #####################################
    args, _ = parser.parse_known_args()
    return args


def arxiv_params():
    parser = argparse.ArgumentParser()
    #####################################
    ## basic info
    parser.add_argument('--dataset_name', type=str, default=dataset)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--rand_split_class', type=bool, default=True,
                        help='use random splits with a fixed number of labeled nodes for each class')
    parser.add_argument('--rand_split', type=bool, default=False)
    parser.add_argument('--label_num_per_class', type=int, default=20,
                        help='labeled nodes per class(randomly selected)')
    parser.add_argument('--valid_num', type=int, default=500,
                        help='Total number of validation')
    parser.add_argument('--test_num', type=int, default=1000,
                        help='Total number of test')
    parser.add_argument('--train_ratio', type=float, default=.1,
                        help='training label proportion')
    parser.add_argument('--valid_ratio', type=float, default=.1,
                        help='validation label proportion')

    ## model
    parser.add_argument('--k_hop', type=int, default=12)
    parser.add_argument('--cluster', type=float, default=0.007)
    parser.add_argument('--hidden_channels', type=int, default=800)#800
    parser.add_argument('--temperature', type=float, default=0.03)
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--batch_norm', type=bool, default=False)
    parser.add_argument('--layer_norm', type=bool, default=True)
    parser.add_argument('--dropout', type=float, default=0.1)#0.1
    parser.add_argument('--batch_size', type=int, default=4096)

    ## optimizer
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--weight_decay', type=float, default=0.000001)

    parser.add_argument('--kd_epochs', type=int, default=1000)
    parser.add_argument('--kd_lr', type=float, default=0.001)
    parser.add_argument('--kd_weight_decay', type=float, default=0.0000001)

    parser.add_argument('--cls_epochs', type=int, default=1000)
    parser.add_argument('--cls_lr', type=float, default=0.05)
    parser.add_argument('--cls_weight_decay', type=float, default=0.00005)
    #####################################
    args, _ = parser.parse_known_args()
    return args


def product_params():
    parser = argparse.ArgumentParser()
    #####################################
    ## basic info
    parser.add_argument('--dataset_name', type=str, default=dataset)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--rand_split_class', type=bool, default=True,
                        help='use random splits with a fixed number of labeled nodes for each class')
    parser.add_argument('--rand_split', type=bool, default=False)
    parser.add_argument('--label_num_per_class', type=int, default=20,
                        help='labeled nodes per class(randomly selected)')
    parser.add_argument('--valid_num', type=int, default=500,
                        help='Total number of validation')
    parser.add_argument('--test_num', type=int, default=1000,
                        help='Total number of test')
    parser.add_argument('--train_ratio', type=float, default=.1,
                        help='training label proportion')
    parser.add_argument('--valid_ratio', type=float, default=.1,
                        help='validation label proportion')

    ## model
    parser.add_argument('--k_hop', type=int, default=10)
    parser.add_argument('--cluster', type=float, default=0.0001)
    parser.add_argument('--hidden_channels', type=int, default=128)#128
    parser.add_argument('--temperature', type=float, default=0.06)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--batch_norm', type=bool, default=False)
    parser.add_argument('--layer_norm', type=bool, default=True)
    parser.add_argument('--dropout', type=float, default=0.05)#0.05
    parser.add_argument('--batch_size', type=int, default=40960)

    ## optimizer
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.00001)

    parser.add_argument('--kd_epochs', type=int, default=400)
    parser.add_argument('--kd_lr', type=float, default=0.0005)
    parser.add_argument('--kd_weight_decay', type=float, default=0.00001)

    parser.add_argument('--cls_epochs', type=int, default=1200)
    parser.add_argument('--cls_lr', type=float, default=0.1)
    parser.add_argument('--cls_weight_decay', type=float, default=0.000005)
    #####################################
    args, _ = parser.parse_known_args()
    return args


def set_params():
    if dataset == "cora":
        args = cora_params()
        args.big = False
        args.homogeneous = True
        args.rand_split_class = True
        args.rand_split = False
    elif dataset == "citeseer":
        args = citeseer_params()
        args.big = False
        args.homogeneous = True
        args.rand_split_class = True
        args.rand_split = False
    elif dataset == "pubmed":
        args = pubmed_params()
        args.big = False
        args.homogeneous = True
        args.rand_split_class = True
        args.rand_split = False
    elif dataset =="amazon-computer":
        args = amazon_computer_params()
        args.big = False
        args.homogeneous = True
        args.rand_split_class = False
        args.rand_split = True
    elif dataset =="amazon-photo":
        args = amazon_photo_params()
        args.big = False
        args.homogeneous = True
        args.rand_split_class = False
        args.rand_split = True
    elif dataset =="coauthor-cs":
        args = coauthor_cs_params()
        args.big = False
        args.homogeneous = True
        args.rand_split_class = False
        args.rand_split = True
    elif dataset =="coauthor-physics":
        args = coauthor_physics_params()
        args.big = False
        args.homogeneous = True
        args.rand_split_class = False
        args.rand_split = True
    elif dataset =="wikics":
        args = wikics_params()
        args.big = False
        args.homogeneous = True
        args.rand_split_class = False
        args.rand_split = True
    elif dataset == "cornell":
        args = cornell_params()
        args.big = False
        args.homogeneous = False
        args.rand_split_class = False
        args.rand_split = False
    elif dataset == "texas":
        args = texas_params()
        args.big = False
        args.homogeneous = False
        args.rand_split_class = False
        args.rand_split = False
    elif dataset == "wisconsin":
        args = wisconsin_params()
        args.big = False
        args.homogeneous = False
        args.rand_split_class = False
        args.rand_split = False
    elif dataset == "film":
        args = actor_params()
        args.big = False
        args.homogeneous = False
        args.rand_split_class = False
        args.rand_split = False
    elif dataset == "crocodile":
        args = crocodile_params()
        args.big = False
        args.homogeneous = False
        args.rand_split_class = False
        args.rand_split = False
    elif dataset =="amazon-ratings":
        args = amazon_rating_params()
        args.big = False
        args.homogeneous = False
        args.rand_split_class = False
        args.rand_split = False
    elif dataset == "roman-empire":
        args = roman_empire_params()
        args.big = False
        args.homogeneous = False
        args.rand_split_class = False
        args.rand_split = False
    elif dataset == "questions":
        args = questions_params()
        args.big = False
        args.homogeneous = False
        args.rand_split_class = False
        args.rand_split = False
    elif dataset =="ogbn-arxiv":
        args = arxiv_params()
        args.big = True
        args.homogeneous = False
        args.rand_split_class = False
        args.rand_split = False
    elif dataset =="ogbn-products":
        args = product_params()
        args.big = True
        args.homogeneous = False
        args.rand_split_class = False
        args.rand_split = False
    return args
