import os
import math

import numpy as np

import torch
from torch.utils.data import Dataset

from torch_geometric.data import Data

from utils import orientation

# AA Letter to id
aa = "ACDEFGHIKLMNPQRSTVWYX"
aa_to_id = {}
for i in range(0, 21):
    aa_to_id[aa[i]] = i

class FoldDataset(Dataset):

    def __init__(self, root='/tmp/protein-data/fold', random_seed=0, split='training'):

        self.random_state = np.random.RandomState(random_seed)
        self.split = split

        npy_dir = os.path.join(root, 'coordinates', split)
        fasta_file = os.path.join(root, split+'.fasta')

        # Load the fasta file.
        protein_seqs = []
        with open(fasta_file, 'r') as f:
            protein_name = ''
            for line in f:
                if line.startswith('>'):
                    protein_name = line.rstrip()[1:]
                else:
                    amino_chain = line.rstrip()
                    amino_ids = []
                    for amino in amino_chain:
                        amino_ids.append(aa_to_id[amino])
                    protein_seqs.append((protein_name, np.array(amino_ids)))

        fold_classes = {}
        with open(os.path.join(root, 'class_map.txt'), 'r') as f:
            for line in f:
                arr = line.rstrip().split('\t')
                fold_classes[arr[0]] = int(arr[1])

        protein_folds = {}
        with open(os.path.join(root, split+'.txt'), 'r') as f:
            for line in f:
                arr = line.rstrip().split('\t')
                protein_folds[arr[0]] = fold_classes[arr[-1]]

        self.data = []
        self.labels = []
        for protein_name, amino_ids in protein_seqs:
            pos = np.load(os.path.join(npy_dir, protein_name+".npy"))

            center = np.sum(a=pos, axis=0, keepdims=True)/pos.shape[0]
            pos = pos - center
            ori = orientation(pos)

            self.data.append((pos, ori, amino_ids.astype(int)))

            self.labels.append(protein_folds[protein_name])

        self.num_classes = max(self.labels) + 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        pos, ori, amino = self.data[idx]
        label = self.labels[idx]

        if self.split == "training":
            pos = pos + self.random_state.normal(0.0, 0.05, pos.shape)

        pos = pos.astype(dtype=np.float32)
        ori = ori.astype(dtype=np.float32)
        seq = np.expand_dims(a=np.arange(pos.shape[0]), axis=1).astype(dtype=np.float32)

        data = Data(x = torch.from_numpy(amino),    # [num_nodes, num_node_features]
                    edge_index = None,              # [2, num_edges]
                    edge_attr = None,               # [num_edges, num_edge_features]
                    y = label,
                    ori = torch.from_numpy(ori),    # [num_nodes, 3, 3]
                    seq = torch.from_numpy(seq),    # [num_nodes, 1]
                    pos = torch.from_numpy(pos))    # [num_nodes, num_dimensions]

        return data

class FuncDataset(Dataset):

    def __init__(self, root='/tmp/protein-data/func', random_seed=0, split='training'):

        self.random_state = np.random.RandomState(random_seed)
        self.split = split

        # Get the paths.
        npy_dir = os.path.join(os.path.join(root, 'coordinates'), split)
        fasta_file = os.path.join(root, 'chain_'+split+'.fasta')

        # Load the fasta file.
        protein_seqs = []
        with open(fasta_file, 'r') as f:
            protein_name = ''
            for line in f:
                if line.startswith('>'):
                    protein_name = line.rstrip()[1:]
                else:
                    amino_chain = line.rstrip()
                    amino_ids = []
                    for amino in amino_chain:
                        amino_ids.append(aa_to_id[amino])
                    protein_seqs.append((protein_name, np.array(amino_ids)))

        protein_functions = {}
        with open(os.path.join(root, 'chain_functions.txt'), 'r') as f:
            for line in f:
                arr = line.rstrip().split(',')
                protein_functions[arr[0]] = int(arr[1])

        self.data = []
        self.labels = []
        for protein_name, amino_ids in protein_seqs:
            pos = np.load(os.path.join(npy_dir, protein_name+".npy"))

            center = np.sum(a=pos, axis=0, keepdims=True)/pos.shape[0]
            pos = pos - center

            ori = orientation(pos)
            self.data.append((pos, ori, amino_ids.astype(int)))
            self.labels.append(protein_functions[protein_name])

        self.num_classes = max(self.labels) + 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        pos, ori, amino = self.data[idx]
        label = self.labels[idx]

        if self.split == "training":
            pos = pos + self.random_state.normal(0.0, 0.05, pos.shape)

        pos = pos.astype(dtype=np.float32)
        ori = ori.astype(dtype=np.float32)
        seq = np.expand_dims(a=np.arange(pos.shape[0]), axis=1).astype(dtype=np.float32)

        data = Data(x = torch.from_numpy(amino),    # [num_nodes, num_node_features]
                    edge_index = None,              # [2, num_edges]
                    edge_attr = None,               # [num_edges, num_edge_features]
                    y = label,
                    ori = torch.from_numpy(ori),    # [num_nodes, 3, 3]
                    seq = torch.from_numpy(seq),    # [num_nodes, 1]
                    pos = torch.from_numpy(pos))    # [num_nodes, num_dimensions]

        return data

class GODataset(Dataset):

    def __init__(self, root='/tmp/protein-data/go', level='mf', percent=30, random_seed=0, split='train'):

        self.random_state = np.random.RandomState(random_seed)
        self.split = split

        # Get the paths.
        npy_dir = os.path.join(root, 'coordinates')
        fasta_file = os.path.join(root, split+'.fasta')

        # Mask test set.
        test_set = set()
        if split == "test":
            with open(os.path.join(root, "nrPDB-GO_2019.06.18_test.csv"), 'r') as f:
                head = True
                for line in f:
                    if head:
                        head = False
                        continue
                    arr = line.rstrip().split(',')
                    if percent == 30 and arr[1] == '1':
                        test_set.add(arr[0])
                    elif percent == 40 and arr[2] == '1':
                        test_set.add(arr[0])
                    elif percent == 50 and arr[3] == '1':
                        test_set.add(arr[0])
                    elif percent == 70 and arr[4] == '1':
                        test_set.add(arr[0])
                    elif percent == 95 and arr[5] == '1':
                        test_set.add(arr[0])
                    else:
                        pass

        # Load the fasta file.
        protein_seqs = []
        with open(fasta_file, 'r') as f:
            protein_name = ''
            for line in f:
                if line.startswith('>'):
                    protein_name = line.rstrip()[1:]
                else:
                    if split == "test" and (protein_name not in test_set):
                        continue
                    amino_chain = line.rstrip()
                    amino_ids = []
                    for amino in amino_chain:
                        amino_ids.append(aa_to_id[amino])
                    protein_seqs.append((protein_name, np.array(amino_ids)))


        self.data = []
        for protein_name, amino_ids in protein_seqs:
            pos = np.load(os.path.join(npy_dir, protein_name+".npy"))

            center = np.sum(a=pos, axis=0, keepdims=True)/pos.shape[0]
            pos = pos - center

            ori = orientation(pos)
            self.data.append((protein_name, pos, ori, amino_ids.astype(int)))

        level_idx = 0
        go_cnt = 0
        go_num = {}
        go_annotations = {}
        self.labels = {}
        with open(os.path.join(root, 'nrPDB-GO_2019.06.18_annot.tsv'), 'r') as f:
            for idx, line in enumerate(f):
                if idx == 1 and level == "mf":
                    level_idx = 1
                    arr = line.rstrip().split('\t')
                    for go in arr:
                        go_annotations[go] = go_cnt
                        go_num[go] = 0
                        go_cnt += 1
                elif idx == 5 and level == "bp":
                    level_idx = 2
                    arr = line.rstrip().split('\t')
                    for go in arr:
                        go_annotations[go] = go_cnt
                        go_num[go] = 0
                        go_cnt += 1
                elif idx == 9 and level == "cc":
                    level_idx = 3
                    arr = line.rstrip().split('\t')
                    for go in arr:
                        go_annotations[go] = go_cnt
                        go_num[go] = 0
                        go_cnt += 1
                elif idx > 12:
                    arr = line.rstrip().split('\t')
                    protein_labels = []
                    if len(arr) > level_idx:
                        protein_go_list = arr[level_idx]
                        protein_go_list = protein_go_list.split(',')
                        for go in protein_go_list:
                            if len(go) > 0:
                                protein_labels.append(go_annotations[go])
                                go_num[go] += 1
                    self.labels[arr[0]] = np.array(protein_labels)

        self.num_classes = len(go_annotations)

        self.weights = np.zeros((go_cnt,), dtype=np.float32)
        for go, idx in go_annotations.items():
            self.weights[idx] = len(self.labels)/go_num[go]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        protein_name, pos, ori, amino = self.data[idx]
        label = np.zeros((self.num_classes,)).astype(np.float32)
        if len(self.labels[protein_name]) > 0:
            label[self.labels[protein_name]] = 1.0

        if self.split == "train":
            pos = pos + self.random_state.normal(0.0, 0.05, pos.shape)

        pos = pos.astype(dtype=np.float32)
        ori = ori.astype(dtype=np.float32)
        seq = np.expand_dims(a=np.arange(pos.shape[0]), axis=1).astype(dtype=np.float32)

        data = Data(x = torch.from_numpy(amino),    # [num_nodes, num_node_features]
                    edge_index = None,              # [2, num_edges]
                    edge_attr = None,               # [num_edges, num_edge_features]
                    y = label,
                    ori = torch.from_numpy(ori),    # [num_nodes, 3, 3]
                    seq = torch.from_numpy(seq),    # [num_nodes, 1]
                    pos = torch.from_numpy(pos))    # [num_nodes, num_dimensions]

        return data

class ECDataset(Dataset):

    def __init__(self, root='/tmp/protein-data/ec', percent=30, random_seed=0, split='train'):

        self.random_state = np.random.RandomState(random_seed)
        self.split = split

        # Get the paths.
        npy_dir = os.path.join(root, 'coordinates')
        fasta_file = os.path.join(root, split+'.fasta')

        # Mask test set.
        test_set = set()
        if split == "test":
            with open(os.path.join(root, "nrPDB-EC_test.csv"), 'r') as f:
                head = True
                for line in f:
                    if head:
                        head = False
                        continue
                    arr = line.rstrip().split(',')
                    if percent == 30 and arr[1] == '1':
                        test_set.add(arr[0])
                    elif percent == 40 and arr[2] == '1':
                        test_set.add(arr[0])
                    elif percent == 50 and arr[3] == '1':
                        test_set.add(arr[0])
                    elif percent == 70 and arr[4] == '1':
                        test_set.add(arr[0])
                    elif percent == 95 and arr[5] == '1':
                        test_set.add(arr[0])
                    else:
                        pass

        # Load the fasta file.
        protein_seqs = []
        with open(fasta_file, 'r') as f:
            protein_name = ''
            for line in f:
                if line.startswith('>'):
                    protein_name = line.rstrip()[1:]
                else:
                    if split == "test" and (protein_name not in test_set):
                        continue
                    amino_chain = line.rstrip()
                    amino_ids = []
                    for amino in amino_chain:
                        amino_ids.append(aa_to_id[amino])
                    protein_seqs.append((protein_name, np.array(amino_ids)))


        self.data = []
        for protein_name, amino_ids in protein_seqs:
            pos = np.load(os.path.join(npy_dir, protein_name+".npy"))
            center = np.sum(a=pos, axis=0, keepdims=True)/pos.shape[0]
            pos = pos - center
            ori = orientation(pos)
            self.data.append((protein_name, pos, ori, amino_ids.astype(int)))

        level_idx = 1
        ec_cnt = 0
        ec_num = {}
        ec_annotations = {}
        self.labels = {}

        with open(os.path.join(root, 'nrPDB-EC_annot.tsv'), 'r') as f:
            for idx, line in enumerate(f):
                if idx == 1:
                    arr = line.rstrip().split('\t')
                    for ec in arr:
                        ec_annotations[ec] = ec_cnt
                        ec_num[ec] = 0
                        ec_cnt += 1

                elif idx > 2:
                    arr = line.rstrip().split('\t')
                    protein_labels = []
                    if len(arr) > level_idx:
                        protein_ec_list = arr[level_idx]
                        protein_ec_list = protein_ec_list.split(',')
                        for ec in protein_ec_list:
                            if len(ec) > 0:
                                protein_labels.append(ec_annotations[ec])
                                ec_num[ec] += 1
                    self.labels[arr[0]] = np.array(protein_labels)

        self.num_classes = len(ec_annotations)
        self.weights = np.zeros((ec_cnt,), dtype=np.float32)
        for ec, idx in ec_annotations.items():
            self.weights[idx] = len(self.labels)/ec_num[ec]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        protein_name, pos, ori, amino = self.data[idx]
        label = np.zeros((self.num_classes,)).astype(np.float32)
        if len(self.labels[protein_name]) > 0:
            label[self.labels[protein_name]] = 1.0

        if self.split == "train":
            pos = pos + self.random_state.normal(0.0, 0.05, pos.shape)

        pos = pos.astype(dtype=np.float32)
        ori = ori.astype(dtype=np.float32)
        seq = np.expand_dims(a=np.arange(pos.shape[0]), axis=1).astype(dtype=np.float32)

        data = Data(x = torch.from_numpy(amino),    # [num_nodes, num_node_features]
                    edge_index = None,              # [2, num_edges]
                    edge_attr = None,               # [num_edges, num_edge_features]
                    y = label,
                    ori = torch.from_numpy(ori),    # [num_nodes, 3, 3]
                    seq = torch.from_numpy(seq),    # [num_nodes, 1]
                    pos = torch.from_numpy(pos))    # [num_nodes, num_dimensions]

        return data
