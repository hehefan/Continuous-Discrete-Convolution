import os.path as osp
import numpy as np
from functools import partial
from typing import Type, Any, Callable, Union, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torchvision

import torch_geometric.transforms as T
from datasets import GODataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, fps, global_max_pool, global_mean_pool, radius

from models import Model
from utils import fmax

def train(epoch, dataloader, loss_fn):
    model.train()
    for data in dataloader:
        data = data.to(device)
        optimizer.zero_grad()
        y = torch.from_numpy(np.stack(data.y, axis=0)).to(device)
        loss = loss_fn(model(data).sigmoid(), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()

def test(dataloader):
    model.eval()
    # Iterate over the validation data.

    probs = []
    labels = []
    for data in dataloader:
        data = data.to(device)
        with torch.no_grad():
            prob = model(data).sigmoid().detach().cpu().numpy()
            y = np.stack(data.y, axis=0)
        probs.append(prob)
        labels.append(y)
    probs = np.concatenate(probs, axis=0)
    labels = np.concatenate(labels, axis=0)

    return fmax(probs, labels)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='CDConv')
    #parser.add_argument('--data-dir', default='/tmp/protein-data/go', type=str, metavar='N', help='data root directory')
    parser.add_argument('--data-dir', default='../data/go', type=str, metavar='N', help='data root directory')
    parser.add_argument('--level', default='mf', type=str, metavar='N', help='mf: molecular function, bp: biological process, cc: cellular component')
    parser.add_argument('--geometric-radius', default=4.0, type=float, metavar='N', help='initial 3D ball query radius')
    parser.add_argument('--sequential-kernel-size', default=21, type=int, metavar='N', help='1D sequential kernel size')
    parser.add_argument('--kernel-channels', default=[24], type=int, metavar='N', help='kernel channels')
    parser.add_argument('--base-width', default=32, type=float, metavar='N', help='bottleneck width')
    parser.add_argument('--channels', nargs='+', default=[256, 512, 1024, 2048], type=int, metavar='N', help='feature channels')
    parser.add_argument('--num-epochs', default=500, type=int, metavar='N', help='number of training epochs')
    parser.add_argument('--batch-size', default=24, type=int, metavar='N', help='batch size')
    parser.add_argument('--lr', default=0.001, type=float, metavar='N', help='learning rate')
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float, metavar='W', help='weight decay (default: 5e-4)', dest='weight_decay')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--lr-milestones', nargs='+', default=[300, 400], type=int, help='decrease lr on milestones')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 8)')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--ckpt-path', default='', type=str, help='path where to save checkpoint')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print(args)
    print("torch version: ", torch.__version__)
    print("torchvision version: ", torchvision.__version__)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = GODataset(root=args.data_dir, random_seed=args.seed, level=args.level, split='train')
    valid_dataset = GODataset(root=args.data_dir, random_seed=args.seed, level=args.level, split='valid')
    test_dataset_30 = GODataset(root=args.data_dir, random_seed=args.seed, level=args.level, percent=30, split='test')
    test_dataset_40 = GODataset(root=args.data_dir, random_seed=args.seed, level=args.level, percent=40, split='test')
    test_dataset_50 = GODataset(root=args.data_dir, random_seed=args.seed, level=args.level, percent=50, split='test')
    test_dataset_70 = GODataset(root=args.data_dir, random_seed=args.seed, level=args.level, percent=70, split='test')
    test_dataset_95 = GODataset(root=args.data_dir, random_seed=args.seed, level=args.level, percent=95, split='test')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.workers)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader_30 = DataLoader(test_dataset_30, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader_40 = DataLoader(test_dataset_40, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader_50 = DataLoader(test_dataset_50, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader_70 = DataLoader(test_dataset_70, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader_95 = DataLoader(test_dataset_95, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = Model(geometric_radii=[2*args.geometric_radius, 3*args.geometric_radius, 4*args.geometric_radius, 5*args.geometric_radius],
                  sequential_kernel_size=args.sequential_kernel_size,
                  kernel_channels=args.kernel_channels, channels=args.channels, base_width=args.base_width,
                  num_classes=train_dataset.num_classes).to(device)
    optimizer = torch.optim.SGD(model.parameters(), weight_decay=args.weight_decay, lr=args.lr, momentum=args.momentum)
    loss_fn = torch.nn.BCELoss(weight=torch.as_tensor(train_dataset.weights).to(device))

    # learning rate scheduler
    lr_weights = []
    for i, milestone in enumerate(args.lr_milestones):
        if i == 0:
            lr_weights += [np.power(args.lr_gamma, i)] * milestone
        else:
            lr_weights += [np.power(args.lr_gamma, i)] * (milestone - args.lr_milestones[i-1])
    if args.lr_milestones[-1] < args.num_epochs:
        lr_weights += [np.power(args.lr_gamma, len(args.lr_milestones))] * (args.num_epochs + 1 - args.lr_milestones[-1])
    lambda_lr = lambda epoch: lr_weights[epoch]
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)

    best_valid = best_test_30 = best_test_40 = best_test_50 = best_test_70 = best_test_95 = best_30 = best_40 = best_50 = best_70 = best_95 = 0.0
    best_epoch = 0
    for epoch in range(args.num_epochs):
        train(epoch, train_loader, loss_fn)
        lr_scheduler.step()
        valid_fmax = test(valid_loader)
        test_30 = test(test_loader_30)
        test_40 = test(test_loader_40)
        test_50 = test(test_loader_50)
        test_70 = test(test_loader_70)
        test_95 = test(test_loader_95)
        print(f'Epoch: {epoch+1:03d}, Validation: {valid_fmax:.4f}, Test: {test_30:.4f}\t{test_40:.4f}\t{test_50:.4f}\t{test_70:.4f}\t{test_95:.4f}')
        if valid_fmax >= best_valid:
            best_valid = valid_fmax
            best_30 = test_30
            best_40 = test_40
            best_50 = test_50
            best_70 = test_70
            best_95 = test_95
            best_epoch = epoch
            checkpoint = model.state_dict()

        best_test_30 = max(test_30, best_test_30)
        best_test_40 = max(test_40, best_test_40)
        best_test_50 = max(test_50, best_test_50)
        best_test_70 = max(test_70, best_test_70)
        best_test_95 = max(test_95, best_test_95)

    print(f'Best: {best_epoch+1:03d}, Validation: {best_valid:.4f}, Test: {best_test_30:.4f}\t{best_test_40:.4f}\t{best_test_50:.4f}\t{best_test_70:.4f}\t{best_test_95:.4f}, Valided Test: {best_30:.4f}\t{best_40:.4f}\t{best_50:.4f}\t{best_70:.4f}\t{best_95:.4f}')
    if args.ckpt_path:
        torch.save(checkpoint, osp.join(args.ckpt_path))
