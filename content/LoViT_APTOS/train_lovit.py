import torch
import torch.nn as nn
import torch.optim as optim
from datasets import PhaseVideoDataset, DataLoader
from model import LoViT

ALPHA = 8
BS = 4

train_ds = PhaseVideoDataset(
    '/content/drive/MyDrive/kaggle/APTOS/APTOS_train-val_annotation.csv',
    '/content/drive/MyDrive/kaggle/APTOS/aptos_videos',
    '/content/drive/MyDrive/kaggle/APTOS/meta/video_fps.csv',
    split='train',
    video_subset=['case_0985','case_0791','case_1362','case_1475','case_1944','case_0690','case_0612'],
    alpha=ALPHA)

loader = DataLoader(train_ds, BS, shuffle=True, num_workers=2)

lovit = LoViT()
lovit.freeze_extractor()
opt = optim.SGD(filter(lambda p: p.requires_grad, lovit.parameters()), lr=3e-4, momentum=0.9)
L1 = nn.L1Loss(); CE = nn.CrossEntropyLoss()

for clip, pid, ht_gt in loader:
    opt.zero_grad()
    pred_p, pred_h = lovit(clip)
    loss = CE(pred_p, pid) + L1(pred_h, ht_gt)
    loss.backward()
    opt.step()
