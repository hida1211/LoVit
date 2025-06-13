import os, math, json, random
from pathlib import Path
from typing import List, Dict
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from decord import VideoReader, cpu
import numpy as np

SIGMA_L = 3   # 左側 3σ_l
SIGMA_R = 12  # 右側 3σ_r

def build_phase_heat(gt_intervals, total_frames):
    """
    gt_intervals: [(start_f, end_f, phase_id), ...] ※フレーム index
    total_frames: int
    return h: (total_frames,)  float32
    """
    h = np.zeros(total_frames, np.float32)
    for s, e, _ in gt_intervals[:-1]:
        bp = e                       # フェーズ切替点を終端フレームに置く
        for t in range(max(0, bp-3*SIGMA_L), bp):
            h[t] = np.exp(-((t-bp)**2)/(2*(SIGMA_L**2)))
        for t in range(bp, min(total_frames, bp+3*SIGMA_R)):
            h[t] = np.exp(-((t-bp)**2)/(2*(SIGMA_R**2)))
    return h

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
IMG_SIZE = 250

_basic_tf = Compose([
    Resize((IMG_SIZE, IMG_SIZE)),
    ToTensor(), Normalize(MEAN, STD)
])

def _sample_indices(fps: float, start_s: float, end_s: float, every=1.0):
    """return frame idx (int) sampled at 'every' sec."""
    frames = []
    t = start_s
    while t < end_s:
        frames.append(int(round(t * fps)))
        t += every
    return frames

class PhaseVideoDataset(Dataset):
    """
    LoViT pre‐training用：各フレーム位置 t で
    α=ALPHA 枚のサブクリップを等間サンプリングし
    (x_{t-w*(α-1)},...,x_t) を返す。
    """
    def __init__(self, csv_path, video_root, fps_csv,
                 split='train', video_subset=None,
                 alpha=8,  # dry‐run を軽くする
                 every_sec=1.0):
        self.anno = pd.read_csv(csv_path)
        self.anno = self.anno[self.anno.split == split]
        if video_subset:
            self.anno = self.anno[self.anno.video_id.isin(video_subset)]
        self.fps_tbl = pd.read_csv(fps_csv).set_index('file')['fps'].to_dict()
        self.video_root = Path(video_root)
        self.alpha = alpha
        self.every_sec = every_sec
        self.samples = []  # (video, sec, phase_id)
        self.heat_map = {}
        self.intervals = {}
        for v, rows in self.anno.groupby('video_id'):
            fps = self.fps_tbl[f'{v}.mp4']
            ivs = []
            for _, r in rows.iterrows():
                idxs = _sample_indices(fps, r.start, r.end, every_sec)
                self.samples += [(v, s/fps, r.phase_id) for s in idxs]
                ivs.append((int(round(r.start*fps)), int(round(r.end*fps)), r.phase_id))
            vr = VideoReader(str(self.video_root/f'{v}.mp4'), ctx=cpu(0))
            total_frames = len(vr)
            self.intervals[v] = ivs
            self.heat_map[v] = build_phase_heat(ivs, total_frames)

    def __len__(self):  return len(self.samples)

    def __getitem__(self, idx):
        vid, sec, pid = self.samples[idx]
        vr = VideoReader(str(self.video_root/f'{vid}.mp4'), ctx=cpu(0))
        fps = vr.get_avg_fps()
        frame_idx = int(round(sec*fps))
        # サンプリング幅 w_t
        w = max(1, frame_idx // self.alpha)
        idxs = [max(0, frame_idx - w*(self.alpha-1-i)) for i in range(self.alpha)]
        clip = torch.stack([_basic_tf(vr[i].asnumpy()) for i in idxs])  # (α,3,H,W)
        ht = torch.tensor(self.heat_map[vid][frame_idx], dtype=torch.float32)
        return clip, pid, ht
