import torch
import torch.nn as nn


class FusionHead(nn.Module):
    def __init__(self, d_model=768, n_cls=35):
        super().__init__()
        self.fuse1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model * 2, 8, batch_first=True), 2)
        self.fuse2 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model * 3, 8, batch_first=True), 1)
        self.phase_cls = nn.Linear(d_model * 3, n_cls)
        self.ht_map = nn.Linear(d_model * 3, 1)      # scalar heat値

    def forward(self, s, l, g):
        flocal = self.fuse1((torch.cat([s, l], dim=-1)).unsqueeze(1)).squeeze(1)
        fglob = self.fuse2((torch.cat([flocal, g], dim=-1)).unsqueeze(1)).squeeze(1)
        return self.phase_cls(fglob), self.ht_map(fglob).squeeze(-1)
