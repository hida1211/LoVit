import torch, torch.nn as nn
from einops import rearrange

class LTrans(nn.Module):
    def __init__(self, d_model=768, nhead=8, num_layers=2, win=64):
        super().__init__()
        self.win = win
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, batch_first=True),
            num_layers=num_layers)
    def forward(self, x):           # x: (B,T,D)
        x = x[:, -self.win:]
        return self.enc(x)[:,-1]    # (B,D)
