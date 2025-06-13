import torch, torch.nn as nn
class GInformer(nn.Module):
    def __init__(self, d_model=768, nhead=8, num_layers=1):
        super().__init__()
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, batch_first=True),
            num_layers=num_layers)
    def forward(self, x):           # x: (B,T,D) 全履歴
        return self.enc(x)[:,-1]
