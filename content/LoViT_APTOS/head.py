import torch, torch.nn as nn
class FusionHead(nn.Module):
    def __init__(self, d_model=768, n_classes=35):
        super().__init__()
        self.fc = nn.Linear(d_model*3, n_classes)
    def forward(self, s,l,g):
        z = torch.cat([s,l,g], dim=-1)
        return self.fc(z)
