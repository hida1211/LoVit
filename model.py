import torch, torch.nn as nn
from extractor import RichExtractor
from ltrans import CascadedLTrans
from ginformer import GInformer
from head import FusionHead

class LoViT(nn.Module):
    def __init__(self, n_classes=35, d_model=768):
        super().__init__()
        self.extractor = RichExtractor(n_classes)     # 先に別フェーズで学習
        self.extractor_head = nn.Identity()          # 推評時 logits 無視
        self.ls = CascadedLTrans(d_model=d_model, nhead=8)
        self.g  = GInformer(layers=2, d_model=d_model)
        self.head = FusionHead(d_model, n_classes)

    @torch.no_grad()
    def freeze_extractor(self):
        for p in self.extractor.parameters(): p.requires_grad=False

    def forward(self, clip_seq):     # clip_seq: (B,T,3,H,W) contiguous stream
        # 1 フレームだけ取り出し e_t を粘止更新する想定
        logits, e_t = self.extractor(clip_seq)        # (B,α)→e_t
        if not hasattr(self, 'buffer'): self.buffer=[]
        self.buffer.append(e_t)                       # list of (B,D)
        seq = torch.stack(self.buffer,1)              # (B,t,D)
        s, l = self.ls(seq)
        g = self.g(seq)
        p_t, h_t = self.head(s, l, g)
        return p_t, h_t
