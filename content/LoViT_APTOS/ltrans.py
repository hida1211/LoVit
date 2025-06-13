import torch, torch.nn as nn
from einops import rearrange

class FusionModule(nn.Module):
    """Encoder (m×SelfAtt) + Decoder (n×{Self+Cross})"""
    def __init__(self, d_model, nhead, m=2, n=2):
        super().__init__()
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, batch_first=True), m)
        self.dec_layers = nn.ModuleList(
            [nn.TransformerDecoderLayer(d_model, nhead, batch_first=True)
             for _ in range(n)])

    def forward(self, aux_feat, main_feat):
        # aux_feat, main_feat: (B,T,D)
        memory = self.enc(aux_feat)
        out = main_feat
        for dec in self.dec_layers:
            out = dec(out, memory)
        return out

class CascadedLTrans(nn.Module):
    """
    Two-scale L_s(L=λ1) & L_l(L=λ2) as in Fig 4.
    巻き戻し勉強節縛の detach() ロジックも含む。
    """
    def __init__(self, d_model=768, nhead=8, lambda1=100, lambda2=500):
        super().__init__()
        self.lambda1, self.lambda2 = lambda1, lambda2
        self.fusion_s = FusionModule(d_model, nhead)
        self.fusion_l = FusionModule(d_model, nhead)

    def forward(self, seq):               # (B,T,D)
        # --- Small window branch ---
        s_in = seq[:, -self.lambda1:].detach()
        s_out = self.fusion_s(s_in, s_in)
        # --- Large window branch (takes small-scale output as aux) ---
        l_in = seq[:, -self.lambda2:].detach()
        l_out = self.fusion_l(s_out.detach(), l_in)
        return s_out[:,-1], l_out[:,-1]   # 最後の token を代表に
