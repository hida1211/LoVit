import torch, torch.nn as nn
import math


def prob_sparse_attention(q, k, v, u):
    # q:(B,Lq,D), k,v:(B,Lk,D)
    B, Lq, D = q.shape
    Lk = k.size(1)
    scores = torch.zeros(B, Lq, Lk, device=q.device)
    idx = torch.randint(0, Lk, (Lq, u), device=q.device)
    for i in range(Lq):
        qi = q[:, i:i+1, :]
        ki = k.gather(1, idx[i].expand(B, u).unsqueeze(-1).repeat(1, 1, D))
        tmp = (qi @ ki.transpose(-2, -1) / math.sqrt(D)).squeeze(-2)
        mm = tmp.max(-1).values - tmp.mean(-1)
        scores[:, i, idx[i]] = mm.unsqueeze(-1)
    attn = torch.softmax(scores, dim=-1)
    return attn @ v


class InformerBlock(nn.Module):
    def __init__(self, d_model=768, nhead=8):
        super().__init__()
        self.proj_q = nn.Linear(d_model, d_model)
        self.proj_k = nn.Linear(d_model, d_model)
        self.proj_v = nn.Linear(d_model, d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        q = self.proj_q(x)
        k = self.proj_k(x)
        v = self.proj_v(x)
        u = int(k.size(1) * math.log(q.size(1)))  # U = Lk ln Lq
        z = prob_sparse_attention(q, k, v, u)
        x = self.norm1(x + z)
        x = self.norm2(x + self.ff(x))
        return x


class GInformer(nn.Module):
    def __init__(self, layers=2, d_model=768):
        super().__init__()
        self.blocks = nn.ModuleList([InformerBlock(d_model) for _ in range(layers)])

    def forward(self, seq):               # seq (B,T,D) 全履歴
        x = seq
        for blk in self.blocks:
            x = blk(x)
        return x[:, -1]                    # g_t
