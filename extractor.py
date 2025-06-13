import torch, torch.nn as nn
from timm import create_model
from einops import rearrange

class RichExtractor(nn.Module):
    def __init__(self, num_classes=35, embed_dim=768):
        super().__init__()
        self.backbone = create_model('vit_base_patch16_224', pretrained=True)
        self.backbone.patch_size = (16,16)
        self.backbone.img_size = 248
        self.temporal = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=12, batch_first=True),
            num_layers=2
        )
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, clip):              # clip: (B,α,3,H,W)
        B, T, C, H, W = clip.shape
        x = rearrange(clip, 'b t c h w -> (b t) c h w')
        feat = self.backbone.forward_features(x)      # (B*T, D)
        feat = rearrange(feat, '(b t) d -> b t d', b=B, t=T)
        z = self.temporal(feat)[:,-1]                 # 最後の token
        return self.head(z), z.detach()               # logits, spatial e_t
