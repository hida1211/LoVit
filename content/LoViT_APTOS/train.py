import torch, torch.nn as nn, torch.optim as optim
from datasets import PhaseVideoDataset, DataLoader
from extractor import RichExtractor
from model import LoViT

ALPHA=8; EPOCHS_EX=3; EPOCHS_LV=2; BS=4      # dry‐run 用

train_ds = PhaseVideoDataset(
    '/content/drive/MyDrive/kaggle/APTOS/APTOS_train-val_annotation.csv',
    '/content/drive/MyDrive/kaggle/APTOS/aptos_videos',
    '/content/drive/MyDrive/kaggle/APTOS/meta/video_fps.csv',
    split='train',
    video_subset=['case_0985','case_0791','case_1362','case_1475','case_1944','case_0690','case_0612'],
    alpha=ALPHA)

loader = DataLoader(train_ds, BS, shuffle=True, num_workers=2)

########################
# Stage‐1: extractor   #
########################
ex = RichExtractor()
opt = optim.SGD(ex.parameters(), lr=1e-3, momentum=0.9)
ce = nn.CrossEntropyLoss()

for epoch in range(EPOCHS_EX):
    for clip, pid in loader:
        opt.zero_grad()
        logits, _ = ex(clip)
        loss = ce(logits, pid)
        loss.backward(); opt.step()
print('Extractor pre‐training done.')
torch.save(ex.state_dict(), 'extractor.pt')

########################
# Stage‐2: LoViT       #
########################
lovit = LoViT(); lovit.extractor.load_state_dict(torch.load('extractor.pt'))
lovit.freeze_extractor()
opt2 = optim.SGD(filter(lambda p:p.requires_grad, lovit.parameters()),
                 lr=3e-4, momentum=0.9)

for epoch in range(EPOCHS_LV):
    for clip, pid in loader:
        opt2.zero_grad()
        pred = lovit(clip)          # 逐次入力簡略化
        loss = ce(pred, pid)
        loss.backward(); opt2.step()
torch.save(lovit.state_dict(), 'lovit_dryrun.pt')
