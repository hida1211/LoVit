import os, pandas as pd, torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from model import LoViT

CSV_VAL = '/content/drive/MyDrive/kaggle/APTOS/APTOS_val2.csv'
FRAME_ROOT = '/content/drive/MyDrive/kaggle/APTOS/val2_videos/aptos_val2/frames'
OUT_CSV = '/content/drive/MyDrive/kaggle/APTOS/APTOS_val2_with_pred.csv'

tf = Compose([Resize((250,250)),
              ToTensor(),
              Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

lovit = LoViT(); lovit.load_state_dict(torch.load('lovit_dryrun.pt'))
lovit.eval(); lovit.cuda()

rows=[]
for vname, g in pd.read_csv(CSV_VAL).groupby('Video_name'):
    lovit.buffer=[]                         # reset sequence
    for _, r in g.sort_values('Frame_id').iterrows():
        img = Image.open(os.path.join(FRAME_ROOT, r.Frame_id))
        clip = tf(img).unsqueeze(0).unsqueeze(0).cuda()   # shape (1,1,3,H,W)
        pred = lovit(clip)[0].argmax(-1).item()
        rows.append(pred)
out = pd.read_csv(CSV_VAL).copy()
out['Predict_phase_id'] = rows
out.to_csv(OUT_CSV, index=False)
print('saved to', OUT_CSV)
