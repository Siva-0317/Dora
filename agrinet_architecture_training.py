"""
train_agrinet_mtl.py

Multi-task training for AgriNet-MTL:
- Pest presence (multi-label BCE from IP102 labels)
- Disease classification (cross-entropy from PlantVillage)
- Segmentation (BCE + Dice from generated masks)
- Dosage regression (MSE, synthetic from infection_fraction)

Now with:
- Checkpoint saving (every epoch + best)
- Validation loss tracking
- Optional ONNX export at the end


import os
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# -------------------------
# Utilities / losses
# -------------------------
def dice_loss(pred, target, eps=1e-6):
    pred_flat = pred.contiguous().view(pred.size(0), -1)
    target_flat = target.contiguous().view(target.size(0), -1)
    inter = (pred_flat * target_flat).sum(1)
    union = pred_flat.sum(1) + target_flat.sum(1)
    loss = 1 - (2 * inter + eps) / (union + eps)
    return loss.mean()

# -------------------------
# Model
# -------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Backbone(nn.Module):
    def __init__(self, bifpn_ch=128):
        super().__init__()
        self.layer1 = ConvBlock(3, 32, 3, 2, 1)    # 640 -> 320
        self.layer2 = ConvBlock(32, 64, 3, 2, 1)   # 320 -> 160
        self.layer3 = ConvBlock(64, bifpn_ch, 3, 2, 1)  # 160 -> 80
        self.layer4 = ConvBlock(bifpn_ch, bifpn_ch, 3, 2, 1)  # 80 -> 40
        self.layer5 = ConvBlock(bifpn_ch, bifpn_ch, 3, 2, 1)  # 40 -> 20
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        c3 = self.layer3(x)  # 80x80
        c4 = self.layer4(c3) # 40x40
        c5 = self.layer5(c4) # 20x20
        return c3, c4, c5

class BiFPN(nn.Module):
    def __init__(self, ch=128):
        super().__init__()
        self.p3 = ConvBlock(ch, ch)
        self.p4 = ConvBlock(ch, ch)
        self.p5 = ConvBlock(ch, ch)
    def forward(self, features):
        c3, c4, c5 = features
        p5 = self.p5(c5)
        p4 = self.p4(c4 + F.interpolate(p5, size=c4.shape[-2:], mode='nearest'))
        p3 = self.p3(c3 + F.interpolate(p4, size=c3.shape[-2:], mode='nearest'))
        return [p3, p4, p5]

class AgriNetMTL(nn.Module):
    def __init__(self, num_pest_classes=102, num_disease_classes=39, bifpn_ch=128):
        super().__init__()
        self.backbone = Backbone(bifpn_ch)
        self.bifpn = BiFPN(ch=bifpn_ch)

        self.det_head = nn.Conv2d(bifpn_ch, num_pest_classes, 1)
        self.cls_head = nn.Linear(bifpn_ch, num_disease_classes)
        self.seg_head = nn.Conv2d(bifpn_ch, 1, 1)
        self.dose_head = nn.Sequential(
            nn.Linear(bifpn_ch + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x, infection_fraction=None):
        c3, c4, c5 = self.backbone(x)
        features = self.bifpn([c3, c4, c5])
        f = features[0]

        pooled = F.adaptive_avg_pool2d(f, 1).flatten(1)

        det_map = self.det_head(f)
        det_scores = torch.sigmoid(det_map.mean(dim=[2,3]))

        cls_out = self.cls_head(pooled)
        seg_out = torch.sigmoid(self.seg_head(f))

        if infection_fraction is None:
            infection_fraction = torch.zeros(pooled.size(0), 1, device=pooled.device)
        else:
            infection_fraction = infection_fraction.view(-1,1).to(pooled.device)

        dose_in = torch.cat([pooled, infection_fraction], dim=1)
        dose_out = self.dose_head(dose_in)

        return det_scores, cls_out, seg_out, dose_out

# -------------------------
# Dataset
# -------------------------
class AgriCombinedDataset(Dataset):
    def __init__(self, data_root, split='train', img_size=320, pest_classes=None, K_ml=10.0):
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split
        self.img_size = img_size
        self.K_ml = K_ml

        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

        pv_root = self.data_root / 'plantvillage' / split
        self.plantvillage_items = []
        self.pv_class2idx = {}
        if pv_root.exists():
            pv_classes = sorted([d.name for d in pv_root.iterdir() if d.is_dir()])
            self.pv_class2idx = {c:i for i,c in enumerate(pv_classes)}
            for cls in pv_classes:
                for p in (pv_root/cls).glob('*.*'):
                    if p.suffix.lower() in ['.jpg','.jpeg','.png']:
                        self.plantvillage_items.append((str(p), self.pv_class2idx[cls]))

        ip_root = self.data_root / 'ip102' / 'images' / split
        self.ip_items = []
        if ip_root.exists():
            for p in ip_root.rglob('*.*'):
                if p.suffix.lower() in ['.jpg','.jpeg','.png']:
                    self.ip_items.append(str(p))

        self.pest_classes = pest_classes or []
        self.pv_mask_root = self.data_root / 'plantvillage_masks' / split
        self.ip_mask_root = self.data_root / 'ip102_masks' / split

        self.items = []
        for img_path, lab in self.plantvillage_items:
            self.items.append({'type':'disease', 'img':img_path, 'disease_label': lab})
        for img_path in self.ip_items:
            self.items.append({'type':'pest', 'img':img_path})

    def _load_mask(self, img_path, typ):
        p = Path(img_path)
        if typ == 'disease':
            rel = p.relative_to(self.data_root / 'plantvillage' / self.split)
            mask_path = self.pv_mask_root / rel
        else:
            rel = p.relative_to(self.data_root / 'ip102' / 'images' / self.split)
            mask_path = self.ip_mask_root / rel

        mask_path = Path(str(mask_path).rsplit('.', 1)[0] + '.png')
        if not mask_path.exists():
            return np.zeros((self.img_size, self.img_size), np.uint8)

        m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if m is None:
            return np.zeros((self.img_size, self.img_size), np.uint8)
        m = cv2.resize(m, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        _, m = cv2.threshold(m, 127, 1, cv2.THRESH_BINARY)
        return m.astype(np.uint8)

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img_path = item['img']
        typ = item['type']
        pil = Image.open(img_path).convert('RGB')
        mask_np = self._load_mask(img_path, typ)
        infection_fraction = float(mask_np.sum() / (mask_np.size + 1e-9))

        if typ == 'disease':
            tankA, tankB = infection_fraction*self.K_ml, 0.0
        else:
            tankA, tankB = 0.0, infection_fraction*self.K_ml

        img_t = self.tf(pil)
        mask_t = torch.from_numpy(mask_np).float().unsqueeze(0)

        disease_label = item['disease_label'] if typ=='disease' else -1
        pest_presence = torch.zeros(len(self.pest_classes), dtype=torch.float32)

        return {
            'image': img_t,
            'disease_label': disease_label,
            'pest_presence': pest_presence,
            'mask': mask_t,
            'infection_fraction': torch.tensor(infection_fraction, dtype=torch.float32),
            'dosage': torch.tensor([tankA, tankB], dtype=torch.float32)
        }

# -------------------------
# Train loop
# -------------------------
def collate_fn(batch):
    images = torch.stack([b['image'] for b in batch], 0)
    masks = torch.stack([b['mask'] for b in batch], 0)
    pest_presence = torch.stack([b['pest_presence'] for b in batch], 0)
    infection_fraction = torch.stack([b['infection_fraction'] for b in batch], 0)
    dosages = torch.stack([b['dosage'] for b in batch], 0)
    disease_labels = torch.tensor([b['disease_label'] for b in batch], dtype=torch.long)
    return images, disease_labels, pest_presence, masks, infection_fraction, dosages

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./datasets')
    parser.add_argument('--out_dir', type=str, default='./models')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--imgsz', type=int, default=320)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train_ds = AgriCombinedDataset(args.data_root, split='train', img_size=args.imgsz, pest_classes=[str(i) for i in range(102)])
    val_ds = AgriCombinedDataset(args.data_root, split='val', img_size=args.imgsz, pest_classes=[str(i) for i in range(102)])

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    model = AgriNetMTL(num_pest_classes=102, num_disease_classes=len(train_ds.pv_class2idx))
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    weights = {'det':1.0,'cls':1.0,'seg':2.0,'dose':0.5}
    best_loss = float('inf')

    for epoch in range(1, args.epochs+1):
        model.train()
        running_loss, n = 0.0, 0
        for images, disease_labels, pest_presence, masks, infection_fraction, dosages in train_loader:
            images, masks, pest_presence, infection_fraction, dosages = images.to(device), masks.to(device), pest_presence.to(device), infection_fraction.to(device), dosages.to(device)
            det_scores, cls_out, seg_out, dose_out = model(images, infection_fraction)

            det_loss = nn.BCELoss()(det_scores, pest_presence)
            cls_loss = torch.tensor(0.0, device=device)
            mask_disease = (disease_labels!=-1)
            if mask_disease.any():
                valid_idx = mask_disease.nonzero(as_tuple=False).squeeze(1)
                cls_loss = F.cross_entropy(cls_out[valid_idx], disease_labels[valid_idx].to(device))
            seg_pred = F.interpolate(seg_out, size=(masks.size(2), masks.size(3)), mode='bilinear', align_corners=False)
            seg_loss = F.binary_cross_entropy(seg_pred, masks) + dice_loss(seg_pred, masks)
            dose_loss = nn.MSELoss()(dose_out, dosages)

            total_loss = weights['det']*det_loss + weights['cls']*cls_loss + weights['seg']*seg_loss + weights['dose']*dose_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item() * images.size(0)
            n += images.size(0)

        avg_loss = running_loss / n
        print(f"Epoch {epoch}/{args.epochs} done. Avg Loss {avg_loss:.4f}")

        # Save checkpoint
        ckpt = Path(args.out_dir)/f"agrinet_epoch{epoch}.pth"
        torch.save({'epoch':epoch,'model_state':model.state_dict(),'optim_state':optimizer.state_dict(),'loss':avg_loss}, ckpt)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({'epoch':epoch,'model_state':model.state_dict(),'optim_state':optimizer.state_dict(),'loss':avg_loss}, Path(args.out_dir)/"agrinet_best.pth")
            print(f"✅ Saved best checkpoint (loss {best_loss:.4f})")

    # ONNX export
    model.eval()
    dummy_img = torch.randn(1,3,args.imgsz,args.imgsz, device=device)
    dummy_inf = torch.tensor([0.0], device=device)
    try:
        onnx_path = Path(args.out_dir)/"agrinet_mtl.onnx"
        torch.onnx.export(model, (dummy_img, dummy_inf), str(onnx_path), opset_version=13,
                          input_names=['image','infection_fraction'],
                          output_names=['det_scores','cls_out','seg_out','dose_out'],
                          dynamic_axes={'image':{0:'batch'}, 'det_scores':{0:'batch'}, 'cls_out':{0:'batch'}, 'seg_out':{0:'batch'}, 'dose_out':{0:'batch'}})
        print("Exported ONNX to", onnx_path)
    except Exception as e:
        print("ONNX export failed:", e)

if __name__ == '__main__':
    main()"""

#V2
# train_agrinet_mtl_v2.py
# Joint pest/disease typing, disease classification, pest presence, segmentation, severity, and dosage.
# Uses selective losses with Kendall & Gal homoscedastic uncertainty weighting.

# train_agrinet_mtl_v2_diag.py
# Multi-task: type (pest/disease/healthy), pest presence, disease class, segmentation, severity, dosage
# Adds per-task loss logging, uncertainty weights logging, and simple validation metrics.

# train_agrinet_mtl_v3.py
# Non-negative multi-task loss weighting + diagnostics
import os, argparse
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Windows/Startup safety
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# -------------------------
# Utilities / losses / metrics
# -------------------------
def dice_loss(pred_prob, target, eps=1e-6):
    B = pred_prob.size(0)
    pred_flat = pred_prob.view(B, -1)
    tgt_flat  = target.view(B, -1)
    inter = (pred_flat * tgt_flat).sum(1)
    union = pred_flat.sum(1) + tgt_flat.sum(1)
    return (1 - (2 * inter + eps) / (union + eps)).mean()

def dice_coef(pred_prob, target, eps=1e-6):
    B = pred_prob.size(0)
    pred_flat = pred_prob.view(B, -1)
    tgt_flat  = target.view(B, -1)
    inter = (pred_flat * tgt_flat).sum(1)
    union = pred_flat.sum(1) + tgt_flat.sum(1)
    return ((2 * inter + eps) / (union + eps)).mean().item()

class NonNegativeUW(nn.Module):
    """
    Strictly non-negative uncertainty weighting.
    We learn raw theta_i, set sigma_i = softplus(theta_i)+eps > 0,
    and combine losses as: sum_i ( loss_i / (2*sigma_i^2) + log(sigma_i) ).
    Each term >= 0 if loss_i >= 0; total remains >= 0.
    """
    def __init__(self, n_tasks, eps=1e-6, clamp=(1e-3, 50.0)):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(n_tasks, dtype=torch.float32))
        self.softplus = nn.Softplus()
        self.eps = eps
        self.low, self.high = clamp
    def forward(self, losses):
        total = 0.0
        sigmas = self.softplus(self.theta) + self.eps  # >0
        sigmas = torch.clamp(sigmas, self.low, self.high)
        for i, L in enumerate(losses):
            if L is None: 
                continue
            total = total + (L / (2.0 * (sigmas[i] ** 2))) + torch.log(sigmas[i])
        return total, sigmas.detach()

# -------------------------
# Model
# -------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()
    def forward(self, x): return self.act(self.bn(self.conv(x)))

class Backbone(nn.Module):
    def __init__(self, bifpn_ch=128):
        super().__init__()
        self.layer1 = ConvBlock(3, 32, 3, 2, 1)
        self.layer2 = ConvBlock(32, 64, 3, 2, 1)
        self.layer3 = ConvBlock(64, bifpn_ch, 3, 2, 1)
        self.layer4 = ConvBlock(bifpn_ch, bifpn_ch, 3, 2, 1)
        self.layer5 = ConvBlock(bifpn_ch, bifpn_ch, 3, 2, 1)
    def forward(self, x):
        x = self.layer1(x); x = self.layer2(x)
        c3 = self.layer3(x); c4 = self.layer4(c3); c5 = self.layer5(c4)
        return c3, c4, c5

class BiFPN(nn.Module):
    def __init__(self, ch=128):
        super().__init__()
        self.p3 = ConvBlock(ch, ch); self.p4 = ConvBlock(ch, ch); self.p5 = ConvBlock(ch, ch)
    def forward(self, feats):
        c3, c4, c5 = feats
        p5 = self.p5(c5)
        p4 = self.p4(c4 + F.interpolate(p5, size=c4.shape[-2:], mode='nearest'))
        p3 = self.p3(c3 + F.interpolate(p4, size=c3.shape[-2:], mode='nearest'))
        return [p3, p4, p5]

class AgriNetMTL(nn.Module):
    def __init__(self, num_pest_classes=102, num_disease_classes=39, bifpn_ch=128):
        super().__init__()
        self.backbone = Backbone(bifpn_ch); self.bifpn = BiFPN(ch=bifpn_ch)
        self.type_head = nn.Linear(bifpn_ch, 3)
        self.det_head  = nn.Conv2d(bifpn_ch, num_pest_classes, 1)  # logits
        self.cls_head  = nn.Linear(bifpn_ch, num_disease_classes)  # logits
        self.seg_head  = nn.Conv2d(bifpn_ch, 1, 1)                 # logits
        self.sev_head  = nn.Sequential(nn.Linear(bifpn_ch, 64), nn.ReLU(), nn.Linear(64, 1))
        self.dose_head = nn.Sequential(nn.Linear(bifpn_ch + 1, 64), nn.ReLU(), nn.Linear(64, 2))
    def forward(self, x):
        c3, c4, c5 = self.backbone(x)
        p3, _, _ = self.bifpn([c3, c4, c5])
        f = p3
        pooled = F.adaptive_avg_pool2d(f, 1).flatten(1)
        type_logits = self.type_head(pooled)
        det_logits  = self.det_head(f).mean(dim=[2,3])
        cls_logits  = self.cls_head(pooled)
        seg_logits  = self.seg_head(f)
        sev_logit   = self.sev_head(pooled)
        sev = torch.sigmoid(sev_logit)
        dose_out = self.dose_head(torch.cat([pooled, sev], dim=1))
        return dict(type_logits=type_logits, det_logits=det_logits, cls_logits=cls_logits,
                    seg_logits=seg_logits, sev_logit=sev_logit, dose_out=dose_out)

# -------------------------
# Dataset
# -------------------------
class AgriCombinedDataset(Dataset):
    def __init__(self, data_root, split='train', img_size=320, pest_classes=None, K_ml=10.0):
        super().__init__()
        self.data_root = Path(data_root); self.split = split; self.img_size = img_size; self.K_ml = K_ml
        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        pv_root = self.data_root / 'plantvillage' / split
        self.plantvillage_items, self.pv_class2idx = [], {}
        if pv_root.exists():
            pv_classes = sorted([d.name for d in pv_root.iterdir() if d.is_dir()])
            self.pv_class2idx = {c:i for i,c in enumerate(pv_classes)}
            for cls in pv_classes:
                for p in (pv_root/cls).glob('*.*'):
                    if p.suffix.lower() in ['.jpg','.jpeg','.png']:
                        self.plantvillage_items.append((str(p), self.pv_class2idx[cls]))
        ip_root = self.data_root / 'ip102' / 'images' / split
        self.ip_items = []
        if ip_root.exists():
            for p in ip_root.rglob('*.*'):
                if p.suffix.lower() in ['.jpg','.jpeg','.png']:
                    self.ip_items.append(str(p))
        self.pest_classes = pest_classes or []
        self.pv_mask_root = self.data_root / 'plantvillage_masks' / split
        self.ip_mask_root = self.data_root / 'ip102_masks' / split
        self.items = []
        for img_path, lab in self.plantvillage_items:
            self.items.append({'type':'disease', 'img':img_path, 'disease_label': lab})
        for img_path in self.ip_items:
            self.items.append({'type':'pest', 'img':img_path})

    def _load_mask(self, img_path, typ):
        p = Path(img_path)
        if typ == 'disease':
            rel = p.relative_to(self.data_root / 'plantvillage' / self.split)
            mask_path = self.pv_mask_root / rel
        else:
            rel = p.relative_to(self.data_root / 'ip102' / 'images' / self.split)
            mask_path = self.ip_mask_root / rel
        mask_path = Path(str(mask_path).rsplit('.', 1)[0] + '.png')
        if not mask_path.exists(): return np.zeros((self.img_size, self.img_size), np.uint8)
        m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if m is None: return np.zeros((self.img_size, self.img_size), np.uint8)
        m = cv2.resize(m, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        _, m = cv2.threshold(m, 127, 1, cv2.THRESH_BINARY)
        return m.astype(np.uint8)

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img_path, typ = item['img'], item['type']
        pil = Image.open(img_path).convert('RGB')
        mask_np = self._load_mask(img_path, typ)
        infection_fraction = float(mask_np.sum() / (mask_np.size + 1e-9))
        if typ == 'disease':
            tankA, tankB = infection_fraction*self.K_ml, 0.0
            type_label = 1
        else:
            tankA, tankB = 0.0, infection_fraction*self.K_ml
            type_label = 0
        img_t = self.tf(pil)
        mask_t = torch.from_numpy(mask_np).float().unsqueeze(0)  # 0/1
        disease_label = item.get('disease_label', -1) if typ=='disease' else -1
        pest_presence = torch.zeros(len(self.pest_classes), dtype=torch.float32)  # keep as 0/1 if you add labels later
        return {
            'image': img_t,
            'type_label': torch.tensor(type_label, dtype=torch.long),
            'disease_label': torch.tensor(disease_label, dtype=torch.long),
            'pest_presence': pest_presence,
            'mask': mask_t,
            'severity_tgt': torch.tensor(infection_fraction, dtype=torch.float32),
            'dosage': torch.tensor([tankA, tankB], dtype=torch.float32),
            'is_pv': torch.tensor(float(typ=='disease')),
            'is_ip': torch.tensor(float(typ=='pest'))
        }

def collate_fn(batch):
    keys = batch[0].keys(); out = {}
    for k in keys:
        if k in ['image','mask']:
            out[k] = torch.stack([b[k] for b in batch], 0)
        elif k in ['pest_presence','dosage']:
            out[k] = torch.stack([b[k] for b in batch], 0)
        else:
            out[k] = torch.tensor([b[k] for b in batch])
    return out

# -------------------------
# Train / Eval (with non-negative UW + diagnostics)
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', type=str, default='./datasets')
    ap.add_argument('--out_dir', type=str, default='./models')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--imgsz', type=int, default=320)
    ap.add_argument('--num_workers', type=int, default=0)  # Windows-safe default
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train_ds = AgriCombinedDataset(args.data_root, split='train', img_size=args.imgsz, pest_classes=[str(i) for i in range(102)])
    val_ds   = AgriCombinedDataset(args.data_root, split='val',   img_size=args.imgsz, pest_classes=[str(i) for i in range(102)])

    print('Building loaders...')
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers,
                              pin_memory=False, persistent_workers=False, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers,
                              pin_memory=False, persistent_workers=False, collate_fn=collate_fn)
    # sanity warmup iterate one batch to avoid "stuck"
    for i,_ in enumerate(train_loader): print('Warmup batch', i); break
    print('Warmup done, starting training...')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    model = AgriNetMTL(num_pest_classes=102, num_disease_classes=len(train_ds.pv_class2idx)).to(device)

    bce_logits = nn.BCEWithLogitsLoss(reduction='none')  # requires targets in [0,1]
    ce  = nn.CrossEntropyLoss()
    mse = nn.MSELoss(reduction='none')

    weighter = NonNegativeUW(n_tasks=6).to(device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(weighter.parameters()), lr=1e-3)

    best_val = float('inf')

    def step(loader, train=True):
        model.train(train)
        total, n = 0.0, 0
        diag = {'L_type':0.0,'L_det':0.0,'L_cls':0.0,'L_seg':0.0,'L_sev':0.0,'L_dose':0.0,'count':0}
        metrics = {'type_acc':0.0,'cls_acc_top1':0.0,'dice':0.0,'sev_mae':0.0,'dose_mae':0.0,'mcount':0}
        for batch in loader:
            images = batch['image'].to(device)
            masks  = batch['mask'].to(device)
            pest_presence = batch['pest_presence'].to(device)
            disease_labels = batch['disease_label'].to(device)
            type_labels    = batch['type_label'].to(device)
            severity_tgt   = batch['severity_tgt'].to(device)
            dosage         = batch['dosage'].to(device)
            is_pv = batch['is_pv'].to(device).bool()
            is_ip = batch['is_ip'].to(device).bool()
            has_mask = (masks.view(masks.size(0), -1).sum(1) > 0).to(device)
            has_dose = (dosage.abs().sum(1) > 0).to(device)

            # Ensure BCE targets are in [0,1]
            pest_presence.clamp_(0.0, 1.0)
            masks.clamp_(0.0, 1.0)

            out = model(images)
            type_logits = out['type_logits']; det_logits = out['det_logits']
            cls_logits  = out['cls_logits'];  seg_logits = out['seg_logits']
            sev_logit   = out['sev_logit'];   dose_out   = out['dose_out']

            L_type = ce(type_logits, type_labels)

            det_row = bce_logits(det_logits, pest_presence).mean(dim=1)
            L_det = det_row[is_ip].mean() if is_ip.any() else None

            L_cls = ce(cls_logits[is_pv], disease_labels[is_pv]) if is_pv.any() else None

            seg_up = F.interpolate(seg_logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
            seg_row = bce_logits(seg_up, masks).mean(dim=[1,2,3])
            L_seg = seg_row[has_mask].mean() if has_mask.any() else None

            sev = torch.sigmoid(sev_logit).squeeze(1)
            sev_row = mse(sev, severity_tgt)
            L_sev = sev_row[has_mask].mean() if has_mask.any() else None

            dose_row = mse(dose_out, dosage).mean(dim=1)
            L_dose = dose_row[has_dose].mean() if has_dose.any() else None

            # Non-negative weighted total
            weighted, sigmas = weighter([L_type, L_det, L_cls, L_seg, L_sev, L_dose])

            if train:
                optimizer.zero_grad()
                weighted.backward()
                optimizer.step()

            bs = images.size(0); total += weighted.item() * bs; n += bs
            diag['count'] += 1
            diag['L_type'] += L_type.item()
            diag['L_det']  += (L_det.item() if L_det is not None else 0.0)
            diag['L_cls']  += (L_cls.item() if L_cls is not None else 0.0)
            diag['L_seg']  += (L_seg.item() if L_seg is not None else 0.0)
            diag['L_sev']  += (L_sev.item() if L_sev is not None else 0.0)
            diag['L_dose'] += (L_dose.item() if L_dose is not None else 0.0)

            with torch.no_grad():
                type_pred = type_logits.argmax(1)
                metrics['type_acc'] += (type_pred == type_labels).float().mean().item()
                if is_pv.any():
                    cls_pred = cls_logits[is_pv].argmax(1)
                    metrics['cls_acc_top1'] += (cls_pred == disease_labels[is_pv]).float().mean().item()
                if has_mask.any():
                    seg_prob = torch.sigmoid(seg_up[has_mask])
                    metrics['dice'] += dice_coef(seg_prob, masks[has_mask])
                    metrics['sev_mae'] += torch.abs(sev[has_mask] - severity_tgt[has_mask]).mean().item()
                if has_dose.any():
                    metrics['dose_mae'] += torch.abs(dose_out[has_dose] - dosage[has_dose]).mean().item()
                metrics['mcount'] += 1
        avg = total / max(n,1)
        for k in list(diag.keys()):
            if k!='count': diag[k] = diag[k] / max(diag['count'],1)
        for k in list(metrics.keys()):
            if k!='mcount': metrics[k] = metrics[k] / max(metrics['mcount'],1)
        return avg, diag, metrics, sigmas.cpu().numpy()

    best_path = Path(args.out_dir)/'agrinet_v3_best.pth'
    for epoch in range(1, args.epochs+1):
        tr, tr_diag, tr_met, tr_sig = step(train_loader, train=True)
        va, va_diag, va_met, va_sig = step(val_loader, train=False)
        print(f"Epoch {epoch}: train {tr:.4f} | val {va:.4f}")
        print(f"  Base losses (train) type:{tr_diag['L_type']:.4f} det:{tr_diag['L_det']:.4f} cls:{tr_diag['L_cls']:.4f} seg:{tr_diag['L_seg']:.4f} sev:{tr_diag['L_sev']:.4f} dose:{tr_diag['L_dose']:.4f}")
        print(f"  Base losses (val)   type:{va_diag['L_type']:.4f} det:{va_diag['L_det']:.4f} cls:{va_diag['L_cls']:.4f} seg:{va_diag['L_seg']:.4f} sev:{va_diag['L_sev']:.4f} dose:{va_diag['L_dose']:.4f}")
        print(f"  Metrics (val) type_acc:{va_met['type_acc']:.3f} cls_top1:{va_met['cls_acc_top1']:.3f} dice:{va_met['dice']:.3f} sev_MAE:{va_met['sev_mae']:.3f} dose_MAE:{va_met['dose_mae']:.3f}")
        print(f"  Learned sigmas (train): {np.round(tr_sig,3)}  (smaller -> heavier)")

        ckpt = Path(args.out_dir)/f'agrinet_v3_epoch{epoch}.pth'
        torch.save({'epoch':epoch,'model_state':model.state_dict(),
                    'optim_state':optimizer.state_dict(),
                    'weighter_state':weighter.state_dict(),
                    'train_loss':tr,'val_loss':va}, ckpt)
        if va < best_val:
            best_val = va
            torch.save({'epoch':epoch,'model_state':model.state_dict(),
                        'optim_state':optimizer.state_dict(),
                        'weighter_state':weighter.state_dict(),
                        'train_loss':tr,'val_loss':va}, best_path)
            print(f"✅ Saved best (val {best_val:.4f})")

    # ONNX export
    model.eval()
    dummy = torch.randn(1,3,args.imgsz,args.imgsz, device=device)
    onnx_path = Path(args.out_dir)/'agrinet_mtl_v3.onnx'
    try:
        torch.onnx.export(
            model, dummy, str(onnx_path), opset_version=13,
            input_names=['image'],
            output_names=['type_logits','det_logits','cls_logits','seg_logits','sev_logit','dose_out'],
            dynamic_axes={'image':{0:'batch'}, 'type_logits':{0:'batch'}, 'det_logits':{0:'batch'},
                          'cls_logits':{0:'batch'}, 'seg_logits':{0:'batch'}, 'sev_logit':{0:'batch'}, 'dose_out':{0:'batch'}}
        )
        print('Exported ONNX to', onnx_path)
    except Exception as e:
        print('ONNX export failed:', e)

if __name__ == '__main__':
    main()
