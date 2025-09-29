# export_agrinet_epoch28.py
# Loads epoch 28 checkpoint, exports ONNX, and does a quick sanity forward.

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# ==== Model definition (must match training) ====
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
        dose_out    = self.dose_head(torch.cat([pooled, sev], dim=1))
        return dict(type_logits=type_logits, det_logits=det_logits, cls_logits=cls_logits,
                    seg_logits=seg_logits, sev_logit=sev_logit, dose_out=dose_out)

# ==== Export ====
def main():
    out_dir = Path('./models')
    ckpt_path = out_dir / 'agrinet_v3_epoch28.pth'  # adjust if filename differs
    onnx_path = out_dir / 'agrinet_mtl_epoch28.onnx'
    imgsz = 320

    # Infer class count from training metadata if saved; else set explicitly
    num_pest_classes = 102
    num_disease_classes = 39

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    model = AgriNetMTL(num_pest_classes=num_pest_classes,
                       num_disease_classes=num_disease_classes).to(device)
    ckpt = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ckpt['model_state'], strict=True)
    model.eval()
    print(f'Loaded checkpoint from {ckpt_path}')

    # Sanity forward
    dummy = torch.randn(1, 3, imgsz, imgsz, device=device)
    with torch.no_grad():
        out = model(dummy)
        for k, v in out.items():
            print(k, tuple(v.shape))

    # Export ONNX
    input_names  = ['image']
    output_names = ['type_logits','det_logits','cls_logits','seg_logits','sev_logit','dose_out']
    dynamic_axes = {
        'image': {0: 'batch'},
        'type_logits': {0: 'batch'},
        'det_logits': {0: 'batch'},
        'cls_logits': {0: 'batch'},
        'seg_logits': {0: 'batch'},
        'sev_logit': {0: 'batch'},
        'dose_out': {0: 'batch'}
    }
    torch.onnx.export(
        model, dummy, str(onnx_path),
        opset_version=13,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True
    )
    print('Exported ONNX to', onnx_path)

if __name__ == '__main__':
    main()
