"""
test_agrinet_mtl.py

Evaluates a trained AgriNet-MTL model on validation/test dataset.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, mean_squared_error
import numpy as np
from pathlib import Path

from agrinet_architecture_training import AgriNetMTL, AgriCombinedDataset, collate_fn, dice_loss

def evaluate(model, loader, device):
    model.eval()
    all_preds_cls, all_trues_cls = [], []
    all_preds_det, all_trues_det = [], []
    dice_scores, mse_list = [], []

    with torch.no_grad():
        for images, disease_labels, pest_presence, masks, infection_fraction, dosages in loader:
            images, masks, pest_presence, infection_fraction, dosages = (
                images.to(device), masks.to(device),
                pest_presence.to(device), infection_fraction.to(device), dosages.to(device)
            )

            det_scores, cls_out, seg_out, dose_out = model(images, infection_fraction)

            # ---- Disease classification ----
            mask_disease = (disease_labels != -1)
            if mask_disease.any():
                valid_idx = mask_disease.nonzero(as_tuple=False).squeeze(1)
                preds = cls_out[valid_idx].argmax(dim=1).cpu().numpy()
                trues = disease_labels[valid_idx].cpu().numpy()
                all_preds_cls.extend(preds)
                all_trues_cls.extend(trues)

            # ---- Pest detection (multi-label) ----
            all_preds_det.append(det_scores.cpu().numpy())
            all_trues_det.append(pest_presence.cpu().numpy())

            # ---- Segmentation ----
            seg_pred = F.interpolate(seg_out, size=masks.shape[-2:], mode='bilinear', align_corners=False)
            seg_pred_bin = (seg_pred > 0.5).float()
            inter = (seg_pred_bin * masks).sum(dim=[1,2,3])
            union = (seg_pred_bin + masks).sum(dim=[1,2,3])
            dice_batch = (2*inter + 1e-6) / (union + 1e-6)
            dice_scores.extend(dice_batch.cpu().numpy())

            # ---- Dosage ----
            mse = F.mse_loss(dose_out, dosages, reduction='none').mean(dim=1)
            mse_list.extend(mse.cpu().numpy())

    # Classification metrics
    cls_acc, cls_f1 = 0.0, 0.0
    if all_trues_cls:
        cls_acc = accuracy_score(all_trues_cls, all_preds_cls)
        cls_f1 = f1_score(all_trues_cls, all_preds_cls, average='macro')

    # Pest detection metrics
    y_true = np.vstack(all_trues_det)
    y_pred = np.vstack(all_preds_det)
    try:
        pest_map = average_precision_score(y_true, y_pred, average='macro')
    except:
        pest_map = 0.0

    # Segmentation
    dice_mean = np.mean(dice_scores)

    # Dosage
    dose_rmse = np.sqrt(np.mean(mse_list))

    return {
        'cls_acc': cls_acc,
        'cls_f1': cls_f1,
        'pest_mAP': pest_map,
        'seg_dice': dice_mean,
        'dose_RMSE': dose_rmse
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./datasets')
    parser.add_argument('--weights', type=str, default='./models/agrinet_best.pth')
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--imgsz', type=int, default=320)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset (use val or test split)
    ds = AgriCombinedDataset(args.data_root, split='val', img_size=args.imgsz, pest_classes=[str(i) for i in range(102)])
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=2, collate_fn=collate_fn)

    # Load model
    model = AgriNetMTL(num_pest_classes=102, num_disease_classes=len(ds.pv_class2idx))
    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model = model.to(device)

    # Evaluate
    metrics = evaluate(model, loader, device)
    print("✅ Evaluation results:")
    for k,v in metrics.items():
        print(f"{k}: {v:.4f}")
