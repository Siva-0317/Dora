import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from agrinet_architecture_training import AgriNetMTL

def export_to_onnx(pth_path="./models/agrinet_best.pth",
                   onnx_path="./models/agrinet_mtl.onnx",
                   img_size=320,
                   num_pest_classes=102,
                   num_disease_classes=39):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = AgriNetMTL(num_pest_classes=num_pest_classes, num_disease_classes=num_disease_classes)
    checkpoint = torch.load(pth_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    # Dummy inputs
    dummy_input = torch.randn(1, 3, img_size, img_size).to(device)
    dummy_fraction = torch.tensor([0.5], dtype=torch.float32).to(device)

    # Export
    torch.onnx.export(
        model,
        (dummy_input, dummy_fraction),
        onnx_path,
        input_names=["image", "infection_fraction"],
        output_names=["pest_scores", "disease_logits", "segmentation_mask", "dosage"],
        dynamic_axes={
            "image": {0: "batch"},
            "infection_fraction": {0: "batch"},
            "pest_scores": {0: "batch"},
            "disease_logits": {0: "batch"},
            "segmentation_mask": {0: "batch"},
            "dosage": {0: "batch"},
        },
        opset_version=11
    )

    print(f"✅ Exported to {onnx_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth", type=str, default="./models/agrinet_best.pth", help="Path to trained .pth model")
    parser.add_argument("--onnx", type=str, default="./models/agrinet_mtl.onnx", help="Path to save ONNX model")
    parser.add_argument("--imgsz", type=int, default=320)
    args = parser.parse_args()

    export_to_onnx(args.pth, args.onnx, img_size=args.imgsz)
