"""import cv2
import numpy as np
import onnxruntime as ort
from time import time, sleep
from datetime import datetime
import csv
from pathlib import Path

# Model / runtime
ONNX_PATH = './models/agrinet_mtl_epoch28.onnx'
INPUT_SIZE = 320
CAM_INDEX = 0
MASK_ALPHA = 0.35

# Logging
LOG_PATH = './logs/agrinet_webcam_log.csv'
LOG_EVERY_N_FRAMES = 5

# Gating and fallback
TYPE_CONF_THR = 0.50     # center threshold for type
HYST_BAND = 0.10         # 0.45–0.55 keeps last valid dose
CLAMP_MIN = 0.0          # clamp small negatives
K_ML = 10.0              # fallback coefficient (matches training proxy)
EPS_SEV = 0.05           # minimum severity to output any dose
MIN_MODEL_DOSE = 0.05    # if model dose < this, use fallback

PV_NUM_CLASSES = 39
PV_CLASS_NAMES = [f'disease_{i}' for i in range(PV_NUM_CLASSES)]
TYPE_NAMES = ['pest','disease','healthy']

def np_softmax(z):
    z = z.astype(np.float32)
    z = z - z.max()
    e = np.exp(z)
    return e / (e.sum() + 1e-9)

def preprocess(frame_bgr, size=320):
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = np.transpose(img, (2,0,1))[None, ...]
    return img

def overlay_mask_on_frame(frame_bgr, mask_bin, alpha=0.35, color=(0,255,0)):
    if mask_bin.shape[:2] != frame_bgr.shape[:2]:
        mask_bin = cv2.resize(mask_bin, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask_color = np.zeros_like(frame_bgr)
    mask_color[mask_bin == 255] = color
    out = frame_bgr.copy()
    cv2.addWeighted(mask_color, alpha, out, 1.0, 0.0, dst=out)
    return out

def ensure_logger(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not Path(path).exists():
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['timestamp','type','type_conf','disease_id','disease_name','disease_conf',
                        'severity','tankA_ml','tankB_ml','provider','fps','dose_src'])

def open_camera(index=0, w=640, h=480):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    for _ in range(3):
        ok, _ = cap.read()
        if not ok:
            sleep(0.05)
    return cap

def main():
    ensure_logger(LOG_PATH)

    print('Available EPs:', ort.get_available_providers())
    providers = [('CUDAExecutionProvider', {'device_id': 0}), 'CPUExecutionProvider']
    try:
        sess = ort.InferenceSession(ONNX_PATH, providers=providers)
    except Exception as e:
        print('CUDA EP unavailable, using CPU. Reason:', e)
        sess = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
    print('Session EPs:', sess.get_providers())
    provider0 = sess.get_providers()[0]

    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    names = [n for n in ['type_logits','det_logits','cls_logits','seg_logits','sev_logit','dose_out'] if n in out_names]

    cap = open_camera(CAM_INDEX, 640, 480)
    if not cap.isOpened():
        print('Failed to open camera'); return

    fps_t = time(); fps = 0.0; frame_idx = 0
    prevA, prevB = 0.0, 0.0
    low = TYPE_CONF_THR - HYST_BAND/2
    high = TYPE_CONF_THR + HYST_BAND/2

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            cap = open_camera(CAM_INDEX, 640, 480)
            ok, frame = cap.read()
            if not ok or frame is None:
                err = np.zeros((480,640,3), dtype=np.uint8)
                cv2.putText(err, 'Camera read failed - retrying', (12,28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                cv2.imshow('AgriNet-MTL ONNX Webcam', err)
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break
                continue

        x = preprocess(frame, INPUT_SIZE)
        outputs = sess.run(names, {in_name: x})
        out_map = dict(zip(names, outputs))

        type_logits = out_map['type_logits'][0]
        cls_logits  = out_map['cls_logits'][0]
        seg_logits  = out_map['seg_logits'][0,0]
        sev_logit   = out_map['sev_logit'][0,0]
        dose_out    = out_map['dose_out'][0]

        type_prob = np_softmax(type_logits)
        type_idx  = int(type_prob.argmax())
        type_conf = float(type_prob[type_idx])
        type_name = TYPE_NAMES[type_idx]

        p_cls   = np_softmax(cls_logits)
        cls_idx = int(p_cls.argmax())
        cls_prob= float(p_cls.max())
        cls_name= PV_CLASS_NAMES[cls_idx] if 0 <= cls_idx < len(PV_CLASS_NAMES) else f'id_{cls_idx}'

        seg_prob = 1.0 / (1.0 + np.exp(-seg_logits))
        seg_bin  = (seg_prob > 0.5).astype(np.uint8) * 255

        severity = float(1.0 / (1.0 + np.exp(-sev_logit)))
        dose_out = np.maximum(dose_out, CLAMP_MIN)
        A_raw, B_raw = float(dose_out[0]), float(dose_out[1])

        # Severity-derived fallback
        doseA_fb = K_ML * severity if severity >= EPS_SEV else 0.0
        doseB_fb = K_ML * severity if severity >= EPS_SEV else 0.0

        # Gated + hysteresis dosing with fallback
        dose_src = 'none'
        if type_name == 'disease' and type_conf >= high and severity >= EPS_SEV:
            if A_raw >= MIN_MODEL_DOSE:
                tankA_ml, tankB_ml, dose_src = A_raw, 0.0, 'modelA'
            else:
                tankA_ml, tankB_ml, dose_src = doseA_fb, 0.0, 'fallbackA'
        elif type_name == 'pest' and type_conf >= high and severity >= EPS_SEV:
            if B_raw >= MIN_MODEL_DOSE:
                tankA_ml, tankB_ml, dose_src = 0.0, B_raw, 'modelB'
            else:
                tankA_ml, tankB_ml, dose_src = 0.0, doseB_fb, 'fallbackB'
        elif low <= type_conf < high:
            tankA_ml, tankB_ml, dose_src = prevA, prevB, 'hold'
        else:
            tankA_ml, tankB_ml, dose_src = 0.0, 0.0, 'zero'
        prevA, prevB = tankA_ml, tankB_ml

        overlay = overlay_mask_on_frame(frame, seg_bin, alpha=MASK_ALPHA, color=(0,255,0))

        fps = 0.9*fps + 0.1*(1.0 / max(1e-6, (time()-fps_t))); fps_t = time()
        hud1 = f"Type:{type_name}({type_conf:.2f})  Cls:{cls_name}({cls_prob:.2f})  Sev:{severity:.2f}  A:{tankA_ml:.2f}ml  B:{tankB_ml:.2f}ml [{dose_src}]"
        hud2 = f"FPS:{fps:.1f}  EP:{provider0}"
        cv2.putText(overlay, hud1, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
        cv2.putText(overlay, hud2, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)

        ts = datetime.now().isoformat(timespec='seconds')
        print(f"{ts} | {type_name}({type_conf:.2f}) | {cls_name}({cls_prob:.2f}) | sev {severity:.2f} | A {tankA_ml:.2f}ml | B {tankB_ml:.2f}ml | {dose_src} | {provider0} | FPS {fps:.1f}")
        if frame_idx % LOG_EVERY_N_FRAMES == 0:
            with open(LOG_PATH, 'a', newline='') as f:
                w = csv.writer(f)
                w.writerow([ts, type_name, f"{type_conf:.4f}", cls_idx, cls_name, f"{cls_prob:.4f}",
                            f"{severity:.4f}", f"{tankA_ml:.3f}", f"{tankB_ml:.3f}", provider0, f"{fps:.2f}", dose_src])

        frame_idx += 1
        cv2.imshow('AgriNet-MTL ONNX Webcam', overlay)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print('Logs saved to', LOG_PATH)

if __name__ == '__main__':
    main()"""

import cv2
import numpy as np
import onnxruntime as ort
from time import time, sleep
from datetime import datetime
import csv
from pathlib import Path

# Model / runtime
ONNX_PATH = './models/agrinet_mtl_epoch28.onnx'
INPUT_SIZE = 320
CAM_INDEX = 0
MASK_ALPHA = 0.35

# Logging
LOG_PATH = './logs/agrinet_webcam_log.csv'
LOG_EVERY_N_FRAMES = 5

# Type gating with hysteresis
TYPE_CONF_THR = 0.50
HYST_BAND = 0.10
CLAMP_MIN = 0.0

# Dose fallback from severity
K_ML = 10.0
EPS_SEV = 0.05
MIN_MODEL_DOSE = 0.05

# Reject/idle guardrails (tune these)
TYPE_TEMP = 2.0        # temperature for softmax calibration
ENERGY_T = 1.0         # temperature for energy score
ENERGY_THR = -1.5      # frames with E > ENERGY_THR -> unknown (tune)
MIN_AREA_FRAC_LCC = 0.015   # min area for largest component (1.5%)
SOLIDITY_THR = 0.70         # min solidity for largest component
MORPH_KERNEL = (3,3)
CLEAR_OVERLAY_ON_REJECT = True

PV_NUM_CLASSES = 39
PV_CLASS_NAMES = [f'disease_{i}' for i in range(PV_NUM_CLASSES)]
TYPE_NAMES = ['pest','disease','healthy']

def softmax_temp(logits, T=1.0):
    z = logits.astype(np.float32) / max(1e-6, T)
    m = z.max()
    e = np.exp(z - m)
    return e / (e.sum() + 1e-9)

def energy_score(logits, T=1.0):
    z = logits.astype(np.float32) / max(1e-6, T)
    m = z.max()
    # E = -T * logsumexp(z/T) but numerically stable with m
    return float(-T * (np.log(np.exp(z - m).sum() + 1e-9) + m))

def preprocess(frame_bgr, size=320):
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = np.transpose(img, (2,0,1))[None, ...]
    return img

def mask_postprocess(seg_logits, out_size):
    seg_prob = 1.0 / (1.0 + np.exp(-seg_logits))
    m = (seg_prob > 0.5).astype(np.uint8)
    kernel = np.ones(MORPH_KERNEL, np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
    m = cv2.resize(m, out_size, interpolation=cv2.INTER_NEAREST)
    return m

def largest_component_and_solidity(mask01):
    # returns largest 0/1 mask, area_frac, solidity
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask01.astype(np.uint8), connectivity=8)
    if num <= 1:
        return mask01*0, 0.0, 0.0
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))
    comp = (labels == idx).astype(np.uint8)
    area_frac = float(comp.sum() / comp.size)
    contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solidity = 0.0
    if contours:
        cnt = contours[0]
        hull = cv2.convexHull(cnt)
        a = max(cv2.contourArea(cnt), 1e-6)
        h = max(cv2.contourArea(hull), 1e-6)
        solidity = float(a / h)
    return comp, area_frac, solidity

def overlay_mask_on_frame(frame_bgr, mask01, alpha=0.35, color=(0,255,0)):
    mask_color = np.zeros_like(frame_bgr)
    mask_color[mask01 == 1] = color
    out = frame_bgr.copy()
    cv2.addWeighted(mask_color, alpha, out, 1.0, 0.0, dst=out)
    return out

def ensure_logger(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not Path(path).exists():
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['timestamp','state','type','type_conf','energy',
                        'disease_id','disease_name','disease_conf',
                        'severity','area_lcc','solidity_lcc',
                        'tankA_ml','tankB_ml','provider','fps','dose_src'])

def open_camera(index=0, w=640, h=480):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    for _ in range(3):
        ok, _ = cap.read()
        if not ok:
            sleep(0.05)
    return cap

def main():
    ensure_logger(LOG_PATH)

    print('Available EPs:', ort.get_available_providers())
    providers = [('CUDAExecutionProvider', {'device_id': 0}), 'CPUExecutionProvider']
    try:
        sess = ort.InferenceSession(ONNX_PATH, providers=providers)
    except Exception as e:
        print('CUDA EP unavailable, using CPU. Reason:', e)
        sess = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
    print('Session EPs:', sess.get_providers())
    provider0 = sess.get_providers()[0]

    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    names = [n for n in ['type_logits','det_logits','cls_logits','seg_logits','sev_logit','dose_out'] if n in out_names]

    cap = open_camera(CAM_INDEX, 640, 480)
    if not cap.isOpened():
        print('Failed to open camera'); return

    fps_t = time(); fps = 0.0; frame_idx = 0
    prevA, prevB = 0.0, 0.0
    low = TYPE_CONF_THR - HYST_BAND/2
    high = TYPE_CONF_THR + HYST_BAND/2

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            cap = open_camera(CAM_INDEX, 640, 480)
            ok, frame = cap.read()
            if not ok or frame is None:
                err = np.zeros((480,640,3), dtype=np.uint8)
                cv2.putText(err, 'Camera read failed - retrying', (12,28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                cv2.imshow('AgriNet-MTL ONNX Webcam', err)
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break
                continue

        h0, w0 = frame.shape[:2]
        x = preprocess(frame, INPUT_SIZE)
        outputs = sess.run(names, {in_name: x})
        out_map = dict(zip(names, outputs))

        type_logits = out_map['type_logits'][0]
        cls_logits  = out_map['cls_logits'][0]
        seg_logits  = out_map['seg_logits'][0,0]
        sev_logit   = out_map['sev_logit'][0,0]
        dose_out    = out_map['dose_out'][0]

        # Calibrated type probs and energy
        type_prob_T = softmax_temp(type_logits, T=TYPE_TEMP)
        type_idx  = int(type_prob_T.argmax())
        type_conf = float(type_prob_T[type_idx])
        type_name = TYPE_NAMES[type_idx]
        E = energy_score(type_logits, T=ENERGY_T)

        # Disease class for display
        p_cls   = softmax_temp(cls_logits, T=1.0)
        cls_idx = int(p_cls.argmax())
        cls_prob= float(p_cls.max())
        cls_name= PV_CLASS_NAMES[cls_idx] if 0 <= cls_idx < len(PV_CLASS_NAMES) else f'id_{cls_idx}'

        # Mask postprocess and largest component filter
        mask0 = mask_postprocess(seg_logits, (w0, h0))
        lcc, area_lcc, sol_lcc = largest_component_and_solidity(mask0)

        severity = float(1.0 / (1.0 + np.exp(-sev_logit)))
        dose_out = np.maximum(dose_out, CLAMP_MIN)
        A_raw, B_raw = float(dose_out[0]), float(dose_out[1])

        # Reject / idle gating: energy + LCC filters + minimum severity
        passes_energy = (E <= ENERGY_THR)
        has_region    = (area_lcc >= MIN_AREA_FRAC_LCC) and (sol_lcc >= SOLIDITY_THR)
        is_significant= (severity >= EPS_SEV)

        if not (passes_energy and has_region and is_significant):
            state = 'unknown'
            tankA_ml = tankB_ml = 0.0
            dose_src = 'reject'
            overlay = frame if CLEAR_OVERLAY_ON_REJECT else overlay_mask_on_frame(frame, lcc, alpha=MASK_ALPHA)
        else:
            # Severity-derived fallback and type-gated dosing
            doseA_fb = K_ML * severity
            doseB_fb = K_ML * severity
            state = 'actionable'
            if type_name == 'disease' and type_conf >= high:
                if A_raw >= MIN_MODEL_DOSE:
                    tankA_ml, tankB_ml, dose_src = A_raw, 0.0, 'modelA'
                else:
                    tankA_ml, tankB_ml, dose_src = doseA_fb, 0.0, 'fallbackA'
            elif type_name == 'pest' and type_conf >= high:
                if B_raw >= MIN_MODEL_DOSE:
                    tankA_ml, tankB_ml, dose_src = 0.0, B_raw, 'modelB'
                else:
                    tankA_ml, tankB_ml, dose_src = 0.0, doseB_fb, 'fallbackB'
            elif low <= type_conf < high:
                tankA_ml, tankB_ml, dose_src = prevA, prevB, 'hold'
            else:
                tankA_ml, tankB_ml, dose_src = 0.0, 0.0, 'zero'
            overlay = overlay_mask_on_frame(frame, lcc, alpha=MASK_ALPHA, color=(0,255,0))

        prevA, prevB = tankA_ml, tankB_ml

        # HUD + logging
        fps = 0.9*fps + 0.1*(1.0 / max(1e-6, (time()-fps_t))); fps_t = time()
        hud1 = f"{state} | Type:{type_name}({type_conf:.2f}) E:{E:.2f}  Cls:{cls_name}({cls_prob:.2f})  Sev:{severity:.2f}  A:{tankA_ml:.2f}ml  B:{tankB_ml:.2f}ml [{dose_src}]"
        hud2 = f"FPS:{fps:.1f}  EP:{provider0}  Area:{area_lcc*100:.1f}% Sol:{sol_lcc:.2f}"
        cv2.putText(overlay, hud1, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
        cv2.putText(overlay, hud2, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)

        ts = datetime.now().isoformat(timespec='seconds')
        print(f"{ts} | {state} | type {type_name} conf {type_conf:.2f} | E {E:.2f} | cls {cls_name} {cls_prob:.2f} | sev {severity:.2f} | area {area_lcc:.3f} sol {sol_lcc:.2f} | A {tankA_ml:.2f} B {tankB_ml:.2f} | {dose_src} | {provider0} | FPS {fps:.1f}")
        if frame_idx % LOG_EVERY_N_FRAMES == 0:
            with open(LOG_PATH, 'a', newline='') as f:
                w = csv.writer(f)
                w.writerow([ts, state, type_name, f"{type_conf:.4f}", f"{E:.3f}",
                            cls_idx, cls_name, f"{cls_prob:.4f}",
                            f"{severity:.4f}", f"{area_lcc:.4f}", f"{sol_lcc:.3f}",
                            f"{tankA_ml:.3f}", f"{tankB_ml:.3f}", provider0, f"{fps:.2f}", dose_src])

        frame_idx += 1
        cv2.imshow('AgriNet-MTL ONNX Webcam', overlay)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print('Logs saved to', LOG_PATH)

if __name__ == '__main__':
    main()
