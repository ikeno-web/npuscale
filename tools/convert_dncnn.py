"""
Convert DnCNN color blind Gaussian denoiser (KAIR / cszn) to ONNX.

DnCNN predicts the noise residual: output = input - model(input).
Color blind model: 20 conv layers, 3x3, 64 features, ReLU, no BatchNorm.
Input/output: RGB, NCHW, range 0..1, same resolution (denoise, not scale).
"""
import urllib.request
import os
import torch
import torch.nn as nn

# ── DnCNN architecture (matches KAIR network_dncnn.py blind config) ──────────
# Sequential layout so state_dict keys line up as model.0, model.2, ... model.38
#   index 0:      Conv(3 -> 64)         + ReLU(1)
#   index 2..36:  18 x [Conv(64 -> 64)  + ReLU]
#   index 38:     Conv(64 -> 3)         (no activation; tail)

class DnCNN(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=64, nb=20):
        super().__init__()
        layers = [nn.Conv2d(in_nc, nc, 3, 1, 1, bias=True), nn.ReLU(inplace=True)]
        for _ in range(nb - 2):
            layers += [nn.Conv2d(nc, nc, 3, 1, 1, bias=True), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(nc, out_nc, 3, 1, 1, bias=True)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return x - self.model(x)   # residual: clean = noisy - predicted noise


PTH_URL  = "https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_color_blind.pth"
PTH_FILE = "dncnn_color_blind.pth"
OUT_FILE = "dncnn_color.onnx"

if not os.path.exists(PTH_FILE):
    print(f"Downloading {PTH_URL} ...", flush=True)
    def progress(count, block, total):
        if total > 0:
            print(f"\r  {min(100, count*block*100//total)}%", end="", flush=True)
    urllib.request.urlretrieve(PTH_URL, PTH_FILE, reporthook=progress)
    print()

print("Loading weights ...", flush=True)
state = torch.load(PTH_FILE, map_location="cpu", weights_only=True)
if isinstance(state, dict) and "params" in state:
    state = state["params"]

# Diagnostics: confirm the arch matches the checkpoint before loading.
conv_keys = [k for k in state if k.endswith(".weight") and state[k].dim() == 4]
print(f"  checkpoint conv layers: {len(conv_keys)}")
print(f"  first conv shape: {tuple(state[conv_keys[0]].shape)}  "
      f"last conv shape: {tuple(state[conv_keys[-1]].shape)}")

model = DnCNN(in_nc=3, out_nc=3, nc=64, nb=20)
model.load_state_dict(state, strict=True)
model.eval()

print("Exporting ONNX ...", flush=True)
dummy = torch.zeros(1, 3, 256, 256)
torch.onnx.export(
    model, dummy, OUT_FILE,
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input":  {0: "batch", 2: "height", 3: "width"},
                  "output": {0: "batch", 2: "height", 3: "width"}},
    opset_version=17, dynamo=False,
)
print(f"Saved: {OUT_FILE}")

import onnx
m = onnx.load(OUT_FILE)
onnx.checker.check_model(m)
print("Model check passed.")
