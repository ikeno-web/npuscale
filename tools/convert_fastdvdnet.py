"""
Convert FastDVDnet (m-tassano/FastDVDnet, MIT) temporal video denoiser to ONNX.

FastDVDnet denoises the CENTER frame of a 5-frame window. Its forward takes
  x:         (N, 15, H, W)  = 5 frames x 3 RGB channels, range [0,1]
  noise_map: (N, 1,  H, W)  = sigma/255, range [0,1]
and returns the denoised center frame (N, 3, H, W).

We bake a FIXED sigma into a single-input wrapper so the exported ONNX takes
just the 15-channel stack -> 1 frame. This keeps the npuscale pipeline simple:
feed 5 consecutive frames stacked, get 1 denoised frame back.
"""
import os, sys, urllib.request
import torch
import torch.nn as nn

_FD = os.path.join(os.path.dirname(__file__), "fastdvdnet")
os.makedirs(_FD, exist_ok=True)
_MODELS = os.path.join(_FD, "models.py")
if not os.path.exists(_MODELS):
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/m-tassano/fastdvdnet/master/models.py", _MODELS)
sys.path.insert(0, _FD)
from models import FastDVDnet  # official architecture (m-tassano/fastdvdnet, MIT)

PTH_URL  = "https://raw.githubusercontent.com/m-tassano/fastdvdnet/master/model.pth"
PTH_FILE = os.path.join("fastdvdnet", "model.pth")
# Export one model per noise level (light / moderate / heavy). FastDVDnet is
# non-blind, so the baked-in sigma should match the footage for best results.
SIGMAS   = [15, 25, 50]


class FastDVDnetFixedSigma(nn.Module):
    """Single-input wrapper: builds a constant noise map from a fixed sigma."""
    def __init__(self, model, sigma):
        super().__init__()
        self.model = model
        self.level = sigma / 255.0

    def forward(self, x):                       # x: (N, 15, H, W)
        # constant (N,1,H,W) noise map that tracks x's spatial dims for ONNX
        noise_map = torch.zeros_like(x[:, :1, :, :]) + self.level
        return self.model(x, noise_map)


if not os.path.exists(PTH_FILE):
    print(f"Downloading {PTH_URL} ...", flush=True)
    urllib.request.urlretrieve(PTH_URL, PTH_FILE)

print("Loading weights ...", flush=True)
state = torch.load(PTH_FILE, map_location="cpu", weights_only=True)
# Official checkpoint is saved from a DataParallel model -> strip "module." prefix
state = { (k[7:] if k.startswith("module.") else k): v for k, v in state.items() }

net = FastDVDnet(num_input_frames=5)
net.load_state_dict(state, strict=True)
net.eval()

import onnx
dummy = torch.zeros(1, 15, 256, 256)            # 5 frames x 3ch
for sigma in SIGMAS:
    out_file = f"fastdvdnet_s{sigma}.onnx"
    wrapped = FastDVDnetFixedSigma(net, sigma).eval()
    print(f"Exporting {out_file} (baked sigma={sigma}) ...", flush=True)
    torch.onnx.export(
        wrapped, dummy, out_file,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input":  {0: "batch", 2: "height", 3: "width"},
                      "output": {0: "batch", 2: "height", 3: "width"}},
        opset_version=17, dynamo=False,
    )
    m = onnx.load(out_file)
    onnx.checker.check_model(m)
    print(f"  saved + checked: {out_file}")
