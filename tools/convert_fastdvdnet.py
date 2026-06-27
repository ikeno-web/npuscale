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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "fastdvdnet"))
from models import FastDVDnet  # official architecture

PTH_URL  = "https://raw.githubusercontent.com/m-tassano/fastdvdnet/master/model.pth"
PTH_FILE = os.path.join("fastdvdnet", "model.pth")
SIGMA    = 25                      # baked-in noise level (matches our test clip)
OUT_FILE = f"fastdvdnet_s{SIGMA}.onnx"


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

wrapped = FastDVDnetFixedSigma(net, SIGMA).eval()

print(f"Exporting ONNX (baked sigma={SIGMA}) ...", flush=True)
dummy = torch.zeros(1, 15, 256, 256)            # 5 frames x 3ch
torch.onnx.export(
    wrapped, dummy, OUT_FILE,
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input":  {0: "batch", 2: "height", 3: "width"},
                  "output": {0: "batch", 2: "height", 3: "width"}},
    opset_version=17, dynamo=False,
)
print(f"Saved: {OUT_FILE}")

import onnx
m = onnx.load(OUT_FILE)
onnx.checker.check_model(m)
inp = m.graph.input[0].type.tensor_type.shape
out = m.graph.output[0].type.tensor_type.shape
print(f"Input:  {[d.dim_param or d.dim_value for d in inp.dim]}")
print(f"Output: {[d.dim_param or d.dim_value for d in out.dim]}")
print("Model check passed.")
