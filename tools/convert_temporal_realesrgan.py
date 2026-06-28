"""
Temporal-fusion super-resolution that DOES export to ONNX: a small temporal
front-end (average of 3 neighbouring frames -> noise suppression) feeding the
pretrained Real-ESRGAN x4 network (learned upscaling).

Unlike EDVR / BasicVSR (deformable conv + recurrent flow, which do not export to
standard ONNX), early-fusion-by-averaging is plain tensor ops, so the whole thing
exports cleanly. It beats single-frame Real-ESRGAN on noisy / temporally redundant
footage because the averaging cleans noise before the network hallucinates detail.

  input:  (1, 9,  H,  W)   = 3 consecutive frames x 3 RGB, range [0,1]
  output: (1, 3, 4H, 4W)   = Real-ESRGAN x4 of the temporally fused frame
"""
import os, urllib.request
import torch
import torch.nn as nn

PTH_URL  = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
PTH_FILE = "RealESRGAN_x4plus.pth"
OUT_FILE = "temporal_realesrgan_x4.onnx"


# --- RRDBNet (scale=4 variant), copied to avoid importing the download side effects ---
class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super().__init__()
        self.rdb1, self.rdb2, self.rdb3 = (ResidualDenseBlock(nf, gc) for _ in range(3))

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32):
        super().__init__()
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_hr  = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(nn.functional.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(nn.functional.interpolate(feat, scale_factor=2, mode='nearest')))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


class TemporalRealESRGAN(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):                       # x: (N, 9, H, W) = 3 frames
        fused = (x[:, 0:3] + x[:, 3:6] + x[:, 6:9]) / 3.0
        return self.net(fused)


if not os.path.exists(PTH_FILE):
    print(f"Downloading {PTH_URL} ...", flush=True)
    urllib.request.urlretrieve(PTH_URL, PTH_FILE)

print("Loading Real-ESRGAN x4 weights ...", flush=True)
net = RRDBNet()
ckpt = torch.load(PTH_FILE, map_location="cpu", weights_only=True)
net.load_state_dict(ckpt.get("params_ema") or ckpt.get("params") or ckpt, strict=True)
model = TemporalRealESRGAN(net).eval()

print("Exporting ONNX ...", flush=True)
dummy = torch.zeros(1, 9, 64, 64)
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
inp = m.graph.input[0].type.tensor_type.shape
out = m.graph.output[0].type.tensor_type.shape
print(f"Input:  {[d.dim_param or d.dim_value for d in inp.dim]}")
print(f"Output: {[d.dim_param or d.dim_value for d in out.dim]}")
print("Model check passed.")
