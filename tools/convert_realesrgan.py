"""
Convert Real-ESRGAN x4plus .pth weights to ONNX.
Downloads weights from official GitHub release if not present.
"""
import urllib.request
import sys
import os
import torch
import torch.nn as nn

# ── RRDB architecture (from xinntao/Real-ESRGAN, MIT licensed) ──────────────

class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat,                  num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch,    num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2*num_grow_ch,  num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3*num_grow_ch,  num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4*num_grow_ch,  num_feat,    3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, scale=4,
                 num_feat=64, num_block=23, num_grow_ch=32):
        super().__init__()
        self.scale = scale
        # x2 model uses PixelUnshuffle(2) before conv_first → 3*4=12 channels
        # x1 model uses PixelUnshuffle(4) before conv_first → 3*16=48 channels
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr  = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            feat = nn.functional.pixel_unshuffle(x, 2)
        elif self.scale == 1:
            feat = nn.functional.pixel_unshuffle(x, 4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(
            nn.functional.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(
            nn.functional.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


# ── Model definitions ────────────────────────────────────────────────────────

MODELS = [
    {
        "url":      "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "pth":      "RealESRGAN_x2plus.pth",
        "out":      "realesrgan_x2.onnx",
        "scale":    2,
        "num_block": 23,
    },
    {
        "url":      "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "pth":      "RealESRGAN_x4plus.pth",
        "out":      "realesrgan_x4.onnx",
        "scale":    4,
        "num_block": 23,
    },
]

import onnx

def progress(count, block, total):
    pct = min(100, count * block * 100 // total)
    print(f"\r  {pct}%", end="", flush=True)

for m in MODELS:
    print(f"\n=== {m['out']} (x{m['scale']}) ===", flush=True)

    # ── Download weights ──────────────────────────────────────────────────────
    if not os.path.exists(m["pth"]):
        print(f"Downloading {m['url']} ...", flush=True)
        urllib.request.urlretrieve(m["url"], m["pth"], reporthook=progress)
        print()

    # ── Load model ────────────────────────────────────────────────────────────
    print("Loading model ...", flush=True)
    model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=m["scale"],
                    num_feat=64, num_block=m["num_block"], num_grow_ch=32)

    ckpt  = torch.load(m["pth"], map_location="cpu", weights_only=True)
    state = ckpt.get("params_ema") or ckpt.get("params") or ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    # ── Export ONNX ───────────────────────────────────────────────────────────
    print("Exporting ONNX ...", flush=True)
    dummy = torch.zeros(1, 3, 256, 256)

    torch.onnx.export(
        model, dummy, m["out"],
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input":  {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        },
        opset_version=17,
        dynamo=False,
    )
    print(f"Saved: {m['out']}")

    # ── Sanity check ──────────────────────────────────────────────────────────
    om   = onnx.load(m["out"])
    onnx.checker.check_model(om)
    inp  = om.graph.input[0].type.tensor_type.shape
    out  = om.graph.output[0].type.tensor_type.shape
    print(f"Input:  {[d.dim_param or d.dim_value for d in inp.dim]}")
    print(f"Output: {[d.dim_param or d.dim_value for d in out.dim]}")
    print("Model check passed.")
