"""
Build a minimal TEMPORAL super-resolution ONNX model to validate that the
npuscale pipeline handles multi-frame input AND upscaled output together.

This is a *baseline* VSR op (temporal average of 3 frames + 2x upscale), not a
trained network — it proves the pipeline path end-to-end. Production-quality
VSR (EDVR / BasicVSR) relies on deformable convolution / recurrent flow that
does not export to standard ONNX; this validates the plumbing those would use.

  input:  (1, 9, H, W)   = 3 consecutive frames x 3 RGB channels, range [0,1]
  output: (1, 3, 2H, 2W) = 2x-upscaled temporal average, range [0,1]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT_FILE = "temporal_sr2x_avg.onnx"


class TemporalSR2x(nn.Module):
    def forward(self, x):                      # x: (N, 9, H, W)
        f0 = x[:, 0:3, :, :]
        f1 = x[:, 3:6, :, :]
        f2 = x[:, 6:9, :, :]
        avg = (f0 + f1 + f2) / 3.0             # multi-frame fusion (noise averaging)
        return F.interpolate(avg, scale_factor=2.0,
                             mode="bilinear", align_corners=False)


model = TemporalSR2x().eval()
dummy = torch.zeros(1, 9, 64, 64)
print("Exporting ONNX ...", flush=True)
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
