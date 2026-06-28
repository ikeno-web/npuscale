"""
Convert ONNX models to FP16 (half precision) for ~2x faster GPU/NPU inference.

Uses keep_io_types=True: the model's inputs/outputs stay FP32 (Cast nodes are
inserted at the boundaries) while the heavy convolutions run in FP16 internally.
That means npuscale needs NO code change — it still feeds/reads FP32 — but the
compute and weight bandwidth halve.

Usage:  python make_fp16.py model1.onnx [model2.onnx ...]
        (default: the bundled denoise + SR models)
"""
import sys, os
import onnx
from onnxconverter_common import float16

defaults = ["dncnn_color.onnx", "realesrgan_x2.onnx", "realesrgan_x4.onnx"]
models = sys.argv[1:] or [m for m in defaults if os.path.exists(m)]

for path in models:
    model = onnx.load(path)
    fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    out = path.replace(".onnx", "_fp16.onnx")
    onnx.save(fp16, out)
    a, b = os.path.getsize(path) / 1e6, os.path.getsize(out) / 1e6
    print(f"{path} ({a:.1f} MB) -> {out} ({b:.1f} MB)")
