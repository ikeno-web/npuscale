"""
Generate before/after comparison images for the README.
  docs/sr_compare.png      — bicubic 4x vs Real-ESRGAN 4x
  docs/denoise_compare.png — noisy vs DnCNN denoised
Run from the npuscale/tools directory.
"""
import os
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont

SRC   = "_src_frame.png"          # extracted 1080p frame
OUTDIR = os.path.join("..", "docs")
os.makedirs(OUTDIR, exist_ok=True)

def infer(model_path, arr_u8):
    """arr_u8: HxWx3 uint8 -> denoised/upscaled HxWx3 uint8 (NCHW float 0..1)."""
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    x = (arr_u8.astype(np.float32).transpose(2, 0, 1)[None]) / 255.0
    name = sess.get_inputs()[0].name
    y = sess.run(None, {name: x})[0][0]
    y = np.clip(y.transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
    return y

def font(sz):
    try:
        return ImageFont.truetype("arial.ttf", sz)
    except Exception:
        return ImageFont.load_default()

def label(img, text):
    """Draw a caption bar at the bottom-left of a PIL image."""
    d = ImageDraw.Draw(img)
    f = font(max(16, img.width // 22))
    pad = img.width // 60 + 4
    tb = d.textbbox((0, 0), text, font=f)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    d.rectangle([0, img.height - th - 2 * pad, tw + 2 * pad, img.height], fill=(0, 0, 0))
    d.text((pad, img.height - th - pad - tb[1]), text, fill=(255, 255, 255), font=f)
    return img

def side_by_side(left, right, lt, rt, gap=8):
    label(left, lt); label(right, rt)
    w = left.width + right.width + gap
    h = max(left.height, right.height)
    canvas = Image.new("RGB", (w, h), (20, 22, 30))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + gap, 0))
    return canvas

src = Image.open(SRC).convert("RGB")
W, H = src.size

# ── Super-resolution: bicubic 4x vs Real-ESRGAN 4x ───────────────────────────
lr = src.resize((W // 4, H // 4), Image.BICUBIC)             # simulate low-res source
bic = lr.resize((W, H), Image.BICUBIC)
esr = Image.fromarray(infer("realesrgan_x4.onnx", np.asarray(lr)))
# crop the same detailed region from each, then enlarge for visibility
cx, cy, cw, ch = W // 2 - 200, H // 2 - 150, 400, 300
box = (cx, cy, cx + cw, cy + ch)
bic_c = bic.crop(box).resize((cw * 2, ch * 2), Image.NEAREST)
esr_c = esr.crop(box).resize((cw * 2, ch * 2), Image.NEAREST)
side_by_side(bic_c, esr_c, "Bicubic 4x", "Real-ESRGAN 4x").save(
    os.path.join(OUTDIR, "sr_compare.png"))
print("wrote docs/sr_compare.png")

# ── Denoise: noisy vs DnCNN ──────────────────────────────────────────────────
cx, cy, cw, ch = W // 2 - 300, H // 2 - 200, 600, 400
crop = np.asarray(src.crop((cx, cy, cx + cw, cy + ch))).astype(np.int16)
rng = np.random.default_rng(0)
noisy = np.clip(crop + rng.normal(0, 25, crop.shape), 0, 255).astype(np.uint8)
den = infer("dncnn_color.onnx", noisy)
side_by_side(Image.fromarray(noisy), Image.fromarray(den),
             "Noisy (sigma~25)", "Denoised (DnCNN)").save(
    os.path.join(OUTDIR, "denoise_compare.png"))
print("wrote docs/denoise_compare.png")
