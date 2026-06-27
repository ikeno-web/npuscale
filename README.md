# npuscale

**NPU-accelerated video enhancement using ONNX Runtime — super-resolution and denoising.**

Stock FFmpeg handles decode/encode via pipes; npuscale runs AI inference on your NPU, GPU, or CPU with concurrent in-flight requests for maximum throughput. The pipeline is model-agnostic: give it a super-resolution model to upscale, or a denoise model to clean up noisy footage.

## What it does

Takes a video, runs each frame through an ONNX neural network, and re-encodes the result. Two tasks ship today:

- **Super-resolution** (`realesrgan_x2/x4.onnx`) — Real-ESRGAN RRDB ×23 reconstructs edges and texture for 2× / 4× upscaling
- **Denoise** (`dncnn_color.onnx`, via the `npudenoise` wrapper) — DnCNN removes sensor/compression noise while keeping resolution

Shared engine:
- **NPU / GPU acceleration** via DirectML (Windows), or CUDA / CPU on any platform
- **Pipelined**: decode, inference, and encode run concurrently
- **In-flight parallelism**: `--workers N` keeps N frames being inferred simultaneously (CPU/CUDA; DirectML runs single-worker — see note)
- **Frame-order guaranteed**: reorder buffer preserves frame order regardless of inference completion order
- **Audio preserved**: audio stream is copied as-is from the source
- **Cross-platform**: Windows (DirectML/CUDA/CPU), Linux (CUDA/CPU/OpenVINO¹), macOS (CoreML¹/CPU)

¹ OpenVINO and CoreML EPs can be added via ONNX Runtime extensions.

> **DirectML + workers:** the DirectML execution provider is not safe under
> concurrent `Run()` calls on one session, so npuscale automatically uses
> `--workers 1` with `--provider directml`. For multi-worker parallelism use
> `--provider cpu` (or `cuda`).

## Requirements

- [.NET 10 Runtime](https://dotnet.microsoft.com/download/dotnet/10.0)
- `ffmpeg` and `ffprobe` in PATH ([download](https://ffmpeg.org/download.html))
- An ONNX model (download from [Releases](../../releases/latest))

## Installation

1. Download `npuscale-v1.1-win-x64.zip` from [Releases](../../releases/latest) and extract
2. Download the model(s) you need from the same release page

## Models

| Model | Task | Output from 1280×720 | Size | Description |
|-------|------|----------------------|------|-------------|
| `realesrgan_x2.onnx` | SR 2× | 2560×1440 | 64 MB | Real-ESRGAN x2plus — AI upscaling |
| `realesrgan_x4.onnx` | SR 4× | 5120×2880 | 64 MB | Real-ESRGAN x4plus — AI upscaling |
| `dncnn_color.onnx` | Denoise (spatial) | 1280×720 (same) | 2.6 MB | DnCNN color blind — fast |
| `fastdvdnet_s15/s25/s50.onnx` | Denoise (temporal) | 1280×720 (same) | 9.5 MB | FastDVDnet — 5-frame, best quality (pick σ for noise level) |

SR models use official weights from [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN); denoisers use [cszn/KAIR](https://github.com/cszn/KAIR) (DnCNN) and [m-tassano/FastDVDnet](https://github.com/m-tassano/fastdvdnet) weights — all converted to ONNX.

### Real-ESRGAN vs. simple upscaling

| Method | What it does |
|--------|-------------|
| Bicubic / bilinear | Stretches and blurs pixels — no new information |
| **Real-ESRGAN** | Deep neural network reconstructs edges, textures, and fine detail that was not in the original |

## Usage

### 2× AI super-resolution

```bash
# NPU / GPU via DirectML (Windows — fastest)
npuscale -i input.mp4 -o output_2x.mp4 --model realesrgan_x2.onnx --provider directml -v

# CUDA GPU
npuscale -i input.mp4 -o output_2x.mp4 --model realesrgan_x2.onnx --provider cuda -v

# CPU (slower, works everywhere)
npuscale -i input.mp4 -o output_2x.mp4 --model realesrgan_x2.onnx --provider cpu --workers 4 -v
```

### 4× AI super-resolution

```bash
# NPU / GPU via DirectML (Windows — fastest)
npuscale -i input.mp4 -o output_4x.mp4 --model realesrgan_x4.onnx --provider directml -v

# CUDA GPU
npuscale -i input.mp4 -o output_4x.mp4 --model realesrgan_x4.onnx --provider cuda -v

# CPU
npuscale -i input.mp4 -o output_4x.mp4 --model realesrgan_x4.onnx --provider cpu --workers 4 -v
```

### Denoise

Two denoisers ship, trading speed for quality:

| Model | Type | Speed¹ | Quality | Notes |
|-------|------|-------:|---------|-------|
| `dncnn_color.onnx` | Spatial (1 frame) | ~7 fps | good | DnCNN; the `npudenoise` default |
| `fastdvdnet_s25.onnx` | **Temporal (5 frames)** | ~3.5 fps | **best** | FastDVDnet; uses neighbouring frames |

¹ 1080p on an RTX 4090 via DirectML.

**Spatial** (fast) — the `npudenoise` wrapper injects the DnCNN model:
```bash
npudenoise -i noisy.mp4 -o clean.mp4 --provider directml -v          # Windows
./npudenoise.sh -i noisy.mp4 -o clean.mp4 --provider cpu --workers 4  # Linux/macOS
```

**Temporal** (best quality) — point npuscale at a FastDVDnet model; the pipeline
auto-detects the 5-frame window from the model and slides it across the video:
```bash
npuscale -i noisy.mp4 -o clean.mp4 --model fastdvdnet_s25.onnx --provider directml -v
```

FastDVDnet is non-blind, so pick the model whose baked-in σ matches your footage:
`s15` (light), `s25` (moderate), `s50` (heavy). Matching matters — on a σ≈50 clip
the matched `s50` model scored SSIM 0.917 vs 0.887 for the under-estimating `s25`.

**Measured** — clean 1080p reference, Gaussian noise σ≈25 added, then denoised
(metrics vs. the clean reference, higher = closer to clean):

| Method | PSNR | SSIM | VMAF |
|--------|-----:|-----:|-----:|
| Noisy (input) | 31.0 dB | 0.727 | 88.4 |
| DnCNN — spatial | 37.6 dB | 0.941 | 85.0 |
| **FastDVDnet — temporal** | **39.2 dB** | **0.960** | **86.1** |

The temporal model wins on every metric: by reusing detail from neighbouring
frames it removes noise *without* the spatial over-smoothing that a single-frame
denoiser falls back on. PSNR and SSIM are the standard denoising metrics; VMAF is
shown for reference but is tuned for compression artifacts and reads fine noise as
"texture," so it is not a reliable denoise metric.

> `fastdvdnet_s25.onnx` has a fixed noise level (σ=25) baked in — ideal for
> moderate noise. Re-export with a different σ via `tools/convert_fastdvdnet.py`
> for heavier/lighter noise.

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `-i PATH` | (required) | Input video |
| `-o PATH` | (required) | Output video |
| `--model PATH` | (required) | ONNX model (super-resolution or denoise) |
| `--provider NAME` | `directml` | Execution provider: `directml`, `cuda`, `cpu` |
| `--device-id N` | `0` | GPU/NPU device index |
| `--workers N` | `2` | Concurrent inference workers |
| `--encoder NAME` | `libx264` | FFmpeg video encoder |
| `--crf N` | `18` | Encoder quality (lower = better) |
| `--layout NAME` | `nchw` | Tensor layout: `nchw` or `nhwc` |
| `--in-range RANGE` | `0..1` | Input normalization: `0..1` or `0..255` |
| `-v, --verbose` | off | Show progress and timing |

## NPU acceleration

On Windows, `--provider directml` offloads inference to any DirectML-capable device — including Intel, AMD XDNA, and Qualcomm NPUs. To verify NPU usage:

1. Open **Task Manager → Performance → NPU**
2. Run npuscale with `--provider directml`
3. The NPU utilization graph should show activity during processing

On machines without an NPU, DirectML automatically falls back to the GPU.

## Building from source

```bash
git clone https://github.com/ikeno-web/npuscale
cd npuscale
dotnet build -c Release
```

Self-contained publish:
```bash
# Windows
dotnet publish -c Release -r win-x64 --self-contained -o publish/win-x64

# Linux
dotnet publish -c Release -r linux-x64 --self-contained -o publish/linux-x64

# macOS
dotnet publish -c Release -r osx-x64 --self-contained -o publish/osx-x64
```

### Converting models from PyTorch

Requires Python 3.10+ and PyTorch:
```bash
pip install torch onnx onnxscript
cd tools
python convert_realesrgan.py   # -> realesrgan_x2.onnx + realesrgan_x4.onnx (super-resolution)
python convert_dncnn.py        # -> dncnn_color.onnx (spatial denoise)
python convert_fastdvdnet.py   # -> fastdvdnet_s25.onnx (temporal denoise, 5-frame)
```

## How Real-ESRGAN super-resolution works

### Simple upscaling vs. AI super-resolution

**Bicubic / bilinear upscaling** calculates each new pixel as a weighted average of surrounding original pixels. It cannot add information that was not in the source — the result is always a smooth, slightly blurry version of the original.

**Real-ESRGAN** takes a completely different approach: a deep neural network trained on millions of images *generates* the high-frequency detail that is physically absent from the low-resolution source.

```
Simple upscaling:   [blurry pixel average]  →  smooth but soft output
Real-ESRGAN:        [learned reconstruction]  →  sharp edges + restored texture
```

### What "high-frequency reconstruction" means

An image can be decomposed into:
- **Low-frequency components** — overall brightness, color gradients, large shapes (preserved by any upscaler)
- **High-frequency components** — fine edges, texture detail, grain (lost during downscaling / compression)

When a video is encoded at 720p, the encoder discards high-frequency detail to save bitrate. Simple upscaling cannot recover what was thrown away. Real-ESRGAN's network has learned statistical patterns of how real-world textures, edges, and structures look at high resolution, and uses those patterns to *plausibly reconstruct* the missing detail.

### The RRDB network

Real-ESRGAN uses **RRDB (Residual in Residual Dense Block)** — a 23-layer stack of densely connected residual blocks:

```
Input frame (RGB)
    │
    ▼
[PixelUnshuffle]  ← x2 model only: rearranges spatial pixels into channels
    │              for better low-frequency preservation
    ▼
[Conv 3×3]  →  64 feature maps
    │
    ▼  ×23 times
┌─────────────────────────────────┐
│  Residual Dense Block           │
│  ┌──────────────────────────┐   │
│  │ conv → conv → conv → conv│   │  ← each conv sees all previous outputs
│  │  (densely connected)     │   │     (Dense connection: max gradient flow)
│  └──────────────────────────┘   │
│  × 3  +  residual scaling 0.2   │  ← residual scaling stabilizes training
└─────────────────────────────────┘
    │
    ▼
[Nearest-neighbor upsample × 2 → Conv]  × 2  ← upsample in two stages
    │                                          (avoids checkerboard artifacts)
    ▼
Output frame (2× or 4× resolution)
```

Dense connections inside each block mean every layer receives gradients from all subsequent layers — enabling very deep networks to train stably and capture both fine texture (shallow layers) and semantic structure (deep layers) simultaneously.

### Training: degradation modeling

Real-ESRGAN was trained with a **high-order degradation model** that simulates the full chain of real-world quality loss:

```
High-resolution ground truth
    │
    ▼  (simulated degradation pipeline)
blur → downsample → noise → JPEG compression → blur → downsample → noise → JPEG
    │
    ▼
Low-resolution training input  →  network  →  reconstructed HR  →  loss vs. ground truth
```

By training on this wide range of degradation types, the model generalizes to real compressed video, not just clean synthetic downscales. This is why it handles blocky MPEG artifacts, motion blur, and encoding noise — not just pure resolution loss.

### When super-resolution is effective — and when it isn't

The value of Real-ESRGAN is realized **at display time on a high-resolution screen**, not by measuring pixels at the original resolution.

| Use case | Effect | Reason |
|----------|--------|--------|
| 720p video played on a 4K TV / monitor | ✅ High | TV's built-in scaler uses simple bilinear interpolation. Real-ESRGAN pre-upscaling produces visibly sharper edges and reconstructed texture |
| Upscale 720p → 4K and store as archive master | ✅ High | Permanent 4K file with AI-reconstructed detail. Playback quality no longer depends on the player's scaler |
| Upscale 720p → 4K → downscale to 1080p | ⚡ Moderate | SR restores detail before downsampling. The result is often cleaner than direct 720p → 1080p bicubic, because downsampling from 4K acts as antialiasing on the generated detail |
| Upscale 720p → 1440p → **downscale back to 720p** | ❌ Minimal | Returning to the original resolution recovers no information. This is the "upscale then downscale" case where SR provides no benefit |

> **Summary:** if the output will be viewed or stored at the higher resolution, Real-ESRGAN improves quality. Upscaling only to immediately downscale to the same source resolution is pointless — but that is not the intended use of this tool.

## Architecture

```
ffprobe ─ probe width/height/fps/audio
ffmpeg decode (-f rawvideo rgb24 pipe) ──▶ bounded queue
                                              │
                      N inference workers (ORT + DirectML/CUDA/CPU)
                                              │
                                       reorder buffer
                                              │
ffmpeg encode (rawvideo pipe → libx264) ◀────┘
```

Stock FFmpeg is invoked as child processes — no FFmpeg linkage, MIT license.

## License

MIT — see [LICENSE](LICENSE)
