# npuscale

**NPU-accelerated video super-resolution using ONNX Runtime.**

Stock FFmpeg handles decode/encode via pipes; npuscale runs AI inference on your NPU, GPU, or CPU with concurrent in-flight requests for maximum throughput.

## What it does

Takes a video at any resolution, runs each frame through a **Real-ESRGAN** super-resolution model, and produces an upscaled video with AI-reconstructed detail — not just pixel stretching.

- **Real-ESRGAN AI super-resolution**: RRDB (Residual in Residual Dense Block) × 23 layers reconstruct edges, textures, and fine detail
- **2× or 4× upscaling**: 1280×720 → 2560×1440 or 5120×2880
- **NPU / GPU acceleration** via DirectML (Windows), or CUDA / CPU on any platform
- **Pipelined**: decode, inference, and encode run concurrently
- **In-flight parallelism**: `--workers N` keeps N frames being inferred simultaneously
- **Frame-order guaranteed**: reorder buffer preserves frame order regardless of inference completion order
- **Audio preserved**: audio stream is copied as-is from the source
- **Cross-platform**: Windows (DirectML/CUDA/CPU), Linux (CUDA/CPU/OpenVINO¹), macOS (CoreML¹/CPU)

¹ OpenVINO and CoreML EPs can be added via ONNX Runtime extensions.

## Requirements

- [.NET 10 Runtime](https://dotnet.microsoft.com/download/dotnet/10.0)
- `ffmpeg` and `ffprobe` in PATH ([download](https://ffmpeg.org/download.html))
- A Real-ESRGAN ONNX model (download from [Releases](../../releases/latest))

## Installation

1. Download `npuscale-v1.1-win-x64.zip` from [Releases](../../releases/latest) and extract
2. Download `realesrgan_x2.onnx` (2×) and/or `realesrgan_x4.onnx` (4×) from the same release page

## Models

| Model | Scale | Output from 1280×720 | Size | Description |
|-------|-------|----------------------|------|-------------|
| `realesrgan_x2.onnx` | 2× | 2560×1440 | 64 MB | Real-ESRGAN x2plus — AI 2× super-resolution |
| `realesrgan_x4.onnx` | 4× | 5120×2880 | 64 MB | Real-ESRGAN x4plus — AI 4× super-resolution |

Both models use official weights from [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) converted to ONNX format.

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

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `-i PATH` | (required) | Input video |
| `-o PATH` | (required) | Output video |
| `--model PATH` | (required) | ONNX super-resolution model |
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
python convert_realesrgan.py   # downloads weights and exports realesrgan_x2.onnx + realesrgan_x4.onnx
```

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
