# npuscale

**NPU-accelerated video super-resolution using ONNX Runtime.**

Stock FFmpeg handles decode/encode via pipes; npuscale runs the AI inference on your NPU, GPU, or CPU with concurrent in-flight requests for maximum throughput.

## What it does

Takes a video at any resolution, runs each frame through a super-resolution ONNX model, and produces an upscaled video — 2× or 4× depending on the model.

- **NPU / GPU acceleration** via DirectML (Windows), or CUDA / CPU on any platform
- **Pipelined**: decode, inference, and encode run concurrently as separate OS processes/threads
- **In-flight parallelism**: `--workers N` keeps N frames being inferred simultaneously — the NPU never waits idle
- **Frame-order guaranteed**: reorder buffer ensures frames arrive at the encoder in the correct order regardless of inference completion order
- **Audio preserved**: audio stream is copied as-is from the source
- **Cross-platform**: Windows (DirectML/CUDA/CPU), Linux (CUDA/CPU/OpenVINO¹), macOS (CoreML¹/CPU)

¹ OpenVINO and CoreML EPs can be added via ONNX Runtime extensions.

## Requirements

- [.NET 10 Runtime](https://dotnet.microsoft.com/download/dotnet/10.0)
- `ffmpeg` and `ffprobe` in PATH
- A super-resolution ONNX model (see **Models** below)

## Installation

Download the latest release from [Releases](../../releases/latest) and extract.
The archive contains `npuscale.exe` (Windows) or `npuscale` (Linux/macOS) plus the required ONNX Runtime DLLs.

## Usage

```bash
npuscale -i input.mp4 -o output.mp4 --model sr_2x.onnx -v
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

### Examples

```bash
# NPU / GPU via DirectML (Windows)
npuscale -i in.mp4 -o out.mp4 --model sr_2x.onnx --provider directml -v

# CUDA GPU
npuscale -i in.mp4 -o out.mp4 --model sr_2x.onnx --provider cuda -v

# CPU with 4 parallel workers
npuscale -i in.mp4 -o out.mp4 --model sr_2x.onnx --provider cpu --workers 4 -v

# Model that expects 0–255 input range
npuscale -i in.mp4 -o out.mp4 --model sr_4x.onnx --in-range 0..255 -v
```

## NPU acceleration

On Windows, `--provider directml` offloads inference to any DirectML-capable device — including Intel, AMD, and Qualcomm NPUs. To verify that the NPU is being used:

1. Open **Task Manager → Performance → NPU**
2. Run npuscale with `--provider directml`
3. The NPU graph should show activity

On machines without an NPU, DirectML falls back to the GPU automatically.

## Models

npuscale works with any super-resolution ONNX model that:
- Takes a single NCHW or NHWC float tensor as input
- Outputs a single upscaled float tensor

Tested models:
- **sr_2x.onnx** — 2× bicubic-style upscale (included in the release)
- **Real-ESRGAN x4** — high-quality 4× upscale ([ONNX export instructions](https://github.com/xinntao/Real-ESRGAN))

## Building from source

```bash
git clone https://github.com/ikeno-web/npuscale
cd npuscale
dotnet build -c Release
```

To publish a self-contained binary:
```bash
dotnet publish -c Release -r win-x64 --self-contained -o publish/win-x64
```

## Architecture

```
ffprobe ─ probe width/height/fps/audio
ffmpeg decode (-f rawvideo rgb24 pipe) ──▶ bounded queue
                                              │
                               N inference workers (ORT + DirectML)
                                              │
                                        reorder buffer
                                              │
ffmpeg encode (rawvideo pipe → libx264) ◀────┘
```

Stock FFmpeg is invoked as child processes — no FFmpeg linkage, permissive MIT license.

## License

MIT — see [LICENSE](LICENSE)
