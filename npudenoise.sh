#!/usr/bin/env bash
# npudenoise — NPU video denoising (DnCNN color blind), built on npuscale.
# Injects the denoise model and forwards all other npuscale arguments.
#
# Usage:
#   npudenoise.sh -i input.mp4 -o output.mp4 [--provider directml] [-v]
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="$DIR/dncnn_color.onnx"
if [ -x "$DIR/npuscale" ]; then
  exec "$DIR/npuscale" --model "$MODEL" "$@"
else
  exec npuscale --model "$MODEL" "$@"
fi
