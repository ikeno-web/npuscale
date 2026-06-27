@echo off
REM npudenoise — NPU video denoising (DnCNN color blind), built on npuscale.
REM Injects the denoise model and forwards all other npuscale arguments.
REM
REM Usage:
REM   npudenoise -i input.mp4 -o output.mp4 [--provider directml] [-v]
setlocal
set "DIR=%~dp0"
npuscale --model "%DIR%dncnn_color.onnx" %*
