@echo off
REM npudenoise — NPU video denoising (DnCNN), built on npuscale.
REM Injects the bundled denoise model; forwards all other npuscale arguments.
REM Usage: npudenoise -i input.mp4 -o output.mp4 [--provider directml] [-v]
"%~dp0npuscale.exe" --model "%~dp0dncnn_color.onnx" %*
