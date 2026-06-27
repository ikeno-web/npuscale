using System.Buffers;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace NpuScale;

public sealed class OnnxProcessor : IDisposable
{
    private readonly InferenceSession _session;
    private readonly string _inputName;
    private readonly string _outputName;
    private readonly bool _nchw;
    private readonly float _rangeMax;

    // ── Persistent IoBinding state (single-worker only; not thread-safe) ──
    // When enabled, input/output tensors are bound once (dimensions are constant
    // within a video) and reused every frame, so ORT skips re-wrapping buffers.
    private bool _reuse;
    private float[]? _inBuf;
    private FixedBufferOnnxValue? _inFixed;
    private float[]? _outBuf;
    private FixedBufferOnnxValue? _outFixed;
    private string[]? _inNames, _outNames;

    /// Enable persistent buffer reuse (only safe with a single worker thread).
    public bool ReuseBuffers { get => _reuse; set => _reuse = value; }

    public int OutputWidth { get; private set; }
    public int OutputHeight { get; private set; }

    /// Number of consecutive frames the model consumes per inference, inferred
    /// from the input channel count (3·N). 1 for SR / spatial denoise; N>1 for
    /// temporal models (e.g. FastDVDnet = 5).
    public int InputFrameCount { get; }

    public OnnxProcessor(string modelPath, string provider, int deviceId,
        bool nchw = true, float rangeMax = 1.0f)
    {
        _nchw = nchw;
        _rangeMax = rangeMax;

        var opts = new SessionOptions();
        opts.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;

        switch (provider.ToLowerInvariant())
        {
            case "directml":
                opts.AppendExecutionProvider_DML(deviceId);
                break;
            case "cuda":
                opts.AppendExecutionProvider_CUDA(deviceId);
                break;
            // "cpu" or unknown → CPU only (appended below)
        }
        opts.AppendExecutionProvider_CPU();

        _session = new InferenceSession(modelPath, opts);
        _inputName = _session.InputMetadata.Keys.First();
        _outputName = _session.OutputMetadata.Keys.First();

        // Infer how many stacked frames the model expects from its channel dim
        // (NCHW: dims[1], NHWC: dims[3]). Dynamic/unknown -> single frame.
        var dims = _session.InputMetadata[_inputName].Dimensions;
        int channels = _nchw
            ? (dims.Length > 1 ? dims[1] : 3)
            : (dims.Length > 3 ? dims[3] : 3);
        InputFrameCount = channels > 0 ? Math.Max(1, channels / 3) : 1;
    }

    // Single-frame convenience (super-resolution / spatial denoise).
    // Thread-safe: InferenceSession.Run is safe to call concurrently (CPU/CUDA);
    // each worker should pass its own buffers.
    public byte[] Process(byte[] rgb24, int width, int height)
        => ProcessWindow(new[] { rgb24 }, width, height);

    // Processes a window of N consecutive frames stacked on the channel axis:
    // the input tensor is (1, 3·N, H, W). N==1 is the ordinary single-frame case;
    // temporal models (N>1, e.g. FastDVDnet) require NCHW layout.
    public byte[] ProcessWindow(IReadOnlyList<byte[]> frames, int width, int height)
    {
        int n = frames.Count;
        int pixels = width * height;
        float scale = _rangeMax / 255f;

        if (n > 1 && !_nchw)
            throw new InvalidOperationException(
                "Temporal (multi-frame) models require NCHW layout.");

        var dims = _nchw
            ? new[] { 1, 3 * n, height, width }
            : new[] { 1, height, width, 3 };
        int inLen = 3 * n * pixels;

        return _reuse
            ? ProcessReused(frames, n, pixels, scale, inLen, dims)
            : ProcessOneShot(frames, n, pixels, scale, inLen, dims);
    }

    // Per-call path: thread-safe, pools the input buffer, wraps a fresh tensor.
    private byte[] ProcessOneShot(IReadOnlyList<byte[]> frames, int n, int pixels,
        float scale, int inLen, int[] dims)
    {
        float[] input = ArrayPool<float>.Shared.Rent(inLen);
        try
        {
            PackInput(input, frames, n, pixels, inLen, scale);
            var inputTensor = new DenseTensor<float>(input.AsMemory(0, inLen), dims);
            using var results = _session.Run(new[]
            {
                NamedOnnxValue.CreateFromTensor(_inputName, inputTensor)
            });
            var outTensor = (DenseTensor<float>)results.First().AsTensor<float>();
            SetOutputDims(outTensor.Dimensions);
            var output = new byte[OutputWidth * OutputHeight * 3];
            UnpackOutput(outTensor.Buffer.Span, output, OutputWidth * OutputHeight);
            return output;
        }
        finally
        {
            ArrayPool<float>.Shared.Return(input);
        }
    }

    // Reuse path: input/output tensors are bound once and reused every frame, so
    // ORT skips re-wrapping buffers each Run. Single-worker only (not thread-safe).
    private byte[] ProcessReused(IReadOnlyList<byte[]> frames, int n, int pixels,
        float scale, int inLen, int[] dims)
    {
        if (_inBuf == null || _inBuf.Length != inLen)
        {
            _inFixed?.Dispose();
            _outFixed?.Dispose();
            _outFixed = null; _outBuf = null;
            _inBuf = new float[inLen];
            _inFixed = FixedBufferOnnxValue.CreateFromTensor(new DenseTensor<float>(_inBuf, dims));
            _inNames = new[] { _inputName };
            _outNames = new[] { _outputName };
        }

        PackInput(_inBuf, frames, n, pixels, inLen, scale);

        if (_outFixed == null)
        {
            // First frame: discover output dims with a normal Run, then bind output.
            using var results = _session.Run(_inNames!, new[] { _inFixed! });
            var outTensor = (DenseTensor<float>)results.First().AsTensor<float>();
            SetOutputDims(outTensor.Dimensions);
            _outBuf = new float[OutputWidth * OutputHeight * 3];
            _outFixed = FixedBufferOnnxValue.CreateFromTensor(
                new DenseTensor<float>(_outBuf, outTensor.Dimensions.ToArray()));
            outTensor.Buffer.Span.CopyTo(_outBuf);
        }
        else
        {
            _session.Run(_inNames!, new[] { _inFixed! }, _outNames!, new[] { _outFixed });
        }

        var output = new byte[OutputWidth * OutputHeight * 3];
        UnpackOutput(_outBuf!, output, OutputWidth * OutputHeight);
        return output;
    }

    private void PackInput(float[] buf, IReadOnlyList<byte[]> frames, int n, int pixels,
        int inLen, float scale)
    {
        if (_nchw)
        {
            // NCHW: channel plane c starts at c·pixels; pixel i sits at +i.
            for (int f = 0; f < n; f++)
            {
                var src = frames[f];
                for (int c = 0; c < 3; c++)
                {
                    int plane = (3 * f + c) * pixels;
                    for (int i = 0; i < pixels; i++)
                        buf[plane + i] = src[i * 3 + c] * scale;
                }
            }
        }
        else
        {
            // NHWC interleaved == packed RGB order, so just scale in place.
            var src = frames[0];
            for (int k = 0; k < inLen; k++)
                buf[k] = src[k] * scale;
        }
    }

    private void UnpackOutput(ReadOnlySpan<float> os, byte[] output, int outPixels)
    {
        float inv = 255f / _rangeMax;
        if (_nchw)
        {
            for (int c = 0; c < 3; c++)
            {
                int plane = c * outPixels;
                for (int i = 0; i < outPixels; i++)
                    output[i * 3 + c] = Clamp(os[plane + i] * inv);
            }
        }
        else
        {
            for (int k = 0; k < outPixels * 3; k++)
                output[k] = Clamp(os[k] * inv);
        }
    }

    private void SetOutputDims(ReadOnlySpan<int> outDims)
    {
        OutputHeight = _nchw ? outDims[2] : outDims[1];
        OutputWidth  = _nchw ? outDims[3] : outDims[2];
    }

    private static byte Clamp(float v) =>
        v <= 0f ? (byte)0 : v >= 255f ? (byte)255 : (byte)(v + 0.5f);

    public void Dispose()
    {
        _inFixed?.Dispose();
        _outFixed?.Dispose();
        _session.Dispose();
    }
}
