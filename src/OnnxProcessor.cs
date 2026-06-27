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
        var inputTensor = new DenseTensor<float>(dims);

        if (_nchw)
        {
            for (int f = 0; f < n; f++)
            {
                var src = frames[f];
                int c0 = 3 * f;
                for (int i = 0; i < pixels; i++)
                {
                    int y = i / width, x = i % width;
                    inputTensor[0, c0,     y, x] = src[i * 3]     * scale;
                    inputTensor[0, c0 + 1, y, x] = src[i * 3 + 1] * scale;
                    inputTensor[0, c0 + 2, y, x] = src[i * 3 + 2] * scale;
                }
            }
        }
        else
        {
            var src = frames[0];
            for (int i = 0; i < pixels; i++)
            {
                int y = i / width, x = i % width;
                inputTensor[0, y, x, 0] = src[i * 3]     * scale;
                inputTensor[0, y, x, 1] = src[i * 3 + 1] * scale;
                inputTensor[0, y, x, 2] = src[i * 3 + 2] * scale;
            }
        }

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_inputName, inputTensor)
        };

        using var results = _session.Run(inputs);
        var outTensor = results.First().AsTensor<float>();
        var outDims = outTensor.Dimensions;

        int outH = _nchw ? outDims[2] : outDims[1];
        int outW = _nchw ? outDims[3] : outDims[2];
        OutputWidth  = outW;
        OutputHeight = outH;

        int outPixels = outH * outW;
        var output = new byte[outPixels * 3];
        float inv = 255f / _rangeMax;

        if (_nchw)
        {
            for (int i = 0; i < outPixels; i++)
            {
                int y = i / outW, x = i % outW;
                output[i * 3]     = Clamp(outTensor[0, 0, y, x] * inv);
                output[i * 3 + 1] = Clamp(outTensor[0, 1, y, x] * inv);
                output[i * 3 + 2] = Clamp(outTensor[0, 2, y, x] * inv);
            }
        }
        else
        {
            for (int i = 0; i < outPixels; i++)
            {
                int y = i / outW, x = i % outW;
                output[i * 3]     = Clamp(outTensor[0, y, x, 0] * inv);
                output[i * 3 + 1] = Clamp(outTensor[0, y, x, 1] * inv);
                output[i * 3 + 2] = Clamp(outTensor[0, y, x, 2] * inv);
            }
        }

        return output;
    }

    private static byte Clamp(float v) =>
        v <= 0f ? (byte)0 : v >= 255f ? (byte)255 : (byte)(v + 0.5f);

    public void Dispose() => _session.Dispose();
}
