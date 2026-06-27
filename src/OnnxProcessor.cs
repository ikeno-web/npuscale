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
    }

    // Thread-safe: InferenceSession.Run is safe to call concurrently.
    // Each worker should call Process with its own rgb24/output buffers.
    public byte[] Process(byte[] rgb24, int width, int height)
    {
        int pixels = width * height;
        float scale = _rangeMax / 255f;

        var dims = _nchw
            ? new[] { 1, 3, height, width }
            : new[] { 1, height, width, 3 };
        var inputTensor = new DenseTensor<float>(dims);

        if (_nchw)
        {
            for (int i = 0; i < pixels; i++)
            {
                int y = i / width, x = i % width;
                inputTensor[0, 0, y, x] = rgb24[i * 3]     * scale;
                inputTensor[0, 1, y, x] = rgb24[i * 3 + 1] * scale;
                inputTensor[0, 2, y, x] = rgb24[i * 3 + 2] * scale;
            }
        }
        else
        {
            for (int i = 0; i < pixels; i++)
            {
                int y = i / width, x = i % width;
                inputTensor[0, y, x, 0] = rgb24[i * 3]     * scale;
                inputTensor[0, y, x, 1] = rgb24[i * 3 + 1] * scale;
                inputTensor[0, y, x, 2] = rgb24[i * 3 + 2] * scale;
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
