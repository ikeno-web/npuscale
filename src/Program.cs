namespace NpuScale;

public static class Program
{
    public static async Task<int> Main(string[] args)
    {
        string? input = null, output = null, model = null;
        string provider = "directml";
        int deviceId = 0, workers = 2, crf = 18;
        string encoder = "libx264";
        bool nchw = true, verbose = false;
        float rangeMax = 1.0f;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "-i":           input    = args[++i]; break;
                case "-o":           output   = args[++i]; break;
                case "--model":      model    = args[++i]; break;
                case "--provider":   provider = args[++i]; break;
                case "--device-id":  deviceId = int.Parse(args[++i]); break;
                case "--workers":    workers  = int.Parse(args[++i]); break;
                case "--encoder":    encoder  = args[++i]; break;
                case "--crf":        crf      = int.Parse(args[++i]); break;
                case "--layout":
                    nchw = args[++i].Equals("nchw", StringComparison.OrdinalIgnoreCase);
                    break;
                case "--in-range":
                    var range = args[++i];
                    rangeMax = range switch
                    {
                        "0..1"   => 1.0f,
                        "0..255" => 255.0f,
                        _        => float.Parse(range.Split("..")[1],
                                        System.Globalization.CultureInfo.InvariantCulture)
                    };
                    break;
                case "--verbose" or "-v": verbose = true; break;
                case "--help"    or "-h": PrintHelp(); return 0;
                default:
                    Console.Error.WriteLine($"Unknown option: {args[i]}");
                    PrintHelp();
                    return 1;
            }
        }

        if (input == null || output == null || model == null)
        {
            Console.Error.WriteLine("Error: -i, -o, and --model are required.");
            PrintHelp();
            return 1;
        }
        if (!File.Exists(input))  { Console.Error.WriteLine($"Input not found: {input}");  return 1; }
        if (!File.Exists(model))  { Console.Error.WriteLine($"Model not found: {model}");  return 1; }

        if (verbose)
        {
            Console.Error.WriteLine("npuscale — NPU-accelerated video super-resolution");
            Console.Error.WriteLine($"Provider: {provider}, Device: {deviceId}, Workers: {workers}");
            Console.Error.WriteLine($"Model: {model}, Layout: {(nchw ? "NCHW" : "NHWC")}, Range: 0..{rangeMax}");
        }

        try
        {
            using var proc = new OnnxProcessor(model, provider, deviceId, nchw, rangeMax);
            var pipeline   = new Pipeline(input, output, proc, workers, encoder, crf, verbose);
            await pipeline.RunAsync();
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error: {ex.Message}");
            if (verbose) Console.Error.WriteLine(ex.StackTrace);
            return 1;
        }
    }

    private static void PrintHelp() => Console.Error.WriteLine(@"
Usage: npuscale -i INPUT -o OUTPUT --model MODEL.onnx [options]

Required:
  -i PATH           Input video file
  -o PATH           Output video file
  --model PATH      ONNX super-resolution model

Options:
  --provider NAME   Execution provider: directml (default/Windows), cuda, cpu
  --device-id N     Device index (default: 0)
  --workers N       Concurrent inference workers (default: 2)
  --encoder NAME    Video encoder (default: libx264)
  --crf N           Encoder quality (default: 18, lower = better)
  --layout NAME     Tensor layout: nchw (default) or nhwc
  --in-range RANGE  Input normalization: 0..1 (default) or 0..255
  -v, --verbose     Show progress and timing
  -h, --help        Show this help

Examples:
  npuscale -i in.mp4 -o out.mp4 --model sr_2x.onnx -v
  npuscale -i in.mp4 -o out.mp4 --model sr_2x.onnx --provider cpu --workers 4
  npuscale -i in.mp4 -o out.mp4 --model sr_2x.onnx --provider directml --in-range 0..255
");
}
