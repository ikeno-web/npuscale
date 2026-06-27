using System.Diagnostics;
using System.Globalization;
using System.Text.Json;

namespace NpuScale;

public record MediaInfo(
    int Width,
    int Height,
    double Fps,
    string PixFmt,
    double Duration,
    bool HasAudio)
{
    public static MediaInfo Probe(string path)
    {
        var psi = new ProcessStartInfo("ffprobe",
            $"-v quiet -print_format json -show_streams -show_format \"{path}\"")
        {
            RedirectStandardOutput = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };

        using var proc = Process.Start(psi)
            ?? throw new InvalidOperationException("Failed to start ffprobe");
        var json = proc.StandardOutput.ReadToEnd();
        proc.WaitForExit();

        if (proc.ExitCode != 0)
            throw new InvalidOperationException($"ffprobe exited with code {proc.ExitCode}");

        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;

        int width = 0, height = 0;
        double fps = 0;
        string pixFmt = "yuv420p";
        bool hasAudio = false;

        foreach (var stream in root.GetProperty("streams").EnumerateArray())
        {
            var codecType = stream.GetProperty("codec_type").GetString();
            if (codecType == "video" && width == 0)
            {
                width = stream.GetProperty("width").GetInt32();
                height = stream.GetProperty("height").GetInt32();
                pixFmt = stream.TryGetProperty("pix_fmt", out var pf)
                    ? pf.GetString() ?? "yuv420p" : "yuv420p";

                if (stream.TryGetProperty("avg_frame_rate", out var fr))
                {
                    var parts = fr.GetString()?.Split('/');
                    if (parts?.Length == 2
                        && double.TryParse(parts[0], out var num)
                        && double.TryParse(parts[1], out var den) && den > 0)
                        fps = num / den;
                }
            }
            else if (codecType == "audio")
            {
                hasAudio = true;
            }
        }

        double duration = 0;
        if (root.TryGetProperty("format", out var fmt)
            && fmt.TryGetProperty("duration", out var dur))
            double.TryParse(dur.GetString(), NumberStyles.Float,
                CultureInfo.InvariantCulture, out duration);

        if (width == 0 || height == 0)
            throw new InvalidOperationException("No video stream found in input");

        return new MediaInfo(width, height, fps, pixFmt, duration, hasAudio);
    }
}
