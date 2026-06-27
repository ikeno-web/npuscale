using System.Diagnostics;
using System.Globalization;

namespace NpuScale;

public static class FfmpegPipes
{
    public static Process StartDecoder(string inputPath, int width, int height)
    {
        var args = $"-v error -i \"{inputPath}\" -f rawvideo -pix_fmt rgb24 -s {width}x{height} -";
        var psi = new ProcessStartInfo("ffmpeg", args)
        {
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };
        return Process.Start(psi)
            ?? throw new InvalidOperationException("Failed to start ffmpeg decoder");
    }

    public static Process StartEncoder(string inputPath, string outputPath,
        int outWidth, int outHeight, double fps, bool hasAudio,
        string encoder, int crf)
    {
        var fpsStr = fps.ToString("F6", CultureInfo.InvariantCulture);
        var audioMap = hasAudio ? "-map 1:a? -c:a copy" : "";
        var args =
            $"-v error " +
            $"-f rawvideo -pix_fmt rgb24 -s {outWidth}x{outHeight} -r {fpsStr} -i - " +
            $"-i \"{inputPath}\" " +
            $"-map 0:v {audioMap} " +
            $"-c:v {encoder} -crf {crf} -y \"{outputPath}\"";

        var psi = new ProcessStartInfo("ffmpeg", args)
        {
            RedirectStandardInput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };
        return Process.Start(psi)
            ?? throw new InvalidOperationException("Failed to start ffmpeg encoder");
    }

    // Reads exactly buffer.Length bytes from stream; returns false on EOF.
    public static bool ReadFrame(Stream stream, byte[] buffer)
    {
        int offset = 0;
        while (offset < buffer.Length)
        {
            int read = stream.Read(buffer, offset, buffer.Length - offset);
            if (read == 0) return false;
            offset += read;
        }
        return true;
    }
}
