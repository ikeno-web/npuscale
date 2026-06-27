using System.Collections.Concurrent;
using System.Diagnostics;

namespace NpuScale;

public sealed class Pipeline(
    string inputPath,
    string outputPath,
    OnnxProcessor processor,
    int workers,
    string encoder,
    int crf,
    bool verbose)
{
    public async Task RunAsync()
    {
        var info = MediaInfo.Probe(inputPath);
        if (verbose)
            Console.Error.WriteLine(
                $"Input:  {info.Width}x{info.Height} @ {info.Fps:F2} fps, " +
                $"duration={info.Duration:F1}s, audio={info.HasAudio}");

        int frameSize = info.Width * info.Height * 3;
        long totalFrames = info.Duration > 0 ? (long)(info.Duration * info.Fps) : 0;

        // Run first frame to discover output dimensions before starting the encoder.
        var firstFrame = new byte[frameSize];
        using (var probeDecoder = FfmpegPipes.StartDecoder(inputPath, info.Width, info.Height))
        {
            if (!FfmpegPipes.ReadFrame(probeDecoder.StandardOutput.BaseStream, firstFrame))
                throw new InvalidOperationException("Input video has no frames");
            probeDecoder.Kill();
            probeDecoder.WaitForExit();
        }
        var firstOut = processor.Process(firstFrame, info.Width, info.Height);
        int outW = processor.OutputWidth;
        int outH = processor.OutputHeight;

        if (verbose)
            Console.Error.WriteLine(
                $"Output: {outW}x{outH} (scale {outW / (double)info.Width:F2}x)");

        // Bounded input queue — caps memory to ~workers*2 frames in flight.
        var inputQueue  = new BlockingCollection<(long Index, byte[] Data)>(workers * 2);
        var outputDict  = new ConcurrentDictionary<long, byte[]>();
        var writerReady = new ManualResetEventSlim(false);
        var sw          = Stopwatch.StartNew();
        long processed  = 0;

        // ── Decoder ──────────────────────────────────────────────────────────
        using var decoder = FfmpegPipes.StartDecoder(inputPath, info.Width, info.Height);
        using var enc     = FfmpegPipes.StartEncoder(inputPath, outputPath,
                                outW, outH, info.Fps, info.HasAudio, encoder, crf);

        var decodeTask = Task.Run(() =>
        {
            long idx = 0;
            var buf  = new byte[frameSize];
            while (FfmpegPipes.ReadFrame(decoder.StandardOutput.BaseStream, buf))
            {
                var copy = new byte[frameSize];
                Buffer.BlockCopy(buf, 0, copy, 0, frameSize);
                inputQueue.Add((idx++, copy));
            }
            inputQueue.CompleteAdding();
        });

        // ── Inference workers ─────────────────────────────────────────────────
        var workerTasks = Enumerable.Range(0, workers).Select(_ => Task.Run(() =>
        {
            foreach (var (index, data) in inputQueue.GetConsumingEnumerable())
            {
                if (index == 0) continue;           // frame 0 already processed
                outputDict[index] = processor.Process(data, info.Width, info.Height);
                writerReady.Set();
            }
        })).ToArray();

        // ── Ordered writer ────────────────────────────────────────────────────
        var writeTask = Task.Run(() =>
        {
            var stream    = enc.StandardInput.BaseStream;
            long nextIdx  = 0;

            // Emit frame 0 (already processed before encoder started).
            stream.Write(firstOut, 0, firstOut.Length);
            nextIdx = 1;
            Interlocked.Increment(ref processed);

            while (true)
            {
                if (outputDict.TryRemove(nextIdx, out var frame))
                {
                    stream.Write(frame, 0, frame.Length);
                    nextIdx++;
                    long n = Interlocked.Increment(ref processed);

                    if (verbose && n % 30 == 0)
                    {
                        double fps  = n / sw.Elapsed.TotalSeconds;
                        string pct  = totalFrames > 0
                            ? $" ({100.0 * n / totalFrames:F1}%)" : "";
                        Console.Error.Write($"\r{n} frames{pct}, {fps:F1} fps   ");
                    }
                }
                else if (inputQueue.IsCompleted && !outputDict.ContainsKey(nextIdx))
                {
                    break;
                }
                else
                {
                    writerReady.Wait(10);
                    writerReady.Reset();
                }
            }
            stream.Close();
        });

        await decodeTask;
        await Task.WhenAll(workerTasks);
        writerReady.Set();          // unblock writer if it's sleeping on last frames
        await writeTask;

        decoder.WaitForExit();
        enc.WaitForExit();

        if (verbose)
        {
            Console.Error.WriteLine();
            long n = Interlocked.Read(ref processed);
            Console.Error.WriteLine(
                $"Done: {n} frames in {sw.Elapsed.TotalSeconds:F1}s " +
                $"({n / sw.Elapsed.TotalSeconds:F1} fps)");
        }

        if (enc.ExitCode != 0)
            Console.Error.WriteLine($"Warning: encoder exited with code {enc.ExitCode}");
    }
}
