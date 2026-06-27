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

        // Temporal models (N consecutive frames per inference) take a dedicated
        // sliding-window path; single-frame models use the concurrent pipeline.
        if (processor.InputFrameCount > 1)
        {
            await RunTemporalAsync(info);
            return;
        }

        int frameSize = info.Width * info.Height * 3;
        long totalFrames = info.Duration > 0 ? (long)(info.Duration * info.Fps) : 0;

        // A single worker runs inferences sequentially, so bound-buffer reuse is safe.
        if (workers <= 1) processor.ReuseBuffers = true;

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

    // Temporal (multi-frame) path. The model denoises the CENTRE frame of a window
    // of N = processor.InputFrameCount consecutive frames. Windows overlap, so this
    // runs sequentially with a small sliding buffer; edge frames are replicated.
    private async Task RunTemporalAsync(MediaInfo info)
    {
        int window    = processor.InputFrameCount;
        int half      = window / 2;
        int frameSize = info.Width * info.Height * 3;
        long total    = info.Duration > 0 ? (long)(info.Duration * info.Fps) : 0;

        processor.ReuseBuffers = true;     // temporal path is single-threaded

        using var decoder = FfmpegPipes.StartDecoder(inputPath, info.Width, info.Height);
        var dstream = decoder.StandardOutput.BaseStream;

        // Sliding buffer of contiguous frames; buffer[0] is frame index `bufStart`.
        var buffer   = new List<byte[]>();
        long bufStart = 0, read = 0;
        bool eof = false;
        var tmp = new byte[frameSize];

        void ReadUpTo(long idx)                       // read until frame `idx` is buffered or EOF
        {
            while (!eof && read <= idx)
            {
                if (!FfmpegPipes.ReadFrame(dstream, tmp)) { eof = true; break; }
                var copy = new byte[frameSize];
                Buffer.BlockCopy(tmp, 0, copy, 0, frameSize);
                buffer.Add(copy);
                read++;
            }
        }

        byte[][] Window(long centre)                  // N frames around `centre`, edges replicated
        {
            var w = new byte[window][];
            for (int k = -half; k <= half; k++)
            {
                long idx = Math.Clamp(centre + k, 0, read - 1);
                w[k + half] = buffer[(int)(idx - bufStart)];
            }
            return w;
        }

        ReadUpTo(half);                               // enough to build the centre-0 window
        if (read == 0) throw new InvalidOperationException("Input video has no frames");

        var firstOut = processor.ProcessWindow(Window(0), info.Width, info.Height);
        int outW = processor.OutputWidth, outH = processor.OutputHeight;
        if (verbose)
            Console.Error.WriteLine(
                $"Output: {outW}x{outH} (temporal, {window}-frame window, " +
                $"{outW / (double)info.Width:F2}x)");

        using var enc = FfmpegPipes.StartEncoder(inputPath, outputPath,
            outW, outH, info.Fps, info.HasAudio, encoder, crf);
        var estream = enc.StandardInput.BaseStream;
        var sw = Stopwatch.StartNew();

        estream.Write(firstOut, 0, firstOut.Length);
        long processed = 1;

        for (long centre = 1; ; centre++)
        {
            ReadUpTo(centre + half);
            if (eof && centre >= read) break;         // all centres emitted

            var outFrame = processor.ProcessWindow(Window(centre), info.Width, info.Height);
            estream.Write(outFrame, 0, outFrame.Length);
            processed++;

            if (verbose && processed % 30 == 0)
            {
                double fps = processed / sw.Elapsed.TotalSeconds;
                string pct = total > 0 ? $" ({100.0 * processed / total:F1}%)" : "";
                Console.Error.Write($"\r{processed} frames{pct}, {fps:F1} fps   ");
            }

            // Drop frames no longer needed by any future window.
            while (bufStart < centre + 1 - half) { buffer.RemoveAt(0); bufStart++; }
        }

        estream.Close();
        await enc.WaitForExitAsync();
        decoder.WaitForExit();

        if (verbose)
        {
            Console.Error.WriteLine();
            Console.Error.WriteLine(
                $"Done: {processed} frames in {sw.Elapsed.TotalSeconds:F1}s " +
                $"({processed / sw.Elapsed.TotalSeconds:F1} fps)");
        }
        if (enc.ExitCode != 0)
            Console.Error.WriteLine($"Warning: encoder exited with code {enc.ExitCode}");
    }
}
