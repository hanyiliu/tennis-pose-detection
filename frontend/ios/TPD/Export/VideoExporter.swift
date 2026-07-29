//  VideoExporter.swift
//  Burns the overlay into a copy of a clip: AVAssetReader -> inference -> OverlayRenderer ->
//  AVAssetWriter. The engine, not the feature: the caller owns the picker, the progress UI and
//  the PHPhotoLibrary save, and gets a finished file in the temp directory. Video only — the
//  audio pass-through was cut for line budget and is the follow-up's first job.

import AVFoundation
import CoreGraphics
import Foundation

/// Serializes the non-`Sendable` engine so one instance can cross into an export off the caller's
/// actor — what `TPDInferenceEngine` asks for; one task calls it, so the lock never contends.
final class SerializedInferenceEngine: @unchecked Sendable {
    private let lock = NSLock()
    private let engine: TPDInferenceEngine
    init(_ engine: TPDInferenceEngine) { self.engine = engine }
    func predict(_ frame: VideoFrame) throws -> TPDResult {
        lock.lock()
        defer { lock.unlock() }
        return try engine.predict(pixelBuffer: frame.pixelBuffer)
    }
}

enum VideoExportError: Error, Equatable, LocalizedError {
    case noVideoTrack(URL), failed(String)
    var errorDescription: String? {
        switch self {
        case .noVideoTrack(let url): return "\(url.lastPathComponent) has no video track."
        case .failed(let why): return "The overlaid export failed: \(why)"
        }
    }
}

/// `@unchecked` covers `observers` alone — every read and write of it goes through `lock`.
final class VideoExporter: @unchecked Sendable {
    struct Configuration: Sendable {
        /// The four switches `ToggleBar` drives, burned in permanently.
        var overlay = OverlayOptions()
        /// Run the model on every Nth frame and reuse its result in between; 1 is every frame.
        /// **Default 3.** Inference dominates absolutely — ~9.7 s for one 608x1080 frame on the
        /// booted simulator against ~10 ms to draw — so this dial *is* the export time, at
        /// little cost: at 30 fps, 3 still re-reads inside the ~150 ms a stroke phase lasts.
        var inferenceCadence = 3
        /// Defaults to a fresh .mp4 in `FileManager.temporaryDirectory`.
        var outputURL: URL?
    }

    /// 0...1, newest-wins. Every access vends a **fresh** stream, which follows the run in flight
    /// — or the next to start — and is finished when that run ends however it ends, an early
    /// throw included: a progress UI awaiting one never hangs, and the instance stays reusable.
    /// Read it *before* calling `export`, since the buffer holds one value. Runs are sequential:
    /// two overlapping on one instance would share, and cut short, a single set of streams.
    var progress: AsyncStream<Double> {
        AsyncStream(bufferingPolicy: .bufferingNewest(1)) { continuation in
            self.lock.withLock { self.observers.append(continuation) }
        }
    }
    private let lock = NSLock()
    private var observers: [AsyncStream<Double>.Continuation] = []
    private let configuration: Configuration
    private static let bgra = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]

    init(configuration: Configuration = Configuration()) { self.configuration = configuration }
    deinit { endProgress() }  // covers a caller that reads `progress` and never exports

    func export(from source: URL, using engine: SerializedInferenceEngine) async throws -> URL {
        try await export(from: source, analyze: { try engine.predict($0) })
    }

    /// Runs on the caller's task, so cancelling that task cancels this: reader and writer
    /// are torn down and the half-written file deleted before `CancellationError` escapes.
    func export(from source: URL,
                analyze: @Sendable (VideoFrame) throws -> TPDResult) async throws -> URL {
        // The one place a run's streams end, so no exit can skip it: several of the failures
        // below throw above the do/catch, and each used to strand every consumer for good.
        defer { endProgress() }
        let asset = AVURLAsset(url: source)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw VideoExportError.noVideoTrack(source)
        }
        // The device `VideoFileFrameSource` uses, for the same reason: a video composition bakes
        // the track's `preferredTransform` into the decoded buffers, where a raw track output
        // hands back the *encoded* frame — sideways for any iPhone portrait recording.
        let composition = try await AVVideoComposition.videoComposition(withPropertiesOf: asset)
        let size = composition.renderSize
        let width = Int(size.width.rounded()), height = Int(size.height.rounded())
        guard width > 0, height > 0 else { throw VideoExportError.failed("empty render size") }
        let duration = try await asset.load(.duration).seconds
        let reader = try AVAssetReader(asset: asset)
        let videoOutput = AVAssetReaderVideoCompositionOutput(videoTracks: [track],
                                                              videoSettings: Self.bgra)
        videoOutput.videoComposition = composition
        guard reader.canAdd(videoOutput) else { throw VideoExportError.failed("output refused") }
        reader.add(videoOutput)
        let outputURL = configuration.outputURL ?? FileManager.default.temporaryDirectory
            .appendingPathComponent("TPD-overlay-\(UUID().uuidString).mp4")
        try? FileManager.default.removeItem(at: outputURL)
        let writer = try AVAssetWriter(url: outputURL, fileType: .mp4)
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: width,
            AVVideoHeightKey: height])
        input.expectsMediaDataInRealTime = false
        // No `sourcePixelBufferAttributes`, so no pool: a pool would add a copy per frame.
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input,
                                                           sourcePixelBufferAttributes: nil)
        guard writer.canAdd(input) else { throw VideoExportError.failed("input refused") }
        writer.add(input)
        let style = OverlayStyle.fitting(width: size.width)
        do {
            guard writer.startWriting() else { throw Self.failure(writer.error) }
            writer.startSession(atSourceTime: .zero)
            guard reader.startReading() else { throw Self.failure(reader.error) }
            var index = 0, cached: TPDResult?
            let cadence = max(1, configuration.inferenceCadence)
            while let sample = videoOutput.copyNextSampleBuffer() {
                try Task.checkCancellation()
                guard let frame = CMSampleBufferGetImageBuffer(sample) else { continue }
                let time = CMSampleBufferGetPresentationTimeStamp(sample)
                if index % cadence == 0 || cached == nil {
                    cached = try analyze(VideoFrame(pixelBuffer: frame, time: time))
                }
                if let cached { try Self.burn(cached, into: frame, options: configuration.overlay,
                                              style: style, size: size) }
                // Appending while not ready raises an ObjC exception rather than returning false.
                while !input.isReadyForMoreMediaData { try await Task.sleep(nanoseconds: 2_000_000) }
                guard adaptor.append(frame, withPresentationTime: time)
                else { throw Self.failure(writer.error) }
                index += 1
                if duration > 0 { report(min(time.seconds / duration, 1)) }
            }
            guard reader.status != .failed else { throw Self.failure(reader.error) }
            input.markAsFinished()
            await writer.finishWriting()
            guard writer.status == .completed else { throw Self.failure(writer.error) }
        } catch {
            reader.cancelReading()
            writer.cancelWriting()  // deletes the partial file; the remove covers an earlier fail
            try? FileManager.default.removeItem(at: outputURL)
            throw error
        }
        report(1)
        return outputURL
    }

    private func report(_ fraction: Double) {
        lock.withLock { observers }.forEach { $0.yield(fraction) }
    }
    /// Ends every stream vended for the run, unlocked: a termination handler runs inline here.
    private func endProgress() {
        let vended = lock.withLock { let all = observers; observers = []; return all }
        vended.forEach { $0.finish() }
    }

    /// Draws straight into the decoded frame: `copyNextSampleBuffer` hands over a buffer nothing
    /// else holds, so mutating it in place is safe.
    private static func burn(_ result: TPDResult, into buffer: CVPixelBuffer,
                             options: OverlayOptions, style: OverlayStyle, size: CGSize) throws {
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        let rows = CVPixelBufferGetHeight(buffer)
        guard let base = CVPixelBufferGetBaseAddress(buffer), let context = CGContext(
            data: base, width: CVPixelBufferGetWidth(buffer), height: rows, bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer), space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
                | CGBitmapInfo.byteOrder32Little.rawValue) else {
            throw VideoExportError.failed("could not draw into a decoded frame")
        }
        // CoreGraphics' y grows upward; `TPDResult` and `FrameFit` are top-left-origin.
        context.translateBy(x: 0, y: CGFloat(rows))
        context.scaleBy(x: 1, y: -1)
        OverlayRenderer.draw(result, options: options, in: context, size: size, style: style)
    }

    private static func failure(_ error: Error?) -> VideoExportError {
        .failed(error?.localizedDescription ?? "no further detail")
    }
}
