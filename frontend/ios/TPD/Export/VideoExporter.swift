//  VideoExporter.swift
//  Burns the overlay into a copy of a clip: AVAssetReader -> inference -> OverlayRenderer ->
//  AVAssetWriter. The engine, not the feature: the caller owns the picker, the progress UI and
//  the PHPhotoLibrary save, and gets a finished file in the temp directory.
//
//  SCOPE: video only. Copying the audio track through untouched (a second AVAssetReader,
//  `outputSettings: nil` both ends, drained after the video pass) was written and tested, then
//  cut for this PR's line budget — the follow-up should add it back first.

import AVFoundation
import CoreGraphics
import Foundation

/// Serializes the non-`Sendable` engine so one instance can cross into an export running off
/// the caller's actor — `TPDInferenceEngine` asks for exactly this, one instance with calls
/// serialized, and the export calls it from a single task, so the lock never contends.
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

final class VideoExporter: Sendable {
    struct Configuration: Sendable {
        /// The four switches `ToggleBar` drives, burned in permanently.
        var overlay = OverlayOptions()
        /// Run the model on every Nth frame and reuse its result in between; 1 is every
        /// frame. **Default 3.** Inference dominates absolutely — on the booted simulator
        /// (Debug, x86_64 host) one pass over a 608x1080 frame costs ~9.7 s against ~10 ms for
        /// the draw — so this dial *is* the export time, and costs little accuracy: at 30 fps,
        /// 3 still re-reads every 100 ms, inside the ~150 ms a stroke phase lasts.
        var inferenceCadence = 3
        /// Defaults to a fresh .mp4 in `FileManager.temporaryDirectory`.
        var outputURL: URL?
    }

    /// 0...1, newest-wins, finished when the export ends however it ends. Iterate it *before*
    /// calling `export`: the buffer holds one value, so a late consumer sees the current
    /// fraction but none of the earlier ones.
    let progress: AsyncStream<Double>
    private let continuation: AsyncStream<Double>.Continuation
    private let configuration: Configuration
    private static let bgra = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]

    init(configuration: Configuration = Configuration()) {
        self.configuration = configuration
        let made = AsyncStream.makeStream(of: Double.self, bufferingPolicy: .bufferingNewest(1))
        (progress, continuation) = (made.stream, made.continuation)
    }
    deinit { continuation.finish() }

    func export(from source: URL, using engine: SerializedInferenceEngine) async throws -> URL {
        try await export(from: source, analyze: { try engine.predict($0) })
    }

    /// Runs on the caller's task, so cancelling that task cancels this: reader and writer
    /// are torn down and the half-written file deleted before `CancellationError` escapes.
    func export(from source: URL,
                analyze: @Sendable (VideoFrame) throws -> TPDResult) async throws -> URL {
        let asset = AVURLAsset(url: source)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw VideoExportError.noVideoTrack(source)
        }
        // The device `VideoFileFrameSource` uses, for the same reason: a video composition
        // bakes the track's `preferredTransform` into the decoded buffers. A raw track output
        // hands back the *encoded* frame — sideways for any iPhone portrait recording — so
        // the model would read a rotated player and the overlay would land on a rotated
        // picture. Both now see one upright frame; the writer needs no transform of its own.
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
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: width, AVVideoHeightKey: height])
        input.expectsMediaDataInRealTime = false
        // No `sourcePixelBufferAttributes`, so no pool: every buffer appended is one the
        // reader decoded and we drew on in place; a pool would only add a copy per frame.
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
                if let cached {
                    try Self.burn(cached, into: frame, options: configuration.overlay,
                                  style: style, size: size)
                }
                // Appending while the input is not ready raises an ObjC exception rather
                // than returning false, so this wait is mandatory; it doubles as the
                // cancellation check for a writer that has stalled.
                while !input.isReadyForMoreMediaData {
                    try await Task.sleep(nanoseconds: 2_000_000)
                }
                guard adaptor.append(frame, withPresentationTime: time) else {
                    throw Self.failure(writer.error)
                }
                index += 1
                if duration > 0 { continuation.yield(min(time.seconds / duration, 1)) }
            }
            guard reader.status != .failed else { throw Self.failure(reader.error) }
            input.markAsFinished()
            await writer.finishWriting()
            guard writer.status == .completed else { throw Self.failure(writer.error) }
        } catch {
            reader.cancelReading()
            // `cancelWriting` deletes the partial file; the remove covers a writer that
            // failed before it ever owned one.
            writer.cancelWriting()
            try? FileManager.default.removeItem(at: outputURL)
            continuation.finish()
            throw error
        }
        continuation.yield(1)
        continuation.finish()
        return outputURL
    }

    /// Draws straight into the decoded frame: `copyNextSampleBuffer` hands over a buffer
    /// nothing else holds, so mutating it in place is safe and saves a copy per frame.
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
