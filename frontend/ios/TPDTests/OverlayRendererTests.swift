import AVFoundation
import CoreGraphics
import XCTest
@testable import TPD

/// `OverlayGeometry` asserted numerically, then real exports round-tripped over a generated clip.
final class OverlayRendererTests: XCTestCase {
    private let frameSize = CGSize(width: 640, height: 480)
    private let bgra = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
    /// Deterministic stand-in for the two text systems: placement asserted against arithmetic.
    private let measure: (String, CGFloat) -> CGSize =
        { CGSize(width: CGFloat($0.count) * $1 * 0.5, height: $1) }

    /// An identity fit, then a 320x480 view: scale 0.5 plus a 120 pt letterbox band, so every
    /// position has to go through `FrameFit`.
    func testGeometryMapsTheBoxAndOnlyTheVisibleKeypoints() {
        let identity = geometry(OverlayOptions())
        XCTAssertEqual(identity.box, CGRect(x: 100, y: 60, width: 200, height: 300))
        // Four keypoints in, two out: the backend's `visibility <= 0` skip, without which a
        // channel the heatmap never fired on lands at a crop corner.
        XCTAssertEqual(identity.dots.map(\.center),
                       [CGPoint(x: 150, y: 100), CGPoint(x: 220, y: 260)])
        XCTAssertEqual(identity.dots.map(\.radius), [4, 4])
        XCTAssertEqual(identity.pill?.text, "serve   80%")
        let fitted = geometry(OverlayOptions(), in: CGSize(width: 320, height: 480))
        XCTAssertEqual(fitted.box, CGRect(x: 50, y: 150, width: 100, height: 150))
        XCTAssertEqual(fitted.dots.first?.center, CGPoint(x: 75, y: 170))
        // "serve   80%" is 11 characters: 11 * 15 * 0.5 = 82.5 wide plus 20 of padding. A box in
        // the top-right clamps the pill inside the frame (640 - 102.5 - 6) and, since 2 - 25 - 6
        // is off the top, flips it below the box.
        let corner = OverlayGeometry(
            result: result(bbox: CGRect(x: 600, y: 2, width: 39, height: 100)),
            options: OverlayOptions(), size: frameSize, measure: measure)
        XCTAssertEqual(corner.pill?.rect, CGRect(x: 531.5, y: 8, width: 102.5, height: 25))
    }

    func testEachToggleSuppressesOnlyItsOwnElement() {
        var options = OverlayOptions()
        options.boundingBox = false
        XCTAssertNil(geometry(options).box)
        // The pill still hangs off the now-invisible box: hiding the outline moves no reading.
        XCTAssertEqual(geometry(options).pill?.rect, geometry(OverlayOptions()).pill?.rect)
        options = OverlayOptions(); options.keypoints = false
        XCTAssertTrue(geometry(options).dots.isEmpty)
        options = OverlayOptions(); options.confidence = false
        XCTAssertEqual(geometry(options).pill?.text, "serve")
        options = OverlayOptions(); options.label = false
        XCTAssertEqual(geometry(options).pill?.text, "80%")
        options.confidence = false
        XCTAssertNil(geometry(options).pill)
    }

    /// One clip, the exporter's whole contract: a cancelled run leaves nothing behind; the same
    /// instance then exports twice, the second run getting a live progress stream of its own and
    /// not the finished one the first left behind, which would replay a stale 1.0 and stop while
    /// the run it tracks was still going; and that second file keeps the clip's shape, decodes,
    /// and has the overlay the right way up — a wrong CTM flip leaves the numbers above correct.
    func testExportCancelsCleanlyThenRoundTripsTheClipTwiceOnOneInstance() async throws {
        let source = try await makeClip(frames: 24)
        defer { try? FileManager.default.removeItem(at: source) }
        let burned = result(frameSize: CGSize(width: 320, height: 240),
                            bbox: CGRect(x: 40, y: 30, width: 120, height: 150))
        let partial = FileManager.default.temporaryDirectory
            .appendingPathComponent("tpd-partial-\(UUID().uuidString).mp4")
        let cancelled = VideoExporter(configuration: .init(outputURL: partial))
        let task = Task {
            try await cancelled.export(from: source) { _ in
                Thread.sleep(forTimeInterval: 0.02); return burned  // slow: cancelled mid-clip
            }
        }
        try await Task.sleep(nanoseconds: 150_000_000)
        task.cancel()
        do { _ = try await task.value; XCTFail("the export finished despite being cancelled") }
        catch { XCTAssertTrue(error is CancellationError, "\(error)") }
        XCTAssertFalse(FileManager.default.fileExists(atPath: partial.path),
                       "a partial file survived cancellation")
        let exporter = VideoExporter(configuration: .init(inferenceCadence: 4))
        let first = try await exporter.export(from: source) { _ in burned }
        defer { try? FileManager.default.removeItem(at: first) }
        // The reuse assertion: more than one value, which a stream replaying one buffered
        // fraction from the finished first run cannot produce, and then a clean end.
        let tracked = expectation(description: "the second run's stream followed it, then ended")
        let progress = exporter.progress
        Task {
            var seen = 0
            for await _ in progress { seen += 1 }
            if seen > 1 { tracked.fulfill() }
        }
        let output = try await exporter.export(from: source) { _ in
            Thread.sleep(forTimeInterval: 0.02); return burned  // slow enough to be watched live
        }
        defer { try? FileManager.default.removeItem(at: output) }
        await fulfillment(of: [tracked], timeout: 60)
        let made = try await AVURLAsset(url: output).load(.duration).seconds
        let want = try await AVURLAsset(url: source).load(.duration).seconds
        XCTAssertEqual(made, want, accuracy: 2.0 / 30)
        // Decoding it here is also the "readable by AVAssetReader" assertion.
        let decoded = try await decode(AVURLAsset(url: output))
        XCTAssertEqual(decoded.frames, 24)
        XCTAssertTrue(decoded.overlaid, "no overlay stroke on the exported box's top edge")
    }

    /// The audio pass-through. The fixture's track is genuine AAC — `AVAssetWriter` encodes it
    /// from LPCM silence on the way in — so this exercises the compressed path the exporter muxes
    /// across, not a convenient uncompressed one. Duration and track count alone would also pass
    /// for audio that had been decoded and re-encoded, hence the format assertion.
    func testExportMuxesTheClipsAudioTrackThroughUntouched() async throws {
        let source = try await makeClip(frames: 24, audio: true)
        defer { try? FileManager.default.removeItem(at: source) }
        let sourceTracks = try await AVURLAsset(url: source).loadTracks(withMediaType: .audio)
        let want = try await XCTUnwrap(sourceTracks.first).load(.timeRange).duration.seconds
        let exporter = VideoExporter(configuration: .init(inferenceCadence: 24))
        let output = try await exporter.export(from: source) { [stub = result()] _ in stub }
        defer { try? FileManager.default.removeItem(at: output) }
        let tracks = try await AVURLAsset(url: output).loadTracks(withMediaType: .audio)
        XCTAssertEqual(tracks.count, 1, "the export did not carry exactly one audio track")
        let track = try XCTUnwrap(tracks.first)
        let made = try await track.load(.timeRange).duration.seconds
        XCTAssertEqual(made, want, accuracy: 0.05)
        let formats = try await track.load(.formatDescriptions)
        let format = try XCTUnwrap(formats.first)
        XCTAssertEqual(format.mediaSubType, .mpeg4AAC, "the audio was re-encoded, not muxed")
        // And the picture survived the second track: audio did not displace any video.
        let decoded = try await decode(AVURLAsset(url: output))
        XCTAssertEqual(decoded.frames, 24)
    }

    /// The stream ends however the export ends, and the failures that throw before the writer
    /// exists are the ones that used to skip that, hanging a progress UI for good. The timeout
    /// keeps a regression a failure rather than a wedged suite.
    func testAFailureBeforeTheFirstFrameStillFinishesTheProgressStream() async throws {
        let exporter = VideoExporter()
        let ended = expectation(description: "the progress stream finished with the failure")
        let progress = exporter.progress
        Task { for await _ in progress { }; ended.fulfill() }
        let missing = FileManager.default.temporaryDirectory.appendingPathComponent("tpd-none.mp4")
        do { _ = try await exporter.export(from: missing) { [stub = result()] _ in stub }
             XCTFail("a URL with no video track exported") } catch {}
        await fulfillment(of: [ended], timeout: 30)
    }

    // MARK: - Fixtures

    private func result(frameSize: CGSize? = nil,
                        bbox: CGRect = CGRect(x: 100, y: 60, width: 200, height: 300)) -> TPDResult {
        TPDResult(frameSize: frameSize ?? self.frameSize, bbox: bbox,
                  keypoints: [(CGPoint(x: 150, y: 100), 0.9), (CGPoint(x: 220, y: 260), 0.2),
                              (CGPoint(x: 100, y: 60), -0.4), (CGPoint(x: 110, y: 70), 0)]
                      .map { TPDKeypoint(position: $0.0, visibility: $0.1) },
                  probabilities: [0.05, 0.1, 0.8, 0.05],
                  labels: ["forehand", "backhand", "serve", "ready"], bestIndex: 2)
    }

    private func geometry(_ options: OverlayOptions, in size: CGSize? = nil) -> OverlayGeometry {
        OverlayGeometry(result: result(), options: options, size: size ?? frameSize, measure: measure)
    }

    /// Counts the frames and reports whether the first carries the overlay's yellow on the box's
    /// top edge, scanning two rows either side: a 2.5 px stroke and H.264 chroma both blur it.
    private func decode(_ asset: AVAsset) async throws -> (frames: Int, overlaid: Bool) {
        let tracks = try await asset.loadTracks(withMediaType: .video)
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: try XCTUnwrap(tracks.first),
                                              outputSettings: bgra)
        reader.add(output)
        XCTAssertTrue(reader.startReading())
        var frames = 0, overlaid = false
        while let sample = output.copyNextSampleBuffer() {
            if frames == 0, let buffer = CMSampleBufferGetImageBuffer(sample) {
                CVPixelBufferLockBaseAddress(buffer, .readOnly)
                let base = try XCTUnwrap(CVPixelBufferGetBaseAddress(buffer))
                    .assumingMemoryBound(to: UInt8.self)
                let stride = CVPixelBufferGetBytesPerRow(buffer)
                overlaid = (28...32).contains { row in  // BGRA, the box's top edge is y = 30
                    let at = row * stride + 100 * 4
                    return base[at + 2] > 150 && base[at + 1] > 120 && base[at] < 120 }
                CVPixelBufferUnlockBaseAddress(buffer, .readOnly)
            }
            frames += 1
        }
        return (frames, overlaid)
    }

    /// One buffer of 16-bit LPCM silence. The writer input it is handed is configured for AAC, so
    /// AVFoundation encodes it, and the fixture ends up with a compressed track like a real clip's.
    private func silence(seconds: Double, rate: Double = 44_100) throws -> CMSampleBuffer {
        var asbd = AudioStreamBasicDescription(
            mSampleRate: rate, mFormatID: kAudioFormatLinearPCM,
            mFormatFlags: kAudioFormatFlagIsSignedInteger | kAudioFormatFlagIsPacked,
            mBytesPerPacket: 2, mFramesPerPacket: 1, mBytesPerFrame: 2, mChannelsPerFrame: 1,
            mBitsPerChannel: 16, mReserved: 0)
        var format: CMAudioFormatDescription?
        // Every status is checked: a half-built sample buffer does not fail an assertion later, it
        // takes an ObjC exception out of `append` and the whole test host with it.
        XCTAssertEqual(CMAudioFormatDescriptionCreate(
            allocator: nil, asbd: &asbd, layoutSize: 0, layout: nil, magicCookieSize: 0,
            magicCookie: nil, extensions: nil, formatDescriptionOut: &format), noErr)
        let samples = Int(rate * seconds), bytes = samples * 2
        var block: CMBlockBuffer?
        // `AssureMemoryNow`: with a null memory block the allocation is otherwise deferred, and a
        // sample buffer over unallocated storage is exactly the kind that takes the host down.
        XCTAssertEqual(CMBlockBufferCreateWithMemoryBlock(
            allocator: nil, memoryBlock: nil, blockLength: bytes, blockAllocator: nil,
            customBlockSource: nil, offsetToData: 0, dataLength: bytes,
            flags: kCMBlockBufferAssureMemoryNowFlag, blockBufferOut: &block), noErr)
        XCTAssertEqual(CMBlockBufferFillDataBytes(with: 0, blockBuffer: try XCTUnwrap(block),
                                                  offsetIntoDestination: 0, dataLength: bytes),
                       noErr)
        var timing = CMSampleTimingInfo(duration: CMTime(value: 1, timescale: CMTimeScale(rate)),
                                        presentationTimeStamp: .zero, decodeTimeStamp: .invalid)
        var size = 2, buffer: CMSampleBuffer?
        XCTAssertEqual(CMSampleBufferCreate(
            allocator: nil, dataBuffer: block, dataReady: true, makeDataReadyCallback: nil,
            refcon: nil, formatDescription: format, sampleCount: samples,
            sampleTimingEntryCount: 1, sampleTimingArray: &timing, sampleSizeEntryCount: 1,
            sampleSizeArray: &size, sampleBufferOut: &buffer), noErr)
        return try XCTUnwrap(buffer)
    }

    /// A grey 320x240 clip at 30 fps in the temp directory, generated rather than committed.
    private func makeClip(frames: Int, audio: Bool = false) async throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("tpd-source-\(UUID().uuidString).mp4")
        let writer = try AVAssetWriter(url: url, fileType: .mp4)
        let video = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: 320, AVVideoHeightKey: 240])
        video.expectsMediaDataInRealTime = false
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: video,
                                                           sourcePixelBufferAttributes: bgra)
        writer.add(video)
        var sound: AVAssetWriterInput?
        if audio {
            let input = AVAssetWriterInput(mediaType: .audio, outputSettings: [
                AVFormatIDKey: kAudioFormatMPEG4AAC, AVSampleRateKey: 44_100,
                AVNumberOfChannelsKey: 1, AVEncoderBitRateKey: 64_000])
            input.expectsMediaDataInRealTime = false
            XCTAssertTrue(writer.canAdd(input))  // `add` raises rather than returning on a refusal
            writer.add(input)
            sound = input
        }
        XCTAssertTrue(writer.startWriting(), "\(String(describing: writer.error))")
        writer.startSession(atSourceTime: .zero)
        for index in 0..<frames {
            // Appending while the input is not ready raises an ObjC exception, not a false.
            while !video.isReadyForMoreMediaData { try await Task.sleep(nanoseconds: 2_000_000) }
            var buffer: CVPixelBuffer?
            CVPixelBufferPoolCreatePixelBuffer(nil, try XCTUnwrap(adaptor.pixelBufferPool), &buffer)
            let pixels = try XCTUnwrap(buffer)
            CVPixelBufferLockBaseAddress(pixels, [])
            memset(CVPixelBufferGetBaseAddress(pixels), 40 + Int32(index),
                   CVPixelBufferGetBytesPerRow(pixels) * 240)
            CVPixelBufferUnlockBaseAddress(pixels, [])
            XCTAssertTrue(adaptor.append(pixels, withPresentationTime:
                CMTime(value: CMTimeValue(index), timescale: 30)))
        }
        video.markAsFinished()
        if let sound {
            while !sound.isReadyForMoreMediaData { try await Task.sleep(nanoseconds: 2_000_000) }
            XCTAssertTrue(sound.append(try silence(seconds: Double(frames) / 30)))
            sound.markAsFinished()
        }
        await writer.finishWriting()
        XCTAssertEqual(writer.status, .completed, "\(String(describing: writer.error))")
        return url
    }
}
