import AVFoundation
import CoreGraphics
import XCTest
@testable import TPD

/// `OverlayGeometry` asserted numerically, then a real export round-tripped over a clip
/// generated at test time — no binary is committed for any of this.
final class OverlayRendererTests: XCTestCase {
    private let frameSize = CGSize(width: 640, height: 480)
    private let bgra = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
    /// Deterministic stand-in for the two text systems, so pill placement is asserted against
    /// arithmetic rather than whatever the font metrics happen to be.
    private let measure: (String, CGFloat) -> CGSize =
        { CGSize(width: CGFloat($0.count) * $1 * 0.5, height: $1) }

    /// An identity fit, then the same result in a 320x480 view — uniform scale 0.5 with a
    /// 120 pt letterbox band, so positions have to go through `FrameFit`.
    func testGeometryMapsTheBoxAndOnlyTheVisibleKeypoints() {
        let identity = geometry(OverlayOptions())
        XCTAssertEqual(identity.box, CGRect(x: 100, y: 60, width: 200, height: 300))
        // Four keypoints in, two out: visibility -0.4 and 0 are the backend's `<= 0` skip,
        // without which a channel the heatmap never fired on is drawn at a crop corner.
        XCTAssertEqual(identity.dots.map(\.center),
                       [CGPoint(x: 150, y: 100), CGPoint(x: 220, y: 260)])
        XCTAssertEqual(identity.dots.map(\.radius), [4, 4])
        XCTAssertEqual(identity.pill?.text, "serve   80%")
        let fitted = geometry(OverlayOptions(), in: CGSize(width: 320, height: 480))
        XCTAssertEqual(fitted.box, CGRect(x: 50, y: 150, width: 100, height: 150))
        XCTAssertEqual(fitted.dots.first?.center, CGPoint(x: 75, y: 170))
        // "serve   80%" is 11 characters: 11 * 15 * 0.5 = 82.5 wide plus 20 of padding. A box
        // in the top-right corner clamps the pill inside the frame (640 - 102.5 - 6) and
        // flips it below the box, since 2 - 25 - 6 is off the top.
        let corner = OverlayGeometry(
            result: result(bbox: CGRect(x: 600, y: 2, width: 39, height: 100)),
            options: OverlayOptions(), size: frameSize, measure: measure)
        XCTAssertEqual(corner.pill?.rect, CGRect(x: 531.5, y: 8, width: 102.5, height: 25))
    }

    func testEachToggleSuppressesOnlyItsOwnElement() {
        var options = OverlayOptions()
        options.boundingBox = false
        XCTAssertNil(geometry(options).box)
        // The pill still hangs off the now-invisible box: hiding the outline must not move
        // the reading.
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

    /// One clip, both halves of the exporter's contract: a cancelled run leaves nothing behind,
    /// and a completed run keeps the clip's shape, still decodes, and has the overlay burned in
    /// the right way up — a wrong CTM flip leaves every number above correct, picture wrong.
    func testExportCancelsCleanlyThenRoundTripsTheClip() async throws {
        let source = try await makeClip(frames: 24)
        defer { try? FileManager.default.removeItem(at: source) }
        let burned = result(frameSize: CGSize(width: 320, height: 240),
                            bbox: CGRect(x: 40, y: 30, width: 120, height: 150))
        let partial = FileManager.default.temporaryDirectory
            .appendingPathComponent("tpd-cancelled-\(UUID().uuidString).mp4")
        let cancelled = VideoExporter(configuration: .init(outputURL: partial))
        let task = Task {
            try await cancelled.export(from: source) { _ in
                Thread.sleep(forTimeInterval: 0.02)  // slow enough to be cancelled mid-clip
                return burned
            }
        }
        try await Task.sleep(nanoseconds: 150_000_000)
        task.cancel()
        do {
            _ = try await task.value
            XCTFail("the export finished despite being cancelled")
        } catch {
            XCTAssertTrue(error is CancellationError, "\(error)")
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: partial.path),
                       "a partial file survived cancellation")
        let exporter = VideoExporter(configuration: .init(inferenceCadence: 4))
        let output = try await exporter.export(from: source) { _ in burned }
        defer { try? FileManager.default.removeItem(at: output) }
        XCTAssertTrue(FileManager.default.fileExists(atPath: output.path))
        let asset = AVURLAsset(url: output)
        let duration = try await asset.load(.duration).seconds
        let expected = try await AVURLAsset(url: source).load(.duration).seconds
        XCTAssertEqual(duration, expected, accuracy: 2.0 / 30)
        // Decoding it here is also the "readable by AVAssetReader" assertion.
        let decoded = try await decode(asset)
        XCTAssertEqual(decoded.frames, 24)
        XCTAssertTrue(decoded.overlaid, "no overlay stroke on the exported box's top edge")
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

    /// Counts the frames and reports whether the first one carries the overlay's yellow on
    /// the box's top edge, scanning two rows either side because a 2.5 px stroke and H.264
    /// chroma subsampling both blur it.
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
                    return base[at + 2] > 150 && base[at + 1] > 120 && base[at] < 120
                }
                CVPixelBufferUnlockBaseAddress(buffer, .readOnly)
            }
            frames += 1
        }
        return (frames, overlaid)
    }

    /// A grey 320x240 clip at 30 fps in the temp directory, generated rather than committed
    /// since no binary belongs in this repository.
    private func makeClip(frames: Int) async throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("tpd-source-\(UUID().uuidString).mp4")
        let writer = try AVAssetWriter(url: url, fileType: .mp4)
        let video = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: 320, AVVideoHeightKey: 240])
        video.expectsMediaDataInRealTime = false
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: video,
                                                           sourcePixelBufferAttributes: bgra)
        writer.add(video)
        XCTAssertTrue(writer.startWriting(), "\(String(describing: writer.error))")
        writer.startSession(atSourceTime: .zero)
        for index in 0..<frames {
            // Appending while the input is not ready raises an ObjC exception rather than
            // returning false, so this wait is mandatory, not an optimization.
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
        await writer.finishWriting()
        XCTAssertEqual(writer.status, .completed, "\(String(describing: writer.error))")
        return url
    }
}
