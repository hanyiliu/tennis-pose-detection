//  FrameSource.swift
//  The one contract every frame producer implements.
//
//  Two implementations exist: `CameraFrameSource` on device and
//  `VideoFileFrameSource` in the Simulator, which has no camera at all
//  (`AVCaptureDevice.default(...)` returns nil there). Both hand out the same
//  `AsyncStream<VideoFrame>`, so the preview and the Core ML engine see an
//  identical frame sequence in both environments and nothing downstream has to
//  know which one is running.

import CoreMedia
import CoreVideo

/// A single frame on its way from a producer to the preview and the engine.
///
/// Why this wraps the pixel buffer instead of the stream carrying
/// `CVPixelBuffer` directly: the SDK marks `CVBuffer` `@_nonSendable`, so
/// `AsyncStream<CVPixelBuffer>` is itself non-`Sendable` and cannot cross the
/// isolation boundary between the capture queue and the consuming task under
/// Swift 6 strict concurrency. Verified against the macOS SDK on this host:
///
///     error: conformance of 'CVBuffer' to 'Sendable' is unavailable
///     note: conformance ... has been explicitly marked unavailable here
///
/// The handover really is safe — a producer retains the buffer, yields it, and
/// never touches it again, so ownership transfers with the value. That is
/// precisely what `@unchecked Sendable` is for. Do not "simplify" this back to
/// a bare `CVPixelBuffer`; it will not compile in Swift 6 language mode.
struct VideoFrame: @unchecked Sendable {
    /// 32BGRA pixel buffer, already rotated to portrait by the producer.
    let pixelBuffer: CVPixelBuffer
    /// Presentation timestamp, in the producer's own timebase. Only differences
    /// between frames from the same source are meaningful.
    let time: CMTime
}

/// The single stream type both display and inference consume.
typealias FrameStream = AsyncStream<VideoFrame>

extension AsyncStream where Element == VideoFrame {
    /// Builds the frame stream with the project-wide **latest-frame-wins**
    /// policy, so the buffering decision lives in exactly one place.
    ///
    /// `.bufferingNewest(1)` is load-bearing. Inference is far slower than
    /// capture (30–60 fps in, one three-stage pass out), and the default
    /// `.unbounded` policy would let the producer pile up every frame the
    /// consumer failed to keep up with: unbounded memory growth plus an overlay
    /// that drifts further behind reality the longer the app runs. With a
    /// one-slot buffer a new frame simply replaces the pending one, so a
    /// consumer that wakes up late always gets the most recent frame and the
    /// backlog is structurally impossible. PR7's `isInferring` gate is an
    /// additional, cheaper filter on top of this — it is not a substitute, and
    /// this policy must stay bounded even after that gate exists.
    static func makeLatestWins() -> (stream: FrameStream, continuation: FrameStream.Continuation) {
        AsyncStream.makeStream(of: VideoFrame.self, bufferingPolicy: .bufferingNewest(1))
    }
}

/// Everything a frame producer can fail with. Typed on purpose: the UI has to
/// tell "grant camera access in Settings" apart from "this build has no clip
/// bundled", and neither case may crash.
enum FrameSourceError: Error, Equatable, LocalizedError {
    /// No capture device — the normal state in the Simulator.
    case cameraUnavailable
    /// The user denied (or an MDM profile restricts) camera access.
    case cameraAccessDenied
    /// `AVCaptureSession` refused the input/output wiring.
    case captureConfigurationFailed(String)
    /// The bundled clip the Simulator path needs is not in the app bundle.
    case missingBundledVideo(resource: String, fileExtension: String)
    /// The clip loaded but produced no decodable video frames.
    case videoDecodeUnavailable(String)

    var errorDescription: String? {
        switch self {
        case .cameraUnavailable:
            return "No camera is available on this device."
        case .cameraAccessDenied:
            return "Camera access is denied. Enable it in Settings › Privacy › Camera."
        case .captureConfigurationFailed(let detail):
            return "Could not configure the camera: \(detail)"
        case .missingBundledVideo(let resource, let fileExtension):
            return "The sample clip \(resource).\(fileExtension) is not bundled in this build."
        case .videoDecodeUnavailable(let detail):
            return "Could not read frames from the sample clip: \(detail)"
        }
    }
}

/// A producer of camera-shaped frames.
///
/// Lifecycle: `start()` is async and throwing because both implementations do
/// real work that can legitimately fail (permission prompts, device lookup,
/// bundle lookup). `stop()` is synchronous, non-throwing and idempotent so it
/// can be called from `onDisappear` or a scene-phase change.
///
/// `stop()` deliberately does **not** finish the stream: it parks the producer
/// so a later `start()` resumes into the same stream, which is what
/// foreground/background cycling needs. The stream finishes when the source is
/// deallocated, and consumers otherwise end their loop by cancelling their task.
///
/// `frames` is a single stored stream, so it is **single-consumer**: iterating
/// it from two places splits the frames between them rather than duplicating
/// them. PR7's view model owns the one `for await` loop and fans each frame out
/// to the preview and to the engine.
protocol FrameSource: AnyObject, Sendable {
    var frames: FrameStream { get }
    func start() async throws
    func stop()
}

/// The one place in the app that knows the Simulator has no camera.
///
/// Views must never branch on `targetEnvironment` themselves — they ask for a
/// `FrameSource` and get whichever one this build can actually run.
enum FrameSourceFactory {
    static func makeDefault() -> any FrameSource {
        #if targetEnvironment(simulator)
        // No capture hardware in the Simulator; loop the bundled clip instead.
        return VideoFileFrameSource()
        #else
        return CameraFrameSource()
        #endif
    }
}
