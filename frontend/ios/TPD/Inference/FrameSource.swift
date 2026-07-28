//  FrameSource.swift
//  The one contract every frame producer implements: `CameraFrameSource` on
//  device, `VideoFileFrameSource` in the Simulator, which has no camera at all
//  (`AVCaptureDevice.default(...)` returns nil there). Both hand out the same
//  `AsyncStream<VideoFrame>`, so nothing downstream knows which is running.

import CoreMedia
import CoreVideo
import Foundation

/// A single frame on its way from a producer to the preview and the engine.
///
/// Why this wraps the pixel buffer rather than the stream carrying
/// `CVPixelBuffer` directly: the SDK marks `CVBuffer` `@_nonSendable`, so
/// `AsyncStream<CVPixelBuffer>` is itself non-`Sendable` and cannot cross the
/// boundary between the capture queue and the consuming task under Swift 6
/// ("conformance of 'CVBuffer' to 'Sendable' is unavailable", verified against
/// the macOS SDK here). The handover really is safe: a producer retains the
/// buffer, yields it and never touches it again, so ownership transfers with
/// the value. Do not "simplify" this back to a bare `CVPixelBuffer`.
struct VideoFrame: @unchecked Sendable {
    /// 32BGRA, already rotated upright by the producer: the camera path turns
    /// the sensor's landscape output 90° to portrait, the file path bakes in the
    /// track's `preferredTransform`. Both therefore hand the overlay and the
    /// engine one coordinate space — geometry computed against one source's
    /// frames is valid against the other's.
    let pixelBuffer: CVPixelBuffer
    /// Presentation timestamp in the producer's own timebase; only differences
    /// within one source are meaningful.
    let time: CMTime
}

/// The single stream type both display and inference consume.
typealias FrameStream = AsyncStream<VideoFrame>

/// The stop/start bookkeeping both producers share. It touches no AVFoundation,
/// so the whole lifecycle can be driven by a fake producer on any platform.
///
/// **One stream per run.** A single stored stream cannot work: cancelling the
/// consuming task terminates that stream for good, after which every `yield` is
/// silently dropped — a source vending the same stream from a later `start()`
/// would deliver nothing, forever, with no error anywhere. `begin()` opens a
/// fresh one, `stop()` finishes it.
///
/// **Intent survives suspension.** `start()` awaits real work — a permission
/// prompt the user may leave on screen indefinitely, an asset load — and a
/// `stop()` landing during that await has to win. It is recorded here
/// synchronously, as a token bump, so start work resuming behind it re-checks
/// `isCurrent(_:)` and declines to commit.
final class FrameSourceLifecycle: @unchecked Sendable {
    private let lock = NSLock()
    private var continuation: FrameStream.Continuation?
    private var token: UInt64 = 0

    /// Opens the stream for a new start attempt, finishing whatever a previous
    /// run left behind, and returns the token identifying this attempt.
    ///
    /// `.bufferingNewest(1)` is the project-wide **latest-frame-wins** policy,
    /// and this is the only place a frame stream is made. It is load-bearing:
    /// inference is far slower than capture (30–60 fps in, one three-stage pass
    /// out), and `.unbounded` would pile up every frame the consumer missed —
    /// unbounded memory plus an overlay drifting further behind reality the
    /// longer the app runs. One slot means a new frame replaces the pending one,
    /// so a late consumer gets the newest. PR7's `isInferring` gate layers a
    /// cheaper filter on top; it is not a substitute, and this policy must stay
    /// bounded after it lands.
    func begin() -> (stream: FrameStream, token: UInt64) {
        let (stream, continuation) = AsyncStream.makeStream(of: VideoFrame.self,
                                                            bufferingPolicy: .bufferingNewest(1))
        let (previous, issued) = lock.withLock { () -> (FrameStream.Continuation?, UInt64) in
            defer { self.continuation = continuation }
            token &+= 1
            return (self.continuation, token)
        }
        previous?.finish()
        return (stream, issued)
    }

    /// False once a `stop()` — or a newer `start()` — has superseded this
    /// attempt. Producers must consult it after every suspension point, before
    /// any side effect that would leave hardware running.
    func isCurrent(_ token: UInt64) -> Bool {
        lock.withLock { self.token == token && continuation != nil }
    }

    /// The newest token, live or not. A teardown queued by `stop()` compares it
    /// on arrival and stands down if a `start()` has since superseded that stop,
    /// which is what makes the two safe to queue on unordered executors.
    ///
    /// Compare it only against the value `stop()` **returned**. Deriving the
    /// teardown's own token by reading this property a second time reintroduces
    /// the race the comparison exists to close — see `stop()`.
    var latest: UInt64 { lock.withLock { token } }

    /// Hands a frame to the live stream; a no-op once stopped, which is what
    /// lets a producer callback still in flight finish harmlessly.
    func yield(_ frame: VideoFrame) {
        lock.withLock { continuation }?.yield(frame)
    }

    /// Finishes the live stream, invalidates any start attempt still in flight,
    /// and returns the token this stop now owns. Idempotent, callable from any
    /// thread.
    ///
    /// **The token comes back from the same lock acquisition that bumped it**,
    /// and that is the whole point of the return value. A producer whose
    /// teardown runs asynchronously has to know *which* run it was asked to tear
    /// down, and recovering that by reading `latest` afterwards is a second
    /// acquisition with a gap in front of it: a `begin()` landing in the gap
    /// bumps the token again, the teardown reads the **new** run's token, its
    /// stand-down check then compares equal, and it tears down the run that just
    /// started — precisely the stop/start race the check was added to close,
    /// moved rather than fixed. One acquisition leaves no gap to land in.
    @discardableResult
    func stop() -> UInt64 {
        let (live, stopped) = lock.withLock { () -> (FrameStream.Continuation?, UInt64) in
            let previous = continuation
            continuation = nil
            token &+= 1
            return (previous, token)
        }
        live?.finish()
        return stopped
    }
}

/// Everything a frame producer can fail with. Typed on purpose: the UI has to
/// tell "grant camera access in Settings" apart from "no clip bundled", and
/// neither case may crash.
enum FrameSourceError: Error, Equatable, LocalizedError {
    /// No capture device — the normal state in the Simulator.
    case cameraUnavailable
    case cameraAccessDenied
    /// `AVCaptureSession` refused the input/output wiring.
    case captureConfigurationFailed(String)
    case missingBundledVideo(resource: String, fileExtension: String)
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
/// `start()` is async and throwing because both implementations do real work
/// that can fail (permission prompts, device lookup, bundle lookup), and it
/// **returns the stream for that run** — see `FrameSourceLifecycle` for why a
/// stored stream cannot survive a stop/start cycle. Consume only the stream the
/// call handed back, and from one place only: a stream is single-consumer, so
/// iterating it twice splits the frames rather than duplicating them.
///
/// `stop()` is synchronous, non-throwing and idempotent — safe from
/// `onDisappear` or a scene-phase change. It finishes the current stream, so the
/// consumer's loop ends by itself, and it wins even when it lands while `start()`
/// is suspended: that start is abandoned, and the stream it returns is finished.
protocol FrameSource: AnyObject, Sendable {
    func start() async throws -> FrameStream
    func stop()
}

/// The one place that knows the Simulator has no capture hardware and must loop
/// a bundled clip instead. Views never branch on `targetEnvironment` themselves
/// — they ask for a `FrameSource` and get whichever one this build can run.
enum FrameSourceFactory {
    static func makeDefault() -> any FrameSource {
        #if targetEnvironment(simulator)
        return VideoFileFrameSource()
        #else
        return CameraFrameSource()
        #endif
    }
}
