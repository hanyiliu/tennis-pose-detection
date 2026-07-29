//  LiveViewModel.swift
//  The real-time loop: FrameSource -> inference -> published frame + result.

import CoreGraphics
import CoreImage
import Foundation
import Observation
import SwiftUI

/// A frame and the result measured **from that same frame**. Publishing them as
/// one value is what keeps the overlay from being drawn over a picture it does
/// not describe; two separate properties would drift apart by one inference.
///
/// `@unchecked Sendable` because `CGImage` carries no conformance in this SDK.
/// The handover is a true transfer: the image is created inside `process` and
/// never touched again there.
struct RenderedFrame: @unchecked Sendable {
    let image: CGImage
    let result: TPDResult
}

/// Off-main home for the expensive half. `TPDInferenceEngine` is deliberately not
/// `Sendable` — it owns `MLModel`s and reusable pixel buffers it rewrites every
/// call — so actor isolation, one instance, serialized calls, is exactly the
/// contract it asks for, and is what makes this legal under Swift 6.
actor InferenceWorker {
    /// Built by `load()`, **never** by `init` — which is the whole reason this
    /// is a `var?`. An actor's `init` body runs synchronously in the *caller's*
    /// isolation domain; it does not hop to the actor. So `TPDInferenceEngine()`
    /// inside one performs both `MLModel(contentsOf:)` loads on whatever thread
    /// said `InferenceWorker()` — main — and froze the UI for 1.5s at launch.
    /// Do not "simplify" this back to `let engine` plus a throwing `init`.
    private var engine: TPDInferenceEngine?
    /// Display rasterization only. Pointedly *not* the engine's context, whose
    /// colour management is switched off for numeric parity with the Python
    /// pipeline — those settings are for matching floats, not for looking right.
    private let display = CIContext()

    /// Loads both Core ML models, off the main actor. Idempotent, so a caller
    /// re-entering the loop after a failure retries the load rather than
    /// assuming it already happened.
    func load() throws {
        guard engine == nil else { return }
        engine = try TPDInferenceEngine()
        // Cheap standing proof: if the load ever migrates back into `init` this
        // prints YES, which is otherwise visible only in a profile.
        #if DEBUG
        NSLog("TPD model load ran on the main thread: %@", Thread.isMainThread ? "YES" : "NO")
        #endif
    }

    /// One frame, start to finish. Returns `nil` only if the display raster fails,
    /// which is a dropped frame rather than an error worth surfacing.
    func process(_ frame: VideoFrame) throws -> RenderedFrame? {
        // Cheap after the first call, and it keeps the engine's existence an
        // invariant of this method rather than of the caller's call order.
        try load()
        guard let engine else { return nil }
        let image = CIImage(cvPixelBuffer: frame.pixelBuffer)
        let result = try engine.predict(frame: image)
        guard let raster = display.createCGImage(image, from: image.extent) else { return nil }
        return RenderedFrame(image: raster, result: result)
    }
}

/// Live Camera View state. `@MainActor` throughout: every stored property here is
/// read by SwiftUI, and the only thing that leaves main is `InferenceWorker`.
@MainActor
@Observable
final class LiveViewModel {
    /// Why the screen is empty, phrased for a person rather than for a log.
    struct Failure: Equatable {
        let title: String, detail: String, symbol: String
    }

    private(set) var frame: RenderedFrame?
    private(set) var failure: Failure?
    /// **The recovery token.** `run()` returns for good on a failed start and
    /// cannot restart itself, so the view drives `.task(id:)` off this and
    /// bumping it is what re-enters the loop. Without it the camera-denied
    /// state's "enable it in Settings" is advice the app cannot honour: coming
    /// back from Settings finds the loop already gone.
    private(set) var attempt = 0
    var overlay = OverlayOptions()

    /// Held rather than made per run, so a retry re-uses models already loaded
    /// and only re-attempts the load if it was the thing that failed.
    private let worker = InferenceWorker()
    private let source: any FrameSource
    /// Which `run()` owns `source`. `.task(id:)` starts the replacement without
    /// awaiting the cancelled run's teardown, so a stale `defer` could otherwise
    /// call `stop()` *after* the new run's `start()` and silently kill it.
    private var generation = 0
    /// **The drop gate.** Capture runs at 30–60 fps and one three-stage pass does
    /// not. While this is true every arriving frame is discarded unread, so the
    /// engine always works on the newest frame rather than an ageing queue. The
    /// stream's `.bufferingNewest(1)` is the same policy one layer down; this
    /// saves even the buffer swap and, more importantly, the inference task.
    private var isInferring = false

    init(source: any FrameSource = FrameSourceFactory.makeDefault()) {
        self.source = source
    }

    /// Runs until the enclosing `.task` is cancelled, which ends the stream and
    /// therefore this loop. Never throws: every failure becomes a visible state.
    func run() async {
        generation += 1
        let mine = generation
        defer { if generation == mine { source.stop() } }
        do {
            // `await`, not a constructor: this suspends main while the models
            // load on the worker's executor. See `InferenceWorker.load()`.
            try await worker.load()
        } catch {
            failure = Self.failure(for: error)
            return
        }
        let stream: FrameStream
        do {
            stream = try await source.start()
        } catch {
            failure = Self.failure(for: error)
            return
        }
        failure = nil
        for await video in stream {
            guard !isInferring else { continue }
            isInferring = true
            // Inherits this actor, so it resumes on main to publish — but the
            // await below hops to `worker`, and that is where inference runs.
            Task { [weak self, worker] in
                do {
                    let rendered = try await worker.process(video)
                    self?.finish(rendered, nil)
                } catch {
                    self?.finish(nil, error)
                }
            }
        }
    }

    /// Re-enters the loop. Clearing `failure` first is deliberate: it swaps the
    /// dead-end message for the neutral "starting" state, so a retry that fails
    /// again is visibly a second attempt rather than a button that did nothing.
    func retry() {
        failure = nil
        attempt += 1
    }

    private func finish(_ rendered: RenderedFrame?, _ error: Error?) {
        if let rendered {
            frame = rendered
            failure = nil
        } else if let error, frame == nil {
            // Only worth showing while there is nothing else on screen; a
            // transient fault mid-stream must not blank a working preview.
            failure = Self.failure(for: error)
        }
        isInferring = false
    }

    private static func failure(for error: Error) -> Failure {
        if let error = error as? FrameSourceError {
            switch error {
            case .missingBundledVideo(let resource, let fileExtension):
                return Failure(
                    title: "No live feed in the Simulator",
                    detail: "There is no camera here, so TPD loops a bundled clip instead — and "
                        + "\(resource).\(fileExtension) is not in this build. Drop one at "
                        + "frontend/ios/TPD/Media/\(resource).\(fileExtension), re-run "
                        + "`make generate`, and this view fills with frames. On a real device "
                        + "the camera is used directly and nothing needs bundling.",
                    symbol: "film.stack")
            case .cameraAccessDenied:
                return Failure(title: "Camera access is off",
                               detail: error.localizedDescription, symbol: "lock.slash")
            default:
                return Failure(title: "No frames", detail: error.localizedDescription,
                               symbol: "video.slash")
            }
        }
        if let error = error as? TPDInferenceError {
            return Failure(title: "The model is not in this build",
                           detail: error.localizedDescription, symbol: "cube.transparent")
        }
        return Failure(title: "Live view stopped", detail: error.localizedDescription,
                       symbol: "exclamationmark.triangle")
    }
}
