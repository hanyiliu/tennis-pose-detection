//  LiveViewModel.swift
//  The real-time loop: FrameSource -> inference -> published frame + result.

import CoreGraphics
import CoreImage
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
    private let engine: TPDInferenceEngine
    /// Display rasterization only. Pointedly *not* the engine's context, whose
    /// colour management is switched off for numeric parity with the Python
    /// pipeline — those settings are for matching floats, not for looking right.
    private let display = CIContext()

    init() throws { engine = try TPDInferenceEngine() }

    /// One frame, start to finish. Returns `nil` only if the display raster fails,
    /// which is a dropped frame rather than an error worth surfacing.
    func process(_ frame: VideoFrame) throws -> RenderedFrame? {
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
    var overlay = OverlayOptions()

    private let source: any FrameSource
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
        defer { source.stop() }
        let worker: InferenceWorker
        do {
            worker = try InferenceWorker()
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
            Task { [weak self] in
                do {
                    let rendered = try await worker.process(video)
                    self?.finish(rendered, nil)
                } catch {
                    self?.finish(nil, error)
                }
            }
        }
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
