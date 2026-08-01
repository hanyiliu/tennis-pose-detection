//  LiveViewModel.swift
//  The real-time loop: FrameSource -> inference -> published frame + result.

import CoreGraphics
import CoreImage
import CoreML
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
    /// Built by `load(_:)`, **never** by `init` — which is the whole reason this
    /// is a `var?`. An actor's `init` body runs synchronously in the *caller's*
    /// isolation domain; it does not hop to the actor. So building an engine
    /// inside one performs every `MLModel(contentsOf:)` load on whatever thread
    /// said `InferenceWorker()` — main — and froze the UI for 1.5s at launch.
    private var engine: (any TPDEngine)?
    /// Display rasterization only. Pointedly *not* the engine's context, whose
    /// colour management is switched off for numeric parity with the Python
    /// pipeline — those settings are for matching floats, not for looking right.
    ///
    /// Built in `load()` for the same reason `engine` is: as a stored property it
    /// would be initialized by the synthesized `init`, which runs in the caller's
    /// isolation domain — so `CIContext()` would be constructed on main. Smaller
    /// than the model load, same trap.
    private var display: CIContext?
    /// Read back by the HUD, from the configuration actually handed to the engine.
    private(set) var computeUnits: MLComputeUnits?
    var loaded: TPDModelEntry? { engine?.entry }

    func load(_ entry: TPDModelEntry) throws {
        if display == nil { display = CIContext() }
        guard engine?.entry.id != entry.id else { return }
        engine = nil  // released BEFORE the replacement: two model sets at once is the peak
        let configuration = MLModelConfiguration()
        engine = try makeEngine(entry, configuration: configuration)
        computeUnits = configuration.computeUnits
        // Cheap standing proof: if the load ever migrates back into `init` this
        // prints YES, which is otherwise visible only in a profile.
        #if DEBUG
        NSLog("TPD model load ran on the main thread: %@", Thread.isMainThread ? "YES" : "NO")
        #endif
    }

    /// One frame, start to finish, plus what it cost. Returns `nil` only if the
    /// display raster fails, which is a dropped frame rather than an error worth
    /// surfacing. The stamps are `ContinuousClock` reads — monotonic, so a clock
    /// adjustment cannot produce a negative stage, which `Date` can — and the last
    /// travels out in the `Sample`, because the meter needs the *instants* of
    /// consecutive passes to see the idle between them, not only their lengths.
    func process(_ frame: VideoFrame) throws -> (frame: RenderedFrame,
                                                 cost: PerformanceMeter.Sample)? {
        guard let engine, let display else { return nil }
        let image = CIImage(cvPixelBuffer: frame.pixelBuffer)
        let started = ContinuousClock.now
        let result = try engine.predict(frame: image)
        let predicted = ContinuousClock.now
        guard let raster = display.createCGImage(image, from: image.extent) else { return nil }
        let rasterized = ContinuousClock.now
        // Dimensions come from the same `image` the timings were taken over, so
        // the panel cannot print one frame's size beside another frame's cost.
        let cost = PerformanceMeter.Sample(
            model: started.duration(to: predicted).inSeconds,
            raster: predicted.duration(to: rasterized).inSeconds,
            finished: rasterized,
            width: Int(image.extent.width.rounded()),
            height: Int(image.extent.height.rounded()))
        return (RenderedFrame(image: raster, result: result), cost)
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
    /// What the HUD draws. Assigned once per completed pass — the only moment
    /// `frame` changes too — so it costs no redraw of its own.
    private(set) var performance = PerformanceMeter.Snapshot()
    private(set) var computeUnits: MLComputeUnits?

    private(set) var models: [TPDModelEntry] = []
    /// What the user chose, which is not yet what runs: `active` is the entry the worker really
    /// built, nil across the gap, so the HUD cannot name a model still loading. A broken registry
    /// is held, not thrown — there is no view to show it to until `run()`.
    private(set) var selected: TPDModelEntry?
    private(set) var active: TPDModelEntry?
    private let registryFailure: Error?

    /// `@ObservationIgnored` because `drop()` fires at capture rate, up to 60 Hz,
    /// and no view reads it — they read `performance`. Registering a mutation that
    /// often is main-actor time a diagnostic must not take from what it watches.
    @ObservationIgnored private var meter = PerformanceMeter()

    /// Held rather than made per run, so a retry re-uses models already loaded
    /// and only re-attempts the load if it was the thing that failed.
    private let worker = InferenceWorker()
    private let source: any FrameSource
    /// **Which pass may PUBLISH**, and only that. Bumped by `select()` too, re-read after every
    /// await in `run()`. Not what a run owns (`producer`), not when the drop gate is released.
    private var generation = 0
    /// **What teardown may act on**, as a held thing rather than as a counter. `source` is shared
    /// and `stop()` finishes whichever stream is live, so a run stopping because it is merely
    /// *stale* tears down the producer the LIVE run is reading. A run claims this as it starts one
    /// — both sources open their stream synchronously inside `start()` — and stops only while it
    /// still holds the claim. A run that started none holds nothing and stops nothing.
    private final class Producer {}
    private var producer: Producer?
    /// **The drop gate.** Capture runs at 30–60 fps and one three-stage pass does
    /// not. While this is true every arriving frame is discarded unread, so the
    /// engine always works on the newest frame rather than an ageing queue. The
    /// stream's `.bufferingNewest(1)` is the same policy one layer down; this
    /// saves even the buffer swap and, more importantly, the inference task.
    private var isInferring = false

    init(source: any FrameSource = FrameSourceFactory.makeDefault()) {
        self.source = source
        do {
            let registry = try ModelRegistry.load()
            (models, selected, registryFailure) = (registry.models, registry.models[0], nil)
        } catch { registryFailure = error }
    }

    /// Switches models without leaving the live view. Bumping `attempt` restarts the loop:
    /// `.task(id:)` cancels the running one — ending the stream, and with it `run()` and its
    /// `defer` — then enters a fresh one, loading the new entry on the worker's executor.
    /// `generation` moves in the same statement, or a pass landing in the gap republishes the
    /// old model's frame and charges its cost to the meter `select()` just reset.
    func select(_ entry: TPDModelEntry) {
        guard entry.id != selected?.id else { return }
        selected = entry
        // The old model's overlay and timings are lies about the new one, so neither survives.
        (frame, active, failure) = (nil, nil, nil)
        meter = PerformanceMeter()
        performance = meter.snapshot
        (generation, attempt) = (generation + 1, attempt + 1)
    }

    /// Runs until the enclosing `.task` is cancelled, which ends the stream and
    /// therefore this loop. Never throws: every failure becomes a visible state.
    func run() async {
        generation += 1
        let mine = generation, claim = Producer()
        defer { if producer === claim { producer = nil; source.stop() } }
        guard let entry = selected else {
            failure = Self.failure(for: registryFailure
                ?? TPDInferenceError.resourceMissing(ModelRegistry.resource + ".json"))
            return
        }
        do {
            // `await`, not a constructor: this suspends main while the models
            // load on the worker's executor. See `InferenceWorker.load(_:)`.
            try await worker.load(entry)
        } catch {
            if generation == mine { failure = Self.failure(for: error) }
            return
        }
        let built = await worker.loaded, units = await worker.computeUnits
        // Superseded across the load: `active` would name the model just left, and `start()`
        // below would `begin()` a second stream, finishing the LIVE run's — a feed that dies.
        guard generation == mine else { return }
        (active, computeUnits) = (built, units)
        let stream: FrameStream
        producer = claim  // starting one is claiming it, and supersedes the previous claim
        do {
            stream = try await source.start()
        } catch {
            if generation == mine { failure = Self.failure(for: error) }
            return
        }
        // Superseded across `start()`: publish nothing. Teardown is not this guard's call.
        guard generation == mine else { return }
        failure = nil
        for await video in stream {
            // Counted where it happens: one increment on a branch that already
            // existed, and the only honest place to see capture outrun inference.
            guard !isInferring else { meter.drop(); continue }
            isInferring = true
            // Inherits this actor, so it resumes on main to publish — but the
            // await below hops to `worker`, and that is where inference runs.
            Task { [weak self, worker] in
                do {
                    let done = try await worker.process(video)
                    self?.finish(done, nil, mine)
                } catch {
                    self?.finish(nil, error, mine)
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

    private func finish(_ done: (frame: RenderedFrame, cost: PerformanceMeter.Sample)?,
                        _ error: Error?, _ generation: Int) {
        isInferring = false  // RELEASING THE GATE. Unconditional: the pass that set it clears it.
        guard generation == self.generation else { return }  // PUBLISHING, and nothing else, below
        if let done {
            frame = done.frame
            meter.record(done.cost)
            performance = meter.snapshot
            failure = nil
        } else if let error, frame == nil {
            // Only worth showing while there is nothing else on screen; a
            // transient fault mid-stream must not blank a working preview.
            failure = Self.failure(for: error)
        }
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
