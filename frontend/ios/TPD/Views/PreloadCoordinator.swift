//  PreloadCoordinator.swift
//  Picking a clip analyses the WHOLE clip: analysing only the frame on screen meant every scrub to
//  an unvisited one cost a full ~9.7 s pass.

import AVFoundation
import CoreGraphics
import Foundation
import Observation

/// Which frames a sweep still owes and — given where the user is looking — which one to do next.
/// Pure: no AVFoundation, no Core ML, no clock, so the order deciding whether a scrub waits ten
/// seconds is a test rather than a stopwatch.
struct PreloadPlan: Equatable, Sendable {
    /// Every Nth frame of the clip is analysed. See `PreloadCoordinator.defaultCadence`.
    let cadence: Int
    /// Carried only so `tolerance` can be stated in seconds of clip rather than in grid steps.
    let frameRate: Double
    /// Owed. A `Set` and not a queue: `take` searches by cost from wherever the playhead is.
    private(set) var pending: Set<Int>
    /// The M in "N of M", and the N. `analysed` is one-way and clamped — see `complete()`.
    private(set) var total: Int
    private(set) var analysed = 0
    var isComplete: Bool { analysed >= total }
    /// Half the ~150 ms a stroke phase lasts, so a neighbour's result is from the same phase of
    /// the same stroke as the frame on screen — the whole claim serving one makes.
    static let toleranceSeconds = 0.075
    /// How far a preloaded answer may be from the frame on screen: bounded in TIME, converted to
    /// frames only here. Half a cadence is 83 ms at cadence 5, but a long clip thins the grid to
    /// fit the cache and half of *that* is whole seconds — 2 frames either way, not 30.
    var tolerance: Int { min(cadence / 2, Int(Self.toleranceSeconds * frameRate)) }

    /// The grid, sized to the cache rather than the other way round: a sweep scheduling more
    /// frames than `ResultCache` holds would spend ~9.7 s each evicting its own earlier answers.
    init(frameCount: Int, cadence requested: Int, capacity: Int, frameRate: Double = 30) {
        let count = max(frameCount, 0), bound = max(capacity, 1)
        self.frameRate = frameRate > 0 ? frameRate : 30
        cadence = max(1, requested, (count + bound - 1) / bound)
        pending = Set(stride(from: 0, to: count, by: cadence))
        total = pending.count
    }

    /// The next frame to analyse: nearest the playhead, forward first — front-to-back is the wrong
    /// order the moment the user scrubs to the middle. Cost is the distance to the playhead with
    /// frames *behind* it doubled, the only way the transport runs; ties break low, not on hashing.
    mutating func take(nearest playhead: Int) -> Int? {
        guard let pick = pending.min(by: { Self.cost($0, playhead) < Self.cost($1, playhead) })
        else { return nil }
        pending.remove(pick)
        return pick
    }
    static func cost(_ index: Int, _ playhead: Int) -> (Int, Int) {
        index >= playhead ? (index - playhead, index) : (2 * (playhead - index), index)
    }

    /// Clamped and one-way: "analysed N of M" is on screen while the user decides whether to wait.
    mutating func complete() { analysed = min(analysed + 1, total) }
    /// A frame the decoder refused leaves the plan: M shrinks, N stays true, the sweep finishes.
    mutating func drop() { total = max(total - 1, analysed) }
}

/// `@unchecked Sendable` for `RenderedFrame`'s reason: `CGImage` is immutable and never reused.
struct DecodedFrame: @unchecked Sendable { let image: CGImage }

/// Random-access decode of one clip, off the main actor. `AVAssetImageGenerator` and not the
/// exporter's `AVAssetReader`, which is sequential: this sweep jumps to where the user is looking.
actor ClipFrameLoader {
    private let url: URL
    private var generator: AVAssetImageGenerator?
    init(url: URL) { self.url = url }
    /// Built here and never in `init`: an actor's `init` body runs in the **caller's** isolation
    /// domain, so an asset opened in a stored property would open on main.
    func frame(at seconds: Double) async throws -> DecodedFrame {
        let generator = generator ?? Self.make(url)
        self.generator = generator
        try Task.checkCancellation()
        let time = CMTime(seconds: max(seconds, 0), preferredTimescale: 600)
        // Completion-handler form: `await generator.image(at:)` *sends* the non-`Sendable`
        // generator out of this actor, which Swift 6 rejects.
        return try await withCheckedThrowingContinuation { continuation in
            generator.generateCGImageAsynchronously(for: time) { image, _, error in
                if let image { continuation.resume(returning: DecodedFrame(image: image)) }
                else { continuation.resume(throwing: error ?? CancellationError()) }
            }
        }
    }
    private static func make(_ url: URL) -> AVAssetImageGenerator {
        let generator = AVAssetImageGenerator(asset: AVURLAsset(url: url))
        // The upright space the player's composition produces, so a result measured here is valid
        // over the picture on screen. Exact times too: a tolerant generator returns the nearest
        // sync sample, and the result would be filed under a frame it was not measured at.
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero
        return generator
    }
}

/// Owns the sweep: one per presentation, and the only thing in the app that starts inference the
/// user did not ask for. `@MainActor` because `ResultCache` is; every expensive step awaits off it.
@MainActor
@Observable
final class PreloadCoordinator {
    /// **5.** Coverage: the tolerance stays inside the ~150 ms a stroke phase lasts. Cost: one pass
    /// is 8-10 s here, where the 150-frame test clip swept in 4 minutes at 5 and 20 minutes at 1.
    static let defaultCadence = 5

    private(set) var plan: PreloadPlan?
    @ObservationIgnored private var task: Task<Void, Never>?
    @ObservationIgnored private var playhead = 0
    /// Bumped by every `begin`, so a pass resuming after its clip was superseded stands down rather
    /// than writing into the new cache. `cancel()` is observed only between awaits, and the
    /// inference between them is one uninterruptible call.
    @ObservationIgnored private var generation = 0
    @ObservationIgnored private var suspended = false
    /// Run on main after every pass — landed, free or skipped. A preview paused on one frame gets
    /// no new pixel buffers, so without this the sweep's work for it would sit unseen. See `sweep`.
    @ObservationIgnored var onResult: (() -> Void)?
    var tolerance: Int { plan?.tolerance ?? 0 }
    /// Frames are owed and the engine is spoken for, so the preview must not queue behind it.
    var isSweeping: Bool { plan.map { !$0.isComplete } ?? false }

    /// Supersedes whatever was running and returns immediately; the sweep is a task.
    func begin(url: URL, frameCount: Int, frameRate: Double,
               cadence: Int = PreloadCoordinator.defaultCadence, cache: ResultCache) {
        let loader = ClipFrameLoader(url: url), rate = frameRate > 0 ? frameRate : 30
        begin(frameCount: frameCount, cadence: cadence, cache: cache) { index in
            // Mid-period: `index / rate` is the edge between two frames, where float error decodes
            let frame = try await loader.frame(at: (Double(index) + 0.5) / rate)   // the neighbour
            return try await StillInferenceWorker.shared.analyse(frame.image)
        }
    }

    /// The same sweep over an injected step: scheduling tested without a clip or ten seconds a go.
    func begin(frameCount: Int, cadence: Int = PreloadCoordinator.defaultCadence,
               cache: ResultCache,
               analyse: @escaping @Sendable (Int) async throws -> TPDResult) {
        cancel()
        generation += 1
        playhead = 0                  // a new clip is at its start until the first tick says so
        let token = generation
        plan = PreloadPlan(frameCount: frameCount, cadence: cadence, capacity: cache.capacity,
                           frameRate: cache.frameRate)
        task = Task { [weak self] in await self?.sweep(token, cache, analyse) }
    }
    /// Where the user is looking, read at the top of every pass: a scrub re-aims the sweep at the
    /// next frame boundary, not at the end of the run.
    func look(at index: Int) { playhead = index }
    /// Prompt in the only sense Core ML allows: nothing new starts, the pass in flight is dropped.
    func cancel() { task?.cancel(); task = nil }
    /// Stood down for the duration of an export: both are Core ML through the one engine, so a
    /// sweep left running against a burn-in only makes the export slower. A pause and not a cancel
    /// — the plan keeps its pending frames, and the pass in flight is worth ~9.7 s and may land.
    func suspend() {
        guard !suspended else { return }
        suspended = true
        NSLog("TPD preload: sweep suspended at %d of %d", plan?.analysed ?? 0, plan?.total ?? 0)
    }
    /// Picked back up when the export card goes; the same N in both log lines is the proof.
    func resume() {
        guard suspended else { return }
        suspended = false
        NSLog("TPD preload: sweep resumed at %d of %d", plan?.analysed ?? 0, plan?.total ?? 0)
    }
    private func sweep(_ token: Int, _ cache: ResultCache,
                       _ analyse: @Sendable (Int) async throws -> TPDResult) async {
        while generation == token, !Task.isCancelled {
            // Idling rather than unwinding is what makes resuming a flag flip, not a restart.
            if suspended { try? await Task.sleep(for: .milliseconds(200)); continue }
            guard let index = plan?.take(nearest: playhead) else { return }
            // Free ones first: re-running a frame the preview already paid for costs ~9.7 s.
            guard cache.value(at: index) == nil else { plan?.complete(); onResult?(); continue }
            do {
                let result = try await analyse(index)
                // Re-checked after the await: ~9.7 s have passed and the screen may be gone.
                guard generation == token, !Task.isCancelled else { return }
                cache.insert(result, at: index)
                plan?.complete()
            } catch is CancellationError {
                return
            } catch {
                NSLog("TPD preload: frame %d skipped — %@", index, error.localizedDescription)
                plan?.drop()
            }
            // Every way out of a pass, not only the one that produced a result: a frame parked off
            // the grid holds its own pass back while `isSweeping`, so a sweep whose last pass ends
            // in a skip and announces nothing strands it on a neighbour's overlay for good.
            onResult?()
        }
    }
}
