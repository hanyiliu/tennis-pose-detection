//  PreloadCoordinator.swift
//  Picking a clip starts analysing the WHOLE clip, not just the frame on screen: analysing only
//  what was on screen meant every scrub to an unvisited frame cost a full ~9.7 s pass. This
//  fills `ResultCache` ahead of the user, so by the time they scrub the answer is already there.

import AVFoundation
import CoreGraphics
import Foundation
import Observation

/// Which frames a sweep still owes and — given where the user is looking — which one to do next.
/// Pure: no AVFoundation, no Core ML, no clock. This ordering decides whether a scrub waits ten
/// seconds, so it is a test rather than a stopwatch.
struct PreloadPlan: Equatable, Sendable {
    /// Every Nth frame of the clip is analysed. See `PreloadCoordinator.defaultCadence`.
    let cadence: Int
    /// Owed, not yet handed to a worker. A `Set` rather than a queue because `take` is a search
    /// by cost: the order is recomputed from wherever the playhead is, and a stored one would be
    /// stale by the next frame.
    private(set) var pending: Set<Int>
    /// The M in "N of M", and the N. `analysed` is one-way and clamped — see `complete()`.
    private(set) var total: Int
    private(set) var analysed = 0
    var isComplete: Bool { analysed >= total }
    /// How far from the frame on screen a preloaded answer may be and still be worth showing:
    /// half a cadence, so the nearest analysed frame is always inside it. At the default 5 that
    /// is 2 frames — 67 ms at 30 fps, well within one stroke phase.
    var tolerance: Int { cadence / 2 }

    /// The grid, sized to the cache rather than the other way round. A sweep scheduling more
    /// frames than `ResultCache` holds would spend ~9.7 s each evicting its own earlier answers,
    /// and a user scrubbing back would wait again for a frame they already paid for; raising the
    /// cadence until the grid fits is the one lever that cannot regress, and at the 600-entry
    /// bound with cadence 5 it only bites past 100 s of 30 fps footage.
    init(frameCount: Int, cadence requested: Int, capacity: Int) {
        let count = max(frameCount, 0), bound = max(capacity, 1)
        cadence = max(1, requested, (count + bound - 1) / bound)
        pending = Set(stride(from: 0, to: count, by: cadence))
        total = pending.count
    }

    /// The next frame to analyse: the one nearest the playhead, forward first. A front-to-back
    /// sweep is the wrong order the moment the user scrubs to the middle — they would sit through
    /// every frame they skipped. Cost is the distance to the playhead with frames *behind* it
    /// counted double, so the frame on screen goes first, its neighbours next, and what is behind
    /// is filled in only once what lies ahead is done; forward is favoured because the transport
    /// only runs that way. Ties break low, so the order never depends on `Set`'s hashing.
    mutating func take(nearest playhead: Int) -> Int? {
        guard let pick = pending.min(by: { Self.cost($0, playhead) < Self.cost($1, playhead) })
        else { return nil }
        pending.remove(pick)
        return pick
    }
    static func cost(_ index: Int, _ playhead: Int) -> (Int, Int) {
        index >= playhead ? (index - playhead, index) : (2 * (playhead - index), index)
    }

    /// One taken frame produced a result. Clamped and one-way: "analysed N of M" is on screen
    /// while the user decides whether to wait, and a count that jumped about is worse than none.
    mutating func complete() { analysed = min(analysed + 1, total) }
    /// A frame the decoder would not give up. It leaves the plan rather than being retried
    /// forever or counted as analysed — M shrinks, N stays true, and the sweep still finishes.
    mutating func drop() { total = max(total - 1, analysed) }
}

/// A decoded frame on its way to the engine. `@unchecked Sendable` for the reason `RenderedFrame`
/// is: `CGImage` is immutable and the decoder never touches it again.
struct DecodedFrame: @unchecked Sendable { let image: CGImage }

/// Random-access decode of one clip, off the main actor. `AVAssetImageGenerator`, not the
/// `AVAssetReader` the exporter uses: a reader is sequential and the point of this sweep is to
/// jump to whatever the user is looking at.
actor ClipFrameLoader {
    private let url: URL
    private var generator: AVAssetImageGenerator?
    init(url: URL) { self.url = url }
    /// Built here and never in `init`: an actor's `init` body runs in the **caller's** isolation
    /// domain, so an asset opened in a stored property would open on main — the trap
    /// `StillInferenceWorker` documents for its models.
    func frame(at seconds: Double) async throws -> DecodedFrame {
        let generator = generator ?? Self.make(url)
        self.generator = generator
        try Task.checkCancellation()
        let time = CMTime(seconds: max(seconds, 0), preferredTimescale: 600)
        // The completion-handler form, not `await generator.image(at:)`: awaiting that one
        // *sends* the non-`Sendable` generator out of this actor and Swift 6 rejects it. This
        // call returns at once and the handler resumes us, so the generator never leaves.
        return try await withCheckedThrowingContinuation { continuation in
            generator.generateCGImageAsynchronously(for: time) { image, _, error in
                if let image { continuation.resume(returning: DecodedFrame(image: image)) }
                else { continuation.resume(throwing: error ?? CancellationError()) }
            }
        }
    }
    private static func make(_ url: URL) -> AVAssetImageGenerator {
        let generator = AVAssetImageGenerator(asset: AVURLAsset(url: url))
        // Keeps these frames in the same upright space the player's video composition produces,
        // so a result measured here is valid over the picture on screen.
        generator.appliesPreferredTrackTransform = true
        // Exact on both sides: a tolerant generator returns the nearest sync sample, and the
        // result would then be filed under a frame index it was not measured at — precisely the
        // mismatch `ResultCache`'s indexing exists to prevent.
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero
        return generator
    }
}

/// Owns the sweep: one per presentation, and the only thing in the app that starts inference the
/// user did not ask for. `@MainActor` because `ResultCache` is and a view reads the progress;
/// every expensive step is an `await` off this actor, so main is free throughout.
@MainActor
@Observable
final class PreloadCoordinator {
    /// **5.** Two numbers set it. Coverage: half a cadence is 83 ms at 30 fps, comfortably inside
    /// the ~150 ms a stroke phase lasts — the reasoning behind `VideoExporter`'s cadence of 3 — so
    /// the nearest analysed frame still shows the pose on screen. Cost: one pass is 8-10 s on the
    /// booted simulator, where the 150-frame test clip swept in 4 minutes at 5 and would take 20
    /// at 1. On a device it is seconds, which is why this is a parameter and not a fact.
    static let defaultCadence = 5

    private(set) var plan: PreloadPlan?
    @ObservationIgnored private var task: Task<Void, Never>?
    @ObservationIgnored private var playhead = 0
    /// Bumped by every `begin`, so a pass resuming after its clip was superseded sees a token no
    /// longer current and stands down instead of writing into the new clip's cache. `cancel()`
    /// cannot cover this alone: cancellation is only observed between awaits, and the inference
    /// in the middle is one uninterruptible call.
    @ObservationIgnored private var generation = 0
    /// Run on the main actor after each pass lands. A preview paused on one frame gets no new
    /// pixel buffers and so nothing re-reads the cache: without this the answer the sweep just
    /// produced for the frame on screen would sit there unseen until the playhead moved.
    @ObservationIgnored var onResult: (() -> Void)?
    var tolerance: Int { plan?.tolerance ?? 0 }
    /// A sweep is live and still owes frames, so it owns the engine and the preview must not
    /// queue a pass behind it. See `VideoPreviewModel.sync()`.
    var isSweeping: Bool { plan.map { !$0.isComplete } ?? false }

    /// Supersedes whatever was running and returns immediately; the sweep is a task.
    func begin(url: URL, frameCount: Int, frameRate: Double,
               cadence: Int = PreloadCoordinator.defaultCadence, cache: ResultCache) {
        let loader = ClipFrameLoader(url: url), rate = frameRate > 0 ? frameRate : 30
        begin(frameCount: frameCount, cadence: cadence, cache: cache) { index in
            // Mid-period, not the boundary: `index / rate` sits exactly on the edge between two
            // frames and a hair of float error either way decodes the neighbour.
            let frame = try await loader.frame(at: (Double(index) + 0.5) / rate)
            return try await StillInferenceWorker.shared.analyse(frame.image)
        }
    }

    /// The same sweep over an injected step: how the scheduling is tested without a clip, Core
    /// ML, or ten seconds a frame.
    func begin(frameCount: Int, cadence: Int = PreloadCoordinator.defaultCadence,
               cache: ResultCache,
               analyse: @escaping @Sendable (Int) async throws -> TPDResult) {
        cancel()
        generation += 1
        playhead = 0                  // a new clip is at its start until the first tick says so
        let token = generation
        plan = PreloadPlan(frameCount: frameCount, cadence: cadence, capacity: cache.capacity)
        task = Task { [weak self] in await self?.sweep(token, cache, analyse) }
    }
    /// Where the user is looking, in frame indices. Read at the top of every pass, so a scrub
    /// re-aims the sweep from the next frame on rather than at the end of it.
    func look(at index: Int) { playhead = index }
    /// Prompt in the only sense one frame of Core ML allows: nothing new is started, and the
    /// pass in flight is discarded rather than published when it lands.
    func cancel() { task?.cancel(); task = nil }
    private func sweep(_ token: Int, _ cache: ResultCache,
                       _ analyse: @Sendable (Int) async throws -> TPDResult) async {
        while generation == token, !Task.isCancelled {
            guard let index = plan?.take(nearest: playhead) else { return }
            // Free ones first: a frame the user sat on was analysed by the preview's own
            // opportunistic path, and re-running it costs ~9.7 s for a result already held.
            guard cache.value(at: index) == nil else { plan?.complete(); continue }
            do {
                let result = try await analyse(index)
                // Both re-checked after the await: ~9.7 s have passed, and the screen may be
                // gone or showing another clip whose cache this must not write into.
                guard generation == token, !Task.isCancelled else { return }
                cache.insert(result, at: index)
                plan?.complete()
                onResult?()
            } catch is CancellationError {
                return
            } catch {
                NSLog("TPD preload: frame %d skipped — %@", index, error.localizedDescription)
                plan?.drop()
            }
        }
    }
}
