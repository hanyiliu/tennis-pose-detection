//  PerformanceMeter.swift
//  What the live loop cost, as arithmetic. No clock, no UI, no isolation: the
//  caller stamps the times and this only ever divides, so every number the HUD
//  shows can be pinned down in XCTest instead of by squinting at a camera feed.

import Foundation

/// Rolling statistics over completed inference passes.
///
/// Deliberately a `struct` the view model owns rather than an observable of its
/// own: recording is a few adds on a path that already runs, and publishing is
/// one `Snapshot` assignment per completed pass. Nothing here ticks, so the HUD
/// redraws exactly as often as the preview does — once per inference — and adds
/// no work to the thing it is measuring.
struct PerformanceMeter: Equatable, Sendable {
    /// One finished pass **and the frame it was measured on**, together. Kept as
    /// one value on purpose: the panel prints the dimensions next to the timings,
    /// and separate writers are exactly how the stale-overlay badge once came to
    /// name the wrong frame.
    struct Sample: Equatable, Sendable {
        /// Seconds inside `TPDInferenceEngine.predict` — stages 1, 2 and 3 plus
        /// the crop and letterbox between them, which is one uninterruptible call
        /// from out here and so cannot be split further without reaching into the
        /// engine.
        var model: Double
        /// Seconds inside `CIContext.createCGImage`, the display raster. Separate
        /// because it is not inference and is ~400 ms of every simulator frame.
        var raster: Double
        /// Pixel dimensions of the frame the two timings above came from.
        var width: Int
        var height: Int

        /// Wall time one frame owed the pipeline, end to end.
        var total: Double { model + raster }
    }

    /// Everything the HUD draws, as one value published once per pass.
    ///
    /// The three rates are `nil`, never `0` and never infinity, until there is
    /// something real to divide: a readout that claims "0 fps" before the first
    /// pass lands is a lie for the ten seconds it takes, and `1 / 0` is worse.
    struct Snapshot: Equatable, Sendable {
        var latest: Sample?
        /// Throughput over the window: passes ÷ the seconds those passes took.
        /// Not the mean of the per-pass rates — that over-weights the fast ones
        /// and would report a number no user ever experiences.
        var fps: Double?
        /// Rate of the slowest pass in the window, and of the fastest.
        var minFPS: Double?
        var maxFPS: Double?
        /// Frames the latest-frame-wins gate discarded unread, for the session.
        var dropped = 0
        var completed = 0
    }

    /// Passes kept. About a second of history on a device that manages ~10 fps,
    /// which is what makes the min/max read as "recent"; on the simulator, where
    /// one pass is ~10 s, it is instead most of the session — unavoidable when
    /// the sample rate is the thing being measured, and honest either way.
    static let window = 10

    private var recent: [Sample] = []
    private var dropped = 0
    private var completed = 0

    mutating func record(_ sample: Sample) {
        completed += 1
        recent.append(sample)
        if recent.count > Self.window { recent.removeFirst(recent.count - Self.window) }
    }

    mutating func drop() { dropped += 1 }

    var snapshot: Snapshot {
        var stats = Snapshot(latest: recent.last, dropped: dropped, completed: completed)
        // Non-positive durations are dropped rather than divided by. A monotonic
        // clock cannot go backwards, but it can report the same instant twice for
        // work shorter than its tick, and that must read as "not measurable yet"
        // rather than as an infinitely fast pipeline.
        let durations = recent.map(\.total).filter { $0 > 0 }
        guard let longest = durations.max(), let shortest = durations.min() else { return stats }
        stats.fps = Double(durations.count) / durations.reduce(0, +)
        stats.minFPS = 1 / longest
        stats.maxFPS = 1 / shortest
        return stats
    }
}

extension Duration {
    /// Seconds as a `Double`. `components` is the lossless accessor; the
    /// attosecond half matters here because a fast stage is sub-millisecond.
    var inSeconds: Double {
        Double(components.seconds) + Double(components.attoseconds) * 1e-18
    }
}
