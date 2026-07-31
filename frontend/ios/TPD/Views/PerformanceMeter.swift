//  PerformanceMeter.swift
//  What the live loop cost, as arithmetic. No clock, no UI, no isolation: the
//  caller stamps the times and this only divides, so every number the HUD shows
//  is pinned in XCTest, not by squinting at a camera feed.

import Foundation

/// Rolling statistics over completed inference passes. A `struct` the view model
/// owns rather than an observable: nothing here ticks, so the HUD redraws exactly
/// as often as the preview and adds no work to what it measures.
struct PerformanceMeter: Equatable, Sendable {
    /// One finished pass **and the frame it was measured on**: separate writers
    /// are how the stale-overlay badge once came to name the wrong frame.
    struct Sample: Equatable, Sendable {
        /// Seconds inside `TPDInferenceEngine.predict` — stages 1, 2 and 3 plus
        /// the crop and letterbox between, one uninterruptible call.
        var model: Double
        /// Seconds inside `CIContext.createCGImage` — apart, as it is not inference.
        var raster: Double
        /// When the raster finished, on the caller's monotonic clock. **What makes
        /// the headline a rate and not a reciprocal:** two of these bound a period,
        /// idle included, and on device the idle is most of it.
        var finished: ContinuousClock.Instant
        /// Pixel dimensions of the frame the two timings above came from.
        var width: Int
        var height: Int

        /// What one frame cost, end to end. A duration, not a rate: it knows
        /// nothing about the gaps on either side of it.
        var cost: Double { model + raster }
        /// Stages run back to back and end at `finished`: a stamp, not an estimate.
        var started: ContinuousClock.Instant { finished - .seconds(cost) }
    }

    /// Everything the HUD draws, as one value published once per pass.
    struct Snapshot: Equatable, Sendable {
        var latest: Sample?
        /// **Wall-clock** passes per second: whole periods ÷ the span from the
        /// oldest pass's start to the newest one's, so waiting for a frame counts
        /// against it and this is how often an overlay really appears. `1 / cost`
        /// is a larger number the panel prints as a cost instead. `nil` — never
        /// `0`, never infinity — until two starts bound a period to divide by.
        var fps: Double?
        /// The two halves of that division, printed beside it: N samples, N-1 periods.
        var windowPeriods = 0
        var windowElapsed: Double?
        /// The bracket around `latest.cost`, in the same unit, comparable by eye.
        var cheapest: Double?
        var dearest: Double?
        /// Frames the latest-frame-wins gate discarded unread, for the session.
        var dropped = 0
        var completed = 0
    }

    /// Passes kept: about a second of history at ~10 fps, which is what makes the
    /// range read as "recent"; on the simulator it is most of the session.
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
        guard let oldest = recent.first, let newest = recent.last else { return stats }
        stats.cheapest = recent.map(\.cost).min()
        stats.dearest = recent.map(\.cost).max()
        // Start-to-start, so the divisor is whole periods: measuring to `finished`
        // spans N-1 periods plus the newest pass's own cost yet divides by N, and at
        // N == 1 degrades to the 1/cost reciprocal this row exists to stop printing.
        stats.windowPeriods = recent.count - 1
        let elapsed = oldest.started.duration(to: newest.started).inSeconds
        guard stats.windowPeriods > 0, elapsed > 0 else { return stats }
        stats.windowElapsed = elapsed
        stats.fps = Double(stats.windowPeriods) / elapsed
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
