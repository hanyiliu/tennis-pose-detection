//  PerformanceMeter.swift
//  What the live loop cost, as arithmetic. No clock, no UI, no isolation: the
//  caller stamps the times and this only divides, so every number the HUD shows
//  is pinned down in XCTest rather than by squinting at a camera feed.

import Foundation

/// Rolling statistics over completed inference passes. A `struct` the view model
/// owns rather than an observable of its own: nothing here ticks, so the HUD
/// redraws exactly as often as the preview and adds no work to what it measures.
struct PerformanceMeter: Equatable, Sendable {
    /// One finished pass **and the frame it was measured on**, together: the panel
    /// prints the dimensions next to the timings, and separate writers are how the
    /// stale-overlay badge once came to name the wrong frame.
    struct Sample: Equatable, Sendable {
        /// Seconds inside `TPDInferenceEngine.predict` — stages 1, 2 and 3 plus
        /// the crop and letterbox between, one uninterruptible call.
        var model: Double
        /// Seconds inside `CIContext.createCGImage`, the display raster — kept
        /// apart because it is not inference.
        var raster: Double
        /// When the raster finished, on the caller's monotonic clock. **This is
        /// what makes the headline a rate rather than a reciprocal:** two of these
        /// bracket the idle between passes, which on device is most of the period.
        var finished: ContinuousClock.Instant
        /// Pixel dimensions of the frame the two timings above came from.
        var width: Int
        var height: Int

        /// What one frame cost, end to end. A duration, not a rate: it knows
        /// nothing about the gaps on either side of it.
        var cost: Double { model + raster }
        /// The stages run back to back and end at `finished`, so this is a real
        /// start stamp, not an estimate.
        var started: ContinuousClock.Instant { finished - .seconds(cost) }
    }

    /// Everything the HUD draws, as one value published once per pass.
    struct Snapshot: Equatable, Sendable {
        var latest: Sample?
        /// **Wall-clock** passes per second: passes ÷ the span from the start of
        /// the window's oldest pass to the end of its newest. Waiting for a frame
        /// counts against it, so this is how often an overlay really appears;
        /// `1 / cost` is a larger number the panel prints as a cost instead.
        /// `nil`, never `0` and never infinity, until there is something to divide.
        var fps: Double?
        /// The two halves of that division, printed beside it to be checked.
        var windowPasses = 0
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
        stats.windowPasses = recent.count
        stats.cheapest = recent.map(\.cost).min()
        stats.dearest = recent.map(\.cost).max()
        // A monotonic clock can report the same instant twice for work shorter
        // than its tick; that reads as "not measurable", not as infinitely fast.
        let elapsed = oldest.started.duration(to: newest.finished).inSeconds
        guard elapsed > 0 else { return stats }
        stats.windowElapsed = elapsed
        stats.fps = Double(recent.count) / elapsed
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
