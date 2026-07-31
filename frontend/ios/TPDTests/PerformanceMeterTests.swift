//  PerformanceMeterTests.swift
//  The HUD's arithmetic, fed known timelines. A rate readout earns screen space
//  only if it is right at the edges: before the first pass lands, while the
//  pipeline is stalled, and — the case this file was rewritten for — while the
//  loop sits idle between passes.

import XCTest
@testable import TPD

final class PerformanceMeterTests: XCTestCase {
    /// Nothing has finished, so there is no rate — not zero, which claims a
    /// measurement, and not the infinity `1 / 0` hands `String(format:)`.
    func testWithNoPassesYetThereIsNoRateAtAllRatherThanZeroOrInfinity() {
        var meter = PerformanceMeter()
        var stats = meter.snapshot
        XCTAssertNil(stats.fps, "a rate was reported before any pass finished")
        XCTAssertNil(stats.cheapest); XCTAssertNil(stats.dearest); XCTAssertNil(stats.latest)
        XCTAssertNil(stats.windowElapsed)
        XCTAssertEqual(stats.completed, 0); XCTAssertEqual(stats.dropped, 0)
        // The gate can discard hundreds of frames before the first pass returns,
        // and none of them say anything about throughput.
        for _ in 0..<300 { meter.drop() }
        stats = meter.snapshot
        XCTAssertEqual(stats.dropped, 300)
        XCTAssertNil(stats.fps, "frames dropped were mistaken for frames processed")
    }

    /// **The headline is wall-clock.** A device pass is ~30 ms of model and ~5 ms
    /// of raster, then the loop waits ~20 ms for the next frame. Passes ÷ the sum
    /// of their durations reads 29 fps; the overlay in fact redraws about 18 times
    /// a second, and 18 is what the pill must say.
    func testTheRateCountsTheIdleBetweenPassesAndNotOnlyTheirDurations() throws {
        var timeline = Timeline()
        var meter = PerformanceMeter()
        for _ in 0..<10 { meter.record(timeline.pass(model: 0.030, raster: 0.005, idle: 0.020)) }
        let stats = meter.snapshot
        // 10 passes span 9 whole 55 ms periods plus the last pass's own 35 ms.
        XCTAssertEqual(try XCTUnwrap(stats.windowElapsed), 0.530, accuracy: 1e-6)
        XCTAssertEqual(try XCTUnwrap(stats.fps), 18.87, accuracy: 0.01,
                       "the idle between passes was dropped, inflating the rate")
        // 1 / (model + raster) is the ceiling the durations alone would give, and
        // the real rate is under it — both halves of the old bug in one bound.
        XCTAssertLessThan(try XCTUnwrap(stats.fps), 1 / 0.035)
        XCTAssertEqual(try XCTUnwrap(stats.latest?.cost), 0.035, accuracy: 1e-9)
        XCTAssertEqual(try XCTUnwrap(stats.dearest), 0.035, accuracy: 1e-9)
        XCTAssertEqual(try XCTUnwrap(stats.latest?.model), 0.030, accuracy: 1e-9)
    }

    /// One pass is enough to report: the span is that pass, so the rate is its own.
    func testOnePassReportsItsOwnRateAndItsOwnCost() throws {
        var timeline = Timeline()
        var meter = PerformanceMeter()
        meter.record(timeline.pass(model: 9.7, raster: 0.3))
        let stats = meter.snapshot
        XCTAssertEqual(try XCTUnwrap(stats.fps), 0.1, accuracy: 1e-9)
        XCTAssertEqual(try XCTUnwrap(stats.windowElapsed), 10.0, accuracy: 1e-9)
        XCTAssertEqual(stats.windowPasses, 1); XCTAssertEqual(stats.completed, 1)
        // The frame the timings came from travels with them; the panel prints the
        // two side by side and must never pair one pass's cost with another's size.
        XCTAssertEqual(stats.latest?.width, 608); XCTAssertEqual(stats.latest?.height, 1080)
        XCTAssertEqual(try XCTUnwrap(stats.latest?.cost), 10.0, accuracy: 1e-9)
    }

    /// Passes ÷ elapsed, not the mean of the per-pass rates: averaging the rates
    /// would claim 8 fps for four frames that took 0.8 s.
    func testTheRateIsPassesOverElapsedTimeNotTheAverageOfThePerPassRates() throws {
        var timeline = Timeline()
        var meter = PerformanceMeter()
        for seconds in [0.1, 0.1, 0.1, 0.5] { meter.record(timeline.pass(model: seconds)) }
        let stats = meter.snapshot
        XCTAssertEqual(try XCTUnwrap(stats.fps), 5.0, accuracy: 1e-6,
                       "4 passes in 0.8 s is 5 fps; the mean of the rates would be 8")
        XCTAssertEqual(try XCTUnwrap(stats.cheapest), 0.1, accuracy: 1e-9)
        XCTAssertEqual(try XCTUnwrap(stats.dearest), 0.5, accuracy: 1e-9)
    }

    /// A stall is the case the panel is opened for: it must drag the headline down
    /// *and* stay visible as the dearest frame, or it vanishes into the average.
    func testALongStallDragsTheRateDownAndSurvivesAsTheDearestFrame() throws {
        var timeline = Timeline()
        var meter = PerformanceMeter()
        for _ in 0..<9 { meter.record(timeline.pass(model: 0.1)) }
        meter.record(timeline.pass(model: 30.0))
        let stats = meter.snapshot
        XCTAssertEqual(try XCTUnwrap(stats.fps), 10.0 / 30.9, accuracy: 1e-6)
        XCTAssertEqual(try XCTUnwrap(stats.dearest), 30.0, accuracy: 1e-9,
                       "the 30 s pass was averaged away instead of shown")
        XCTAssertEqual(try XCTUnwrap(stats.cheapest), 0.1, accuracy: 1e-9)
    }

    /// "Recent" has to mean recent: once the window rolls past the stall the
    /// readout describes the work happening now.
    func testTheWindowForgetsOlderPassesSoTheRateFollowsRecentWork() throws {
        var timeline = Timeline()
        var meter = PerformanceMeter()
        meter.record(timeline.pass(model: 30.0))
        for _ in 0..<PerformanceMeter.window { meter.record(timeline.pass(model: 0.2)) }
        let stats = meter.snapshot
        XCTAssertEqual(try XCTUnwrap(stats.fps), 5.0, accuracy: 1e-6,
                       "a pass older than the window still counted against the rate")
        XCTAssertEqual(try XCTUnwrap(stats.dearest), 0.2, accuracy: 1e-9)
        XCTAssertEqual(stats.windowPasses, PerformanceMeter.window)
        // `completed` is the session total and is deliberately *not* windowed.
        XCTAssertEqual(stats.completed, PerformanceMeter.window + 1)
    }

    /// The other division by zero: a span shorter than the clock's tick reads as
    /// "not measurable yet", never as an infinitely fast pipeline.
    func testAPassTooShortToMeasureCannotProduceAnInfiniteRate() throws {
        var timeline = Timeline()
        var meter = PerformanceMeter()
        meter.record(timeline.pass(model: 0, raster: 0))
        var stats = meter.snapshot
        XCTAssertNil(stats.fps, "a zero-length pass was reported as an infinite rate")
        XCTAssertNil(stats.windowElapsed)
        XCTAssertEqual(stats.completed, 1, "the pass itself still happened")
        meter.record(timeline.pass(model: 0.25))
        stats = meter.snapshot
        XCTAssertEqual(try XCTUnwrap(stats.fps), 8.0, accuracy: 1e-6,
                       "two passes across 0.25 s of wall clock is 8 fps")
    }

    /// Two populations, named separately on the panel: what the latest-frame-wins
    /// gate threw away, and what actually ran.
    func testDroppedFramesAreCountedApartFromCompletedPasses() throws {
        var timeline = Timeline()
        var meter = PerformanceMeter()
        for _ in 0..<2 { meter.record(timeline.pass(model: 0.5)) }
        for _ in 0..<58 { meter.drop() }
        let stats = meter.snapshot
        XCTAssertEqual(stats.dropped, 58); XCTAssertEqual(stats.completed, 2)
        XCTAssertEqual(try XCTUnwrap(stats.fps), 2.0, accuracy: 1e-6,
                       "dropped frames leaked into the throughput calculation")
    }

    /// The live loop's timing without a camera: a pass runs for `model + raster`,
    /// then the loop waits `idle` for the next frame. Every stamp comes off one
    /// monotonic origin, so the meter sees what `InferenceWorker.process` would
    /// hand it. 608x1080 is the bundled clip's real shape.
    private struct Timeline {
        private let epoch = ContinuousClock.now
        private var elapsed = Duration.zero

        mutating func pass(model: Double, raster: Double = 0,
                           idle: Double = 0) -> PerformanceMeter.Sample {
            elapsed += .seconds(model + raster)
            defer { elapsed += .seconds(idle) }
            return .init(model: model, raster: raster, finished: epoch + elapsed,
                         width: 608, height: 1080)
        }
    }
}
