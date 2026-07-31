//  PerformanceMeterTests.swift
//  The HUD's arithmetic, fed known timelines. A rate readout earns screen space
//  only if it is right at the edges: before the first pass lands, while the
//  pipeline is stalled, and — the case this file exists for — while the loop
//  sits idle between passes. The rate is whole periods ÷ elapsed, measured start
//  of pass to start of pass: N samples bound N-1 periods, and one bounds none.

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
        // The gate discards frames long before the first pass returns, and none
        // of them say anything about throughput.
        for _ in 0..<300 { meter.drop() }
        stats = meter.snapshot
        XCTAssertEqual(stats.dropped, 300)
        XCTAssertNil(stats.fps, "frames dropped were mistaken for frames processed")
    }

    /// **The headline is wall-clock.** A device pass is ~30 ms of model and ~5 ms of
    /// raster, then the loop waits ~20 ms: a 55 ms period. Passes ÷ their durations
    /// read 29 fps; the overlay redraws 18 times a second, and 18 is what must show.
    func testTheRateCountsTheIdleBetweenPassesAndNotOnlyTheirDurations() throws {
        var timeline = Timeline(); var meter = PerformanceMeter()
        for _ in 0..<10 { meter.record(timeline.pass(model: 0.030, raster: 0.005, idle: 0.020)) }
        let stats = meter.snapshot
        // Ten starts bound nine whole periods: 9 × 55 ms. Running the span on to
        // the last pass's *end* adds its 35 ms, divides 10 by 0.530 and reads 18.87.
        XCTAssertEqual(stats.windowPeriods, 9)
        XCTAssertEqual(try XCTUnwrap(stats.windowElapsed), 0.495, accuracy: 1e-6)
        XCTAssertEqual(try XCTUnwrap(stats.fps), 18.18, accuracy: 0.01,
                       "the idle between passes was dropped, inflating the rate")
        // 1 / (model + raster) is the ceiling the durations alone would give, and
        // the real rate is under it — both halves of the old bug in one bound.
        XCTAssertLessThan(try XCTUnwrap(stats.fps), 1 / 0.035)
        XCTAssertEqual(try XCTUnwrap(stats.latest?.cost), 0.035, accuracy: 1e-9)
        XCTAssertEqual(try XCTUnwrap(stats.dearest), 0.035, accuracy: 1e-9)
        XCTAssertEqual(try XCTUnwrap(stats.latest?.model), 0.030, accuracy: 1e-9)
    }

    /// One sample is a cost, not a rate: it has a start and nothing to measure to.
    /// It used to report 1 ÷ its own 10 s — the reciprocal this readout exists to
    /// replace, printed under a wall-clock label.
    func testASinglePassReportsItsCostButNoRateAtAll() throws {
        var timeline = Timeline(); var meter = PerformanceMeter()
        meter.record(timeline.pass(model: 9.7, raster: 0.3))
        let stats = meter.snapshot
        XCTAssertNil(stats.fps, "one sample was reported at its own reciprocal, 0.1 fps")
        XCTAssertNil(stats.windowElapsed)
        XCTAssertEqual(stats.windowPeriods, 0); XCTAssertEqual(stats.completed, 1)
        // The frame the timings came from travels with them; the panel prints the
        // two side by side and must never pair one pass's cost with another's size.
        XCTAssertEqual(stats.latest?.width, 608); XCTAssertEqual(stats.latest?.height, 1080)
        XCTAssertEqual(try XCTUnwrap(stats.latest?.cost), 10.0, accuracy: 1e-9)
    }

    /// Periods ÷ elapsed, not the mean of the per-pass rates: that mean lets four
    /// quick passes outvote the slow one and claim 8.4 fps for 0.8 s of work.
    func testTheRateIsPeriodsOverElapsedTimeNotTheAverageOfThePerPassRates() throws {
        var timeline = Timeline(); var meter = PerformanceMeter()
        for seconds in [0.1, 0.1, 0.1, 0.5, 0.1] { meter.record(timeline.pass(model: seconds)) }
        let stats = meter.snapshot
        XCTAssertEqual(try XCTUnwrap(stats.fps), 5.0, accuracy: 1e-6,
                       "4 periods across 0.8 s is 5 fps; the mean of the rates would be 8.4")
        XCTAssertEqual(try XCTUnwrap(stats.cheapest), 0.1, accuracy: 1e-9)
        XCTAssertEqual(try XCTUnwrap(stats.dearest), 0.5, accuracy: 1e-9)
    }

    /// A stall is the case the panel is opened for: it must drag the headline down
    /// *and* stay visible as the dearest frame, or it vanishes into the average. It
    /// bites when the pass after it starts, i.e. once it is a period, not a pass.
    func testALongStallDragsTheRateDownAndSurvivesAsTheDearestFrame() throws {
        var timeline = Timeline(); var meter = PerformanceMeter()
        for _ in 0..<8 { meter.record(timeline.pass(model: 0.1)) }
        meter.record(timeline.pass(model: 30.0))
        meter.record(timeline.pass(model: 0.1))
        let stats = meter.snapshot
        XCTAssertEqual(try XCTUnwrap(stats.fps), 9 / 30.8, accuracy: 1e-6,
                       "the 30 s period was not counted against the rate")
        XCTAssertEqual(try XCTUnwrap(stats.dearest), 30.0, accuracy: 1e-9,
                       "the 30 s pass was averaged away instead of shown")
        XCTAssertEqual(try XCTUnwrap(stats.cheapest), 0.1, accuracy: 1e-9)
    }

    /// "Recent" has to mean recent: once the window rolls past the stall the
    /// readout describes the work happening now.
    func testTheWindowForgetsOlderPassesSoTheRateFollowsRecentWork() throws {
        var timeline = Timeline(); var meter = PerformanceMeter()
        meter.record(timeline.pass(model: 30.0))
        for _ in 0..<PerformanceMeter.window { meter.record(timeline.pass(model: 0.2)) }
        let stats = meter.snapshot
        XCTAssertEqual(try XCTUnwrap(stats.fps), 5.0, accuracy: 1e-6,
                       "a pass older than the window still counted against the rate")
        XCTAssertEqual(try XCTUnwrap(stats.dearest), 0.2, accuracy: 1e-9)
        XCTAssertEqual(stats.windowPeriods, PerformanceMeter.window - 1)
        // `completed` is the session total and is deliberately *not* windowed.
        XCTAssertEqual(stats.completed, PerformanceMeter.window + 1)
    }

    /// The other division by zero: a span shorter than the clock's tick reads as
    /// "not measurable yet", never as an infinitely fast pipeline.
    func testStartsTooCloseTogetherToMeasureCannotProduceAnInfiniteRate() throws {
        var timeline = Timeline(); var meter = PerformanceMeter()
        meter.record(timeline.pass(model: 0, raster: 0))
        var stats = meter.snapshot
        XCTAssertNil(stats.fps, "a zero-length pass was reported as an infinite rate")
        XCTAssertNil(stats.windowElapsed)
        XCTAssertEqual(stats.completed, 1, "the pass itself still happened")
        // The second pass begins in the instant the empty one ended: two starts,
        // no span between them, a period the clock cannot see.
        meter.record(timeline.pass(model: 0.25))
        stats = meter.snapshot
        XCTAssertNil(stats.fps, "two passes that began together read as an infinite rate")
        meter.record(timeline.pass(model: 0.25))
        stats = meter.snapshot
        XCTAssertEqual(try XCTUnwrap(stats.fps), 8.0, accuracy: 1e-6,
                       "two periods across 0.25 s of wall clock is 8 fps")
    }

    /// Two populations, named separately on the panel: what the latest-frame-wins
    /// gate threw away, and what actually ran.
    func testDroppedFramesAreCountedApartFromCompletedPasses() throws {
        var timeline = Timeline(); var meter = PerformanceMeter()
        for _ in 0..<2 { meter.record(timeline.pass(model: 0.5)) }
        for _ in 0..<58 { meter.drop() }
        let stats = meter.snapshot
        XCTAssertEqual(stats.dropped, 58); XCTAssertEqual(stats.completed, 2)
        XCTAssertEqual(try XCTUnwrap(stats.fps), 2.0, accuracy: 1e-6,
                       "dropped frames leaked into the throughput calculation")
    }

    /// The live loop's timing without a camera: a pass runs for `model + raster`,
    /// then the loop waits `idle` for the next frame. Every stamp comes off one
    /// monotonic origin, as `InferenceWorker.process` hands them over; 608x1080
    /// is the bundled clip's shape.
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
