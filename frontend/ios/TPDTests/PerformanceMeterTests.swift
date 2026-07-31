//  PerformanceMeterTests.swift
//  The HUD's arithmetic, fed known sequences. A rate readout is only worth
//  screen space if it is right at the edges: before the first pass lands, on the
//  pass after that, and while the pipeline is stalled — which on this simulator
//  is most of the time.

import XCTest
@testable import TPD

final class PerformanceMeterTests: XCTestCase {
    /// The obvious failure this file exists to prevent. Nothing has finished, so
    /// there is no rate — not zero, which claims a measurement, and not infinity,
    /// which is what `1 / 0` hands a `String(format:)` that then prints "inf".
    func testWithNoPassesYetThereIsNoRateAtAllRatherThanZeroOrInfinity() {
        var meter = PerformanceMeter()
        var stats = meter.snapshot
        XCTAssertNil(stats.fps, "a rate was reported before any pass finished")
        XCTAssertNil(stats.minFPS); XCTAssertNil(stats.maxFPS); XCTAssertNil(stats.latest)
        XCTAssertEqual(stats.completed, 0); XCTAssertEqual(stats.dropped, 0)

        // Drops alone still teach nothing about throughput: the gate can discard
        // hundreds of frames before the first pass ever returns.
        for _ in 0..<300 { meter.drop() }
        stats = meter.snapshot
        XCTAssertEqual(stats.dropped, 300)
        XCTAssertNil(stats.fps, "frames dropped were mistaken for frames processed")
        XCTAssertEqual(stats.completed, 0)
    }

    /// One pass is enough to report, and the window rules must not need two.
    /// With a single sample the rate, the minimum and the maximum are all it.
    func testOnePassReportsItsOwnRateAsTheRateTheMinimumAndTheMaximum() throws {
        var meter = PerformanceMeter()
        meter.record(Self.pass(model: 9.7, raster: 0.3))
        let stats = meter.snapshot

        XCTAssertEqual(try XCTUnwrap(stats.fps), 0.1, accuracy: 1e-9)
        XCTAssertEqual(try XCTUnwrap(stats.minFPS), 0.1, accuracy: 1e-9)
        XCTAssertEqual(try XCTUnwrap(stats.maxFPS), 0.1, accuracy: 1e-9)
        XCTAssertEqual(stats.completed, 1)
        // The frame the timings came from travels with them; the panel prints the
        // two side by side and must never pair one pass's cost with another's size.
        XCTAssertEqual(stats.latest?.width, 608); XCTAssertEqual(stats.latest?.height, 1080)
        XCTAssertEqual(try XCTUnwrap(stats.latest?.total), 10.0, accuracy: 1e-9)
    }

    /// Throughput is passes ÷ elapsed, not the mean of the per-pass rates. The
    /// two disagree whenever the passes differ, and averaging the rates flatters
    /// the pipeline: here it would claim 8 fps for four frames that took 0.8 s,
    /// a number no user could ever have observed.
    func testTheRateIsPassesOverElapsedTimeNotTheAverageOfThePerPassRates() throws {
        var meter = PerformanceMeter()
        for seconds in [0.1, 0.1, 0.1, 0.5] { meter.record(Self.pass(model: seconds)) }
        let stats = meter.snapshot

        XCTAssertEqual(try XCTUnwrap(stats.fps), 5.0, accuracy: 1e-9,
                       "4 passes in 0.8 s is 5 fps; the mean of the rates would be 8")
        XCTAssertEqual(try XCTUnwrap(stats.minFPS), 2.0, accuracy: 1e-9)
        XCTAssertEqual(try XCTUnwrap(stats.maxFPS), 10.0, accuracy: 1e-9)
    }

    /// A stall is the case the panel is opened for. It has to drag the headline
    /// rate down *and* stay visible as the minimum, or the one bad pass vanishes
    /// into the average and the HUD says everything is fine.
    func testALongStallDragsTheRateDownAndSurvivesAsTheMinimum() throws {
        var meter = PerformanceMeter()
        for _ in 0..<9 { meter.record(Self.pass(model: 0.1)) }
        meter.record(Self.pass(model: 30.0))
        let stats = meter.snapshot

        XCTAssertEqual(try XCTUnwrap(stats.fps), 10.0 / 30.9, accuracy: 1e-9)
        XCTAssertEqual(try XCTUnwrap(stats.minFPS), 1.0 / 30.0, accuracy: 1e-9,
                       "the 30 s pass was averaged away instead of shown")
        XCTAssertEqual(try XCTUnwrap(stats.maxFPS), 10.0, accuracy: 1e-9)
    }

    /// "Recent" has to mean recent: once the window has rolled past the stall the
    /// readout must describe the work happening now, not the worst thing that ever
    /// happened. Every sample here is fast, so the rate is that of a fast pass.
    func testTheWindowForgetsOlderPassesSoTheRateFollowsRecentWork() throws {
        var meter = PerformanceMeter()
        meter.record(Self.pass(model: 30.0))
        for _ in 0..<PerformanceMeter.window { meter.record(Self.pass(model: 0.2)) }
        let stats = meter.snapshot

        XCTAssertEqual(try XCTUnwrap(stats.fps), 5.0, accuracy: 1e-9,
                       "a pass older than the window still counted against the rate")
        XCTAssertEqual(try XCTUnwrap(stats.minFPS), 5.0, accuracy: 1e-9)
        // `completed` is the session total and is deliberately *not* windowed.
        XCTAssertEqual(stats.completed, PerformanceMeter.window + 1)
    }

    /// The other division by zero: a pass shorter than the clock's tick. It reads
    /// as "not measurable yet", never as an infinitely fast pipeline, and a real
    /// pass alongside it is still measured.
    func testAPassTooShortToMeasureCannotProduceAnInfiniteRate() throws {
        var meter = PerformanceMeter()
        meter.record(Self.pass(model: 0, raster: 0))
        var stats = meter.snapshot
        XCTAssertNil(stats.fps, "a zero-length pass was reported as an infinite rate")
        XCTAssertNil(stats.maxFPS)
        XCTAssertEqual(stats.completed, 1, "the pass itself still happened")
        XCTAssertEqual(stats.latest?.total, 0)

        meter.record(Self.pass(model: 0.25))
        stats = meter.snapshot
        XCTAssertEqual(try XCTUnwrap(stats.fps), 4.0, accuracy: 1e-9,
                       "the unmeasurable pass was counted into elapsed time")
        XCTAssertEqual(try XCTUnwrap(stats.minFPS), 4.0, accuracy: 1e-9)
    }

    /// Drops and passes are different populations and the panel names them
    /// separately: one is what the latest-frame-wins gate threw away, the other
    /// is what actually ran.
    func testDroppedFramesAreCountedApartFromCompletedPasses() throws {
        var meter = PerformanceMeter()
        for _ in 0..<2 { meter.record(Self.pass(model: 0.5)) }
        for _ in 0..<58 { meter.drop() }
        let stats = meter.snapshot

        XCTAssertEqual(stats.dropped, 58); XCTAssertEqual(stats.completed, 2)
        XCTAssertEqual(try XCTUnwrap(stats.fps), 2.0, accuracy: 1e-9,
                       "dropped frames leaked into the throughput calculation")
    }

    /// A 608x1080 frame, the shape the simulator's bundled clip actually yields.
    private static func pass(model: Double, raster: Double = 0) -> PerformanceMeter.Sample {
        PerformanceMeter.Sample(model: model, raster: raster, width: 608, height: 1080)
    }
}
