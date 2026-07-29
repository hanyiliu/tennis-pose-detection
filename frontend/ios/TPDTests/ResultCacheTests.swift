//  ResultCacheTests.swift
//  A miss that should have been a hit is a ten-second stall a user feels. Pure
//  logic, checked here rather than by scrubbing and hoping.

import CoreGraphics
import CoreMedia
import XCTest
@testable import TPD
@MainActor
final class ResultCacheTests: XCTestCase {
    /// Why a frame index, not a raw time: a scrub and the tick behind it report
    /// different times inside frame 30's period at 30 fps.
    func testTimesInsideOneFramePeriodShareAKeyAndTheNextPeriodDoesNot() {
        let cache = ResultCache(frameRate: 30)
        XCTAssertEqual(cache.index(for: 1.0), 30)
        XCTAssertEqual(cache.index(for: 1.0166), 30, "mid-period time keyed another frame")
        XCTAssertEqual(cache.index(for: 1.0332), 30, "period-end time keyed another frame")
        XCTAssertEqual(cache.index(for: 1.04), 31)
        cache.insert(Self.result(1), at: cache.index(for: 1.0))
        XCTAssertEqual(cache.value(at: cache.index(for: 1.0166)), Self.result(1))
        XCTAssertNil(cache.value(at: cache.index(for: 1.04)),
                     "a neighbouring frame was served the previous frame's result")
        // `CMTime.seconds` is NaN before the item is ready and `Int(nan)` traps; a
        // track with no nominal rate reports 0, one bucket for the whole clip.
        XCTAssertEqual(cache.index(for: CMTime.invalid.seconds), 0)
        XCTAssertEqual(cache.index(for: -4.0), 0); XCTAssertEqual(cache.index(for: .infinity), 0)
        let fallback = ResultCache(frameRate: 0)
        XCTAssertEqual(fallback.frameRate, 30)
        XCTAssertNotEqual(fallback.index(for: 1.0), fallback.index(for: 2.0))
    }

    /// A miss runs the analyser once and becomes a hit; a hit never pays again.
    func testAMissRunsTheAnalyserOnceAndBecomesAHitWhileOtherFramesStayMisses() async throws {
        let cache = ResultCache(frameRate: 30)
        var runs = 0
        for _ in 0..<3 { _ = try await cache.result(at: 8) { runs += 1; return Self.result(5) } }
        XCTAssertEqual(runs, 1, "the analyser ran \(runs) times for one frame")
        XCTAssertEqual(cache.value(at: 8), Self.result(5))
        XCTAssertNil(cache.value(at: 9)); XCTAssertEqual(cache.count, 1)
    }

    /// Eviction is least-recently-*used*, so the frame a user keeps scrubbing back
    /// to survives; overwrites cost no capacity.
    func testTheBoundEvictsTheLeastRecentlyUsedEntryAndOverwritesAreFree() {
        let cache = ResultCache(frameRate: 30, capacity: 3)
        for index in 0..<5 { cache.insert(Self.result(CGFloat(index)), at: index) }
        XCTAssertEqual(cache.count, 3, "the cache grew past its capacity")
        XCTAssertNil(cache.value(at: 0), "the oldest entry survived the bound")
        XCTAssertEqual(cache.value(at: 2), Self.result(2))   // touches frame 2
        cache.insert(Self.result(9), at: 9)
        XCTAssertEqual(cache.value(at: 2), Self.result(2), "the re-visited frame was evicted")
        XCTAssertNil(cache.value(at: 3), "the least recently used frame survived")
        cache.insert(Self.result(8), at: 9)
        XCTAssertEqual(cache.count, 3, "an overwrite consumed capacity")
        XCTAssertEqual(cache.value(at: 9), Self.result(8))
    }
    /// The headline safety property: a recorded half-pass would serve keypoints
    /// from an inference that never ended, silently, forever.
    func testACancelledAnalysisLeavesAMissBehindRatherThanAPartialEntry() async throws {
        let cache = ResultCache(frameRate: 30)
        do {
            _ = try await cache.result(at: 12) { throw CancellationError() }
            XCTFail("a cancelled analysis returned a result")
        } catch {
            XCTAssertTrue(error is CancellationError, "expected CancellationError, got \(error)")
        }
        XCTAssertNil(cache.value(at: 12), "a cancelled analysis poisoned the cache")
        XCTAssertEqual(cache.count, 0)
        let served = try await cache.result(at: 12) { Self.result(4) }
        XCTAssertEqual(served, Self.result(4))
        XCTAssertEqual(cache.value(at: 12), Self.result(4))
    }
    /// The badge is the screen's only promise about a stale overlay, so the time it
    /// names must belong to the result on screen. Walked as a user does — analyse
    /// 0.00, analyse 2.00, back onto the cached 0.00, on to an uncached 4.00 — since
    /// the hit in the middle kept the result and dropped its time, so the last step
    /// drew 0.00's overlay under a badge reading "from 2.00 s".
    func testTheReportedSourceTimeAlwaysMatchesTheResultOnScreen() {
        let cache = ResultCache(frameRate: 30)
        var held: VideoPreviewModel.Held?
        var overlay = VideoPreviewModel.Overlay.none
        func visit(_ seconds: Double, analysing: Bool = false) {
            let index = cache.index(for: seconds)
            if analysing { cache.insert(Self.result(CGFloat(seconds)), at: index) }
            (held, overlay) = VideoPreviewModel.resolve(frameAt: seconds,
                                                       cached: cache.value(at: index), held: held)
            if case .stale(let at) = overlay {
                XCTAssertEqual(at, held?.time, "the badge named a frame the overlay is not from")
            }
            if let held {   // the standing invariant, checked after every step
                XCTAssertEqual(cache.value(at: cache.index(for: held.time)), held.result,
                               "the overlay held is not the one measured at that time")
            }
        }
        visit(0.00, analysing: true)                 // miss, then the pass lands
        XCTAssertEqual(overlay, .current)
        visit(2.00, analysing: true)
        visit(0.00)                                  // hit: 0.00's result is back
        XCTAssertEqual(overlay, .current)
        XCTAssertEqual(held, VideoPreviewModel.Held(result: Self.result(0), time: 0))
        visit(4.00)                                  // miss: whose overlay is this?
        XCTAssertEqual(overlay, .stale(0.00), "the badge named the frame before the hit")
        XCTAssertEqual(held?.result, Self.result(0))
    }

    /// `TPDResult` is `Equatable`, so a distinguishable bbox tells entries apart.
    private static func result(_ tag: CGFloat) -> TPDResult {
        TPDResult(frameSize: .zero, bbox: CGRect(x: tag, y: 0, width: 10, height: 10),
                  keypoints: [], probabilities: [1], labels: ["serve"], bestIndex: 0)
    }
}
