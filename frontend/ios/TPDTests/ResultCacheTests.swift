//  ResultCacheTests.swift
//  One pass costs ~9.7 s here, so a miss that should have been a hit is a ten-second
//  stall a user feels. The cache is pure logic, checked here rather than by
//  scrubbing and hoping.

import CoreGraphics
import CoreMedia
import XCTest
@testable import TPD

@MainActor
final class ResultCacheTests: XCTestCase {
    /// Why the key is a frame index and not a raw time: a scrub reports one and the
    /// tick behind it reports another, both inside frame 30's period at 30 fps.
    /// Keyed apart, every re-visit re-runs inference and the cache never hits — and
    /// the boundary at 1.0333 s still has to separate 30 from 31.
    func testTimesInsideOneFramePeriodShareAKeyAndTheNextPeriodDoesNot() {
        let cache = ResultCache(frameRate: 30)
        XCTAssertEqual(cache.index(for: 1.0), 30)
        XCTAssertEqual(cache.index(for: 1.0166), 30, "mid-period time keyed a different frame")
        XCTAssertEqual(cache.index(for: 1.0332), 30, "end-of-period time keyed a different frame")
        XCTAssertEqual(cache.index(for: 1.04), 31)
        cache.insert(Self.result(1), at: cache.index(for: 1.0))
        XCTAssertEqual(cache.value(at: cache.index(for: 1.0166)), Self.result(1))
        XCTAssertNil(cache.value(at: cache.index(for: 1.04)),
                     "a neighbouring frame was served the previous frame's result")

        // `CMTime.seconds` is NaN for the invalid times `AVPlayer` hands out before
        // its item is ready, and `Int(nan)` traps. A track with no nominal rate
        // reports 0, which unhandled makes the whole timeline one bucket.
        XCTAssertEqual(cache.index(for: CMTime.invalid.seconds), 0)
        XCTAssertEqual(cache.index(for: -4.0), 0)
        XCTAssertEqual(cache.index(for: .infinity), 0)
        let fallback = ResultCache(frameRate: 0)
        XCTAssertEqual(fallback.frameRate, 30)
        XCTAssertNotEqual(fallback.index(for: 1.0), fallback.index(for: 2.0))
    }

    /// The read-through path the view model calls: a miss runs the analyser once and
    /// becomes a hit, every other frame stays a miss, and a hit never pays again.
    func testAMissRunsTheAnalyserOnceAndBecomesAHitWhileOtherFramesStayMisses() async throws {
        let cache = ResultCache(frameRate: 30)
        var runs = 0
        for _ in 0..<3 {
            _ = try await cache.result(at: 8) { runs += 1; return Self.result(5) }
        }
        XCTAssertEqual(runs, 1, "the analyser ran \(runs) times for one frame")
        XCTAssertEqual(cache.value(at: 8), Self.result(5))
        XCTAssertNil(cache.value(at: 9))
        XCTAssertEqual(cache.count, 1)
    }

    /// The bound is enforced, and eviction is least-recently-*used* rather than
    /// least-recently-inserted — the point being that the frame a user keeps
    /// scrubbing back to survives. Re-analysing a cached frame overwrites it.
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

    /// The headline safety property. Leaving the preview cancels the analysis in
    /// flight; if a half-finished pass could still write an entry, a later visit
    /// would be served keypoints from an inference that never ran to the end, and it
    /// would look like a hit, silently, forever. The frame must also stay
    /// analysable: a cancelled pass leaves a miss, not a poisoned key.
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

    /// `TPDResult` is `Equatable`, so a distinguishable bbox tells entries apart.
    private static func result(_ tag: CGFloat) -> TPDResult {
        TPDResult(frameSize: .zero, bbox: CGRect(x: tag, y: 0, width: 10, height: 10),
                  keypoints: [], probabilities: [1], labels: ["serve"], bestIndex: 0)
    }
}
