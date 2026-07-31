//  PreloadCoordinatorTests.swift
//  The sweep costs ~9.7 s a frame, so every property that matters about it — what order it runs
//  in, when it stops, what it claims to have done — is checked here as pure scheduling over an
//  injected step, rather than by picking a clip and watching for five minutes.

import CoreGraphics
import Foundation
import XCTest
@testable import TPD

/// What the sweep asked for, in order. `@unchecked Sendable` over a lock: the step runs off the
/// main actor and the test reads it on main.
private final class Issued: @unchecked Sendable {
    private let lock = NSLock()
    private var indices: [Int] = []
    var value: [Int] { lock.withLock { indices } }
    var count: Int { lock.withLock { indices.count } }
    func record(_ index: Int) { lock.withLock { indices.append(index) } }
}

/// A delay that **ignores cancellation**, standing in for the one uninterruptible `predict` call
/// in the middle of every real pass. `Task.sleep` would throw the instant the sweep was
/// cancelled and so could never exercise the checks that run after the await returns.
private func stubbornDelay(_ seconds: Double) async {
    await withCheckedContinuation { continuation in
        DispatchQueue.global().asyncAfter(deadline: .now() + seconds) { continuation.resume() }
    }
}

/// A distinguishable result; `TPDResult` is `Equatable`, so the bbox tells one apart.
private func stub(_ tag: CGFloat) -> TPDResult {
    TPDResult(frameSize: .zero, bbox: CGRect(x: tag, y: 0, width: 10, height: 10),
              keypoints: [], probabilities: [1], labels: ["serve"], bestIndex: 0)
}

@MainActor
final class PreloadCoordinatorTests: XCTestCase {
    /// The headline. A front-to-back sweep is the wrong order the moment the user scrubs to the
    /// middle: they would sit through every frame they skipped. The frame on screen goes first,
    /// then outwards — forward before back, because the transport only runs that way.
    func testTheFrameOnScreenGoesFirstAndTheSweepWorksOutwardsForwardBeforeBack() {
        var plan = PreloadPlan(frameCount: 51, cadence: 5, capacity: 600), order: [Int] = []
        XCTAssertEqual(plan.pending, Set(stride(from: 0, to: 51, by: 5)))
        for _ in 0..<6 { order.append(plan.take(nearest: 25) ?? -1) }  // scrubbed to the middle
        XCTAssertEqual(order, [25, 30, 20, 35, 40, 15],
                       "a front-to-back sweep would have started at 0 and made the user wait for "
                           + "every frame they skipped")
        // ...and it re-aims the moment they scrub elsewhere, rather than finishing its ring.
        XCTAssertEqual(plan.take(nearest: 0), 0)
        XCTAssertEqual(plan.take(nearest: 50), 50)
    }

    /// Preloading must not push the cache past its bound: a sweep that outgrew it would spend
    /// ~9.7 s a frame evicting its own answers, and a user scrubbing back would wait again for a
    /// frame they had already paid for. The cadence absorbs the clip's length instead.
    func testTheCadenceRisesUntilThePlanFitsTheCacheSoASweepCannotEvictItsOwnAnswers() {
        let clip = PreloadPlan(frameCount: 150, cadence: 5, capacity: 600)
        XCTAssertEqual([clip.cadence, clip.total, clip.tolerance], [5, 30, 2])
        // 20 minutes at 30 fps: 7200 frames at the default cadence into a 600-entry bound.
        let long = PreloadPlan(frameCount: 36_000, cadence: 5, capacity: 600)
        XCTAssertLessThanOrEqual(long.total, 600,
                                 "the plan scheduled \(long.total) frames into a 600-entry cache")
        XCTAssertEqual(long.cadence, 60)
        XCTAssertEqual(PreloadPlan(frameCount: 150, cadence: 1, capacity: 600).total, 150)
        var empty = PreloadPlan(frameCount: 0, cadence: 0, capacity: 0)   // no divide by zero
        XCTAssertTrue(empty.isComplete); XCTAssertNil(empty.take(nearest: 0))
    }

    /// "analysed N of M" is on screen while the user decides whether to wait, so N may only
    /// rise, may never pass M, and a frame the decoder refuses leaves the denominator rather
    /// than stranding the sweep one short of finishing forever.
    func testProgressOnlyRisesNeverPassesTheTotalAndAlwaysReachesIt() {
        var plan = PreloadPlan(frameCount: 20, cadence: 5, capacity: 600)
        XCTAssertEqual([plan.total, plan.analysed], [4, 0])
        var last = 0
        while plan.take(nearest: 0) != nil {
            plan.complete()
            XCTAssertGreaterThanOrEqual(plan.analysed, last, "progress went backwards")
            XCTAssertLessThanOrEqual(plan.analysed, plan.total, "progress passed the total")
            last = plan.analysed
        }
        XCTAssertTrue(plan.isComplete)
        for _ in 0..<5 { plan.complete() }
        XCTAssertEqual(plan.analysed, 4, "extra completions ran the count past the plan")
        var refused = PreloadPlan(frameCount: 20, cadence: 5, capacity: 600)
        _ = refused.take(nearest: 0); refused.complete()
        _ = refused.take(nearest: 0); refused.drop()
        XCTAssertEqual([refused.analysed, refused.total], [1, 3],
                       "a frame the decoder refused stayed in the denominator forever")
    }

    /// Lesson 2 on a sweep rather than a still: leaving the preview stops the work promptly.
    /// Measured against what the same interval bought while it ran, so it reads the same however
    /// fast this machine is.
    func testCancellingASweepStopsItIssuingWork() async throws {
        let cache = ResultCache(frameRate: 30), issued = Issued()
        let coordinator = PreloadCoordinator()
        coordinator.begin(frameCount: 400, cadence: 1, cache: cache) { index in
            issued.record(index); await stubbornDelay(0.02); return stub(CGFloat(index))
        }
        try await Task.sleep(nanoseconds: 300_000_000)
        let running = issued.count
        XCTAssertGreaterThan(running, 2, "the sweep never started")
        coordinator.cancel()
        let atCancel = issued.count
        try await Task.sleep(nanoseconds: 400_000_000)
        // One more may already have been taken when the cancel landed; nothing beyond that.
        XCTAssertLessThanOrEqual(issued.count - atCancel, 1,
                                 "a cancelled sweep issued \(issued.count - atCancel) more frames "
                                     + "against \(running) in the same interval running")
        XCTAssertEqual(coordinator.plan?.analysed, cache.count,
                       "progress counted a frame the cache never got")
    }

    /// A new pick supersedes the old sweep rather than racing it. The step ignores cancellation
    /// on purpose — one `predict` is uninterruptible, so the pass in flight when the second clip
    /// arrives *will* finish, and what must not happen is its result landing anywhere.
    func testASecondPickAbandonsTheSweepAlreadyInFlight() async throws {
        let abandoned = ResultCache(frameRate: 30), current = ResultCache(frameRate: 30)
        let first = Issued(), second = Issued()
        let coordinator = PreloadCoordinator()
        coordinator.begin(frameCount: 200, cadence: 1, cache: abandoned) { index in
            first.record(index); await stubbornDelay(0.25); return stub(1)
        }
        try await Task.sleep(nanoseconds: 60_000_000)
        coordinator.begin(frameCount: 20, cadence: 5, cache: current) { index in
            second.record(index); await stubbornDelay(0.01); return stub(2)
        }
        let issuedBefore = first.count
        try await Task.sleep(nanoseconds: 600_000_000)
        XCTAssertEqual(first.count, issuedBefore, "the superseded sweep kept issuing work")
        XCTAssertEqual(abandoned.count, 0,
                       "a pass from the previous clip landed in its cache after it was replaced")
        XCTAssertEqual(coordinator.plan?.total, 4, "the plan still describes the previous clip")
        XCTAssertEqual(current.count, 4)
        XCTAssertEqual(second.value.sorted(), [0, 5, 10, 15])
    }

    /// The sweep and the preview's opportunistic pass share one cache, so a frame the user already
    /// waited for must not be paid for twice — but it still counts, or progress never reaches M.
    func testFramesTheCacheAlreadyHasCostNothingAndStillCount() async throws {
        let cache = ResultCache(frameRate: 30), issued = Issued()
        for index in [0, 5, 10] { cache.insert(stub(CGFloat(index)), at: index) }
        let coordinator = PreloadCoordinator()
        coordinator.begin(frameCount: 20, cadence: 5, cache: cache) { index in
            issued.record(index); await stubbornDelay(0.01); return stub(99)
        }
        for _ in 0..<40 where coordinator.plan?.isComplete != true {
            try await Task.sleep(nanoseconds: 25_000_000)
        }
        XCTAssertEqual(issued.value, [15],
                       "the sweep re-ran frames the preview had already paid ~9.7 s for")
        XCTAssertEqual(coordinator.plan?.analysed, 4, "the free frames never reached the count")
        XCTAssertEqual(cache.value(at: 0), stub(0), "a cached answer was overwritten")
    }

    /// Why a cadence above 1 is usable at all: the grid leaves four frames in five unanalysed, so
    /// a scrub landing between two of them is served by its neighbour — and the badge names the
    /// frame that neighbour really is, the invariant `ResultCacheTests` guards for exact hits.
    func testAPreloadedNeighbourIsShownInsteadOfPayingForAnotherPass() {
        let cache = ResultCache(frameRate: 30)
        cache.insert(stub(10), at: 10)
        XCTAssertEqual(cache.nearest(to: 12, within: 2)?.index, 10)
        XCTAssertNil(cache.nearest(to: 13, within: 2),
                     "a frame outside the tolerance was served someone else's result")
        cache.insert(stub(14), at: 14)
        XCTAssertEqual(cache.nearest(to: 12, within: 2)?.index, 14, "a tie did not break forward")
        let (held, overlay) = VideoPreviewModel.resolve(
            frameAt: 12.0 / 30, near: cache.nearest(to: 12, within: 2), index: 12,
            frameRate: 30, held: nil)
        XCTAssertEqual(overlay, .stale(14.0 / 30))
        XCTAssertEqual(held, VideoPreviewModel.Held(result: stub(14), time: 14.0 / 30))
        // An exact hit is still exact, and a miss still keeps whatever was on screen.
        let (_, exact) = VideoPreviewModel.resolve(
            frameAt: 10.0 / 30, near: cache.nearest(to: 10, within: 2), index: 10,
            frameRate: 30, held: nil)
        XCTAssertEqual(exact, .current)
        let (kept, miss) = VideoPreviewModel.resolve(frameAt: 3, near: nil, index: 90,
                                                     frameRate: 30, held: held)
        XCTAssertEqual(kept, held, "a miss dropped the overlay that was on screen")
        XCTAssertEqual(miss, .stale(14.0 / 30))
    }
}
