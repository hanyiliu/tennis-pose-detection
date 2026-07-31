//  PreloadCoordinatorTests.swift
//  The sweep costs ~9.7 s a frame, so its order, its stopping and what it claims to have done are
//  checked as scheduling over an injected step, not by watching a clip.

import CoreGraphics
import Foundation
import XCTest
@testable import TPD

/// What the sweep asked for, in order. `@unchecked Sendable` over a lock: the step runs off main.
private final class Issued: @unchecked Sendable {
    private let lock = NSLock()
    private var indices: [Int] = []
    var value: [Int] { lock.withLock { indices } }
    var count: Int { lock.withLock { indices.count } }
    func record(_ index: Int) { lock.withLock { indices.append(index) } }
}

/// A delay that **ignores cancellation**, standing in for the one uninterruptible `predict` every
/// pass makes: `Task.sleep` would throw before the checks that run after the await ever ran.
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
    /// The headline: front-to-back is the wrong order the moment the user scrubs to the middle.
    func testTheFrameOnScreenGoesFirstAndTheSweepWorksOutwardsForwardBeforeBack() {
        var plan = PreloadPlan(frameCount: 51, cadence: 5, capacity: 600), order: [Int] = []
        XCTAssertEqual(plan.pending, Set(stride(from: 0, to: 51, by: 5)))
        for _ in 0..<6 { order.append(plan.take(nearest: 25) ?? -1) }  // scrubbed to the middle
        XCTAssertEqual(order, [25, 30, 20, 35, 40, 15],
                       "a front-to-back sweep starts at 0, making the user sit through the skipped")
        // ...and it re-aims the moment they scrub elsewhere, rather than finishing its ring.
        XCTAssertEqual(plan.take(nearest: 0), 0)
        XCTAssertEqual(plan.take(nearest: 50), 50)
    }

    /// Preloading must not push the cache past its bound — a sweep that outgrew it would spend
    /// ~9.7 s a frame evicting its own answers — and the tolerance must not follow the cadence up.
    func testTheCadenceRisesUntilThePlanFitsTheCacheButTheToleranceStaysAStrokePhase() {
        let clip = PreloadPlan(frameCount: 150, cadence: 5, capacity: 600, frameRate: 30)
        XCTAssertEqual([clip.cadence, clip.total, clip.tolerance], [5, 30, 2])
        // 20 minutes at 30 fps: 36 000 frames at the default cadence into a 600-entry bound.
        let long = PreloadPlan(frameCount: 36_000, cadence: 5, capacity: 600, frameRate: 30)
        XCTAssertLessThanOrEqual(long.total, 600,
                                 "the plan scheduled \(long.total) frames into a 600-entry cache")
        XCTAssertEqual(long.cadence, 60)
        XCTAssertEqual(long.tolerance, 2,
                       "half this cadence is \(long.cadence / 2) frames — a second of clip away")
        // The bound is 75 ms of clip whatever the frame rate, so it converts rather than hardcodes.
        XCTAssertEqual(PreloadPlan(frameCount: 36_000, cadence: 5, capacity: 600,
                                   frameRate: 60).tolerance, 4)
        XCTAssertEqual(PreloadPlan(frameCount: 150, cadence: 1, capacity: 600).total, 150)
        var empty = PreloadPlan(frameCount: 0, cadence: 0, capacity: 0)   // no divide by zero
        XCTAssertTrue(empty.isComplete); XCTAssertNil(empty.take(nearest: 0))
    }

    /// "analysed N of M" is on screen while the user decides whether to wait, so N may only rise,
    /// may never pass M, and a refused frame leaves M rather than stranding the sweep one short.
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

    /// The sweep's whole lifecycle, each stage measured against what the same interval bought while
    /// it ran, so it reads the same however fast this machine is. An export holds the one engine
    /// for minutes, so Share suspends the sweep, closing the card resumes it, leaving cancels it.
    func testASweepSuspendsForAnExportResumesIntactAndCancelsForGood() async throws {
        let cache = ResultCache(frameRate: 30), issued = Issued()
        let coordinator = PreloadCoordinator()
        coordinator.begin(frameCount: 400, cadence: 1, cache: cache) { index in
            issued.record(index); await stubbornDelay(0.02); return stub(CGFloat(index))
        }
        try await Task.sleep(nanoseconds: 300_000_000)
        let running = issued.count
        XCTAssertGreaterThan(running, 2, "the sweep never started")
        coordinator.suspend()                                    // Share tapped
        let atSuspend = issued.count
        try await Task.sleep(nanoseconds: 400_000_000)
        // One more may already have been taken when it landed; nothing beyond that.
        XCTAssertLessThanOrEqual(issued.count - atSuspend, 1, "a suspended sweep issued "
                                 + "\(issued.count - atSuspend) more, against \(running) running")
        coordinator.resume()                                     // the export card closed
        try await Task.sleep(nanoseconds: 300_000_000)
        XCTAssertGreaterThan(issued.count - atSuspend, 2, "the sweep never picked back up")
        XCTAssertEqual(issued.value, Array(0..<issued.count),
                       "the pause skipped or repeated a frame instead of holding its place")
        coordinator.cancel()                                     // the preview went away
        let atCancel = issued.count
        try await Task.sleep(nanoseconds: 400_000_000)
        XCTAssertLessThanOrEqual(issued.count - atCancel, 1,
                                 "a cancelled sweep issued \(issued.count - atCancel) more frames")
        XCTAssertEqual(coordinator.plan?.analysed, cache.count,
                       "progress counted a frame the cache never got")
    }

    /// A new pick supersedes the old sweep rather than racing it. The step ignores cancellation on
    /// purpose: the pass in flight *will* finish, and its result must not land anywhere.
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
        XCTAssertEqual(abandoned.count, 0, "a pass from the replaced clip landed in its cache")
        XCTAssertEqual(coordinator.plan?.total, 4, "the plan still describes the previous clip")
        XCTAssertEqual(current.count, 4)
        XCTAssertEqual(second.value.sorted(), [0, 5, 10, 15])
    }

    /// A frame the user already waited for must not be paid for twice — but it still counts.
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

    /// The own-pass guarantee has to hold on the error path too. A frame parked off the grid holds
    /// its own pass back until the sweep gives the engine up, so a sweep whose last pass ends in a
    /// skip and announces nothing leaves it on a stranger's pose for good — the very thing serving
    /// a neighbour has to avoid. The closure is `sync()`'s guard: its own pass once the sweep ends.
    func testASweepEndingInASkipStillLetsTheParkedFrameGetItsOwnPass() async throws {
        let cache = ResultCache(frameRate: 30), coordinator = PreloadCoordinator()
        let parked = 12, own = stub(99)   // off the cadence-5 grid: only frame 10 can serve it
        coordinator.onResult = {
            guard !coordinator.isSweeping, cache.value(at: parked) == nil else { return }
            cache.insert(own, at: parked)
        }
        coordinator.begin(frameCount: 20, cadence: 5, cache: cache) { index in
            await stubbornDelay(0.01)
            // 15 is last from a playhead at 0, so the sweep ends in `drop()`, not `complete()`.
            if index == 15 { throw CocoaError(.fileReadCorruptFile) }
            return stub(CGFloat(index))
        }
        for _ in 0..<80 where coordinator.plan?.isComplete != true {
            try await Task.sleep(nanoseconds: 25_000_000)
        }
        XCTAssertEqual([coordinator.plan?.analysed, coordinator.plan?.total], [3, 3],
                       "the refused frame never left the denominator, so the sweep never ended")
        XCTAssertEqual(cache.nearest(to: parked, within: 2)?.index, parked)
        XCTAssertEqual(cache.value(at: parked), own, "the sweep ended in a skip and said nothing, "
                       + "so the parked frame kept frame 10's pose with no pass of its own coming")
    }

    /// Why a cadence above 1 is usable at all: a scrub between two grid frames is served by its
    /// neighbour, badged with the frame that neighbour really is — and still owed its own pass.
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
