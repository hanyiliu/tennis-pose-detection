//  StillInferenceWorkerTests.swift
//  The two things the still path now promises about work nobody can see: a
//  render whose task is cancelled stops at the next stage boundary, and no
//  number of presentations builds more than one engine.

import CoreGraphics
import CoreImage
import Foundation
import XCTest
@testable import TPD

final class StillInferenceWorkerTests: XCTestCase {
    /// The size the defect was measured at — an ordinary 12 MP phone photo. Built
    /// rather than bundled: no binary belongs in the repo, and the cost this file
    /// is about is a function of the pixel count, not of the subject.
    private static let twelveMegapixel = (width: 4032, height: 3024)

    // MARK: - Cancellation

    /// The headline. `render` used to be one uninterruptible block: the caller's
    /// single `checkCancellation` only caught an exit that beat the actor hop,
    /// and after that decode -> predict -> rasterize ran to the end no matter
    /// what. Measured on this simulator with a 12 MP still, a cancel at 0.5 s
    /// returned a finished frame 58.4 s later.
    ///
    /// This asserted a 15 s wall-clock bound and that was the wrong instrument.
    /// The stages are nothing like equal: timed here at 12 MP they are upright
    /// 1.2 s, `predict` 52.9 s, rasterize 2.8 s. `predict` is ~95% of the render
    /// and one uninterruptible call, so whether 15 s was generous or impossible
    /// came down to which side of the upright/predict boundary the cancel
    /// happened to land on — 0.6 s if it beat `predict`, 52-56 s if it did not,
    /// on the same code. That is a coin toss weighted by machine load, and this
    /// box is a shared one: it went red at 56.6 s on unmodified `main`.
    ///
    /// Both of the things the checks actually buy are asserted below instead,
    /// each against a cost measured in the same run on the same input, so they
    /// read the same however fast this machine happens to be:
    ///
    /// 1. a render cancelled *while it is running* throws instead of handing back
    ///    a frame — something past the first check is looking at cancellation;
    /// 2. a render cancelled *before its turn* costs a small fraction of what one
    ///    that ran costs — the checks sit in front of the work rather than after
    ///    it, which is the whole reason they earn their keep.
    ///
    /// The actor is what makes the second one a fact rather than a race. `render`
    /// is synchronous, so it cannot be reentered; `queued` is created and
    /// cancelled while `inFlight` still owns the actor, so it is provably still
    /// waiting when the cancel reaches it, and what it costs is the interval
    /// after `inFlight` let go.
    func testCancellingAStillStopsItAtTheNextStageBoundary() async throws {
        // Warm the engine first: which boundary this caught used to depend on
        // whether anything else had built the models yet — run alone it cancelled
        // inside the model load, run in the suite inside `predict`. Now it is
        // always one of the interior ones.
        _ = try await StillInferenceWorker.shared.render(try Self.jpeg(320, 240))
        let data = try Self.jpeg(Self.twelveMegapixel.width, Self.twelveMegapixel.height)

        let start = Date()
        let inFlight = Self.attempt(data)
        // A lower bound and nothing else: long enough that the render owns the
        // actor, and shorter than the shortest stage it could be inside — the
        // decode, 0.5 s here at 12 MP and 1.2 s with the box under load. Load
        // only widens that margin. Too short and the cancel would beat the task
        // onto the pool, which makes the first pair of assertions vacuous rather
        // than wrong; the old 0.5 s was already past the decode half the time,
        // which is what made its 15 s bound a coin toss.
        try await Task.sleep(nanoseconds: 250_000_000)
        // Created and cancelled while the actor is still held, so this one is
        // known not to have started.
        let queued = Self.attempt(data)
        queued.cancel()
        inFlight.cancel()
        let ranOutcome = await inFlight.value, queuedOutcome = await queued.value

        // Both intervals end on a stamp taken inside the task that finished, so
        // the ratio is the worker's doing and not the test's own hops.
        let ran = ranOutcome.finished.timeIntervalSince(start)
        let waited = queuedOutcome.finished.timeIntervalSince(ranOutcome.finished)
        NSLog("TPD test: 12 MP render cancelled in flight returned at %.1f s; one cancelled "
                  + "before its turn cost %.3f s of that", ran, waited)

        XCTAssertFalse(ranOutcome.producedAFrame,
                       "a render cancelled 0.25 s in handed back a finished frame — nothing past "
                           + "the first check in render() is looking at cancellation")
        XCTAssertTrue(ranOutcome.cancelled, "expected CancellationError, got \(ranOutcome.error)")
        XCTAssertFalse(queuedOutcome.producedAFrame,
                       "a render cancelled before its turn handed back a frame")
        XCTAssertTrue(queuedOutcome.cancelled,
                      "expected CancellationError, got \(queuedOutcome.error)")
        XCTAssertLessThan(waited, ran / 4,
                          "a render cancelled before it started still cost \(waited) s against "
                              + "\(ran) s for one that ran — the checks in render() are no longer "
                              + "in front of the work they exist to skip")
    }

    /// The queued case, which is what repeated pick-and-exit actually produces:
    /// one shared actor serializes renders, so a still whose preview is already
    /// gone by the time its turn comes must not start at all.
    func testAnAlreadyCancelledStillNeverStarts() async throws {
        let data = try Self.jpeg(320, 240)
        let worker = StillInferenceWorker.shared
        let task = Task.detached { try await worker.render(data) }
        task.cancel()
        do {
            _ = try await task.value
            XCTFail("a cancelled render returned a frame")
        } catch {
            XCTAssertTrue(error is CancellationError, "expected CancellationError, got \(error)")
        }
    }

    // MARK: - One engine, not one per presentation

    /// Every presentation used to build its own `StillInferenceWorker` and so its
    /// own `TPDInferenceEngine` — a second copy of both Core ML models and a
    /// second `CIContext` per pick. `shared` is now the only way in (`init` is
    /// private) and this is the observable half of that.
    func testTheEngineIsBuiltOnceForTheWholeApp() async throws {
        let data = try Self.jpeg(480, 360)
        let worker = StillInferenceWorker.shared
        _ = try await worker.render(data)
        let afterFirst = await worker.modelLoadCount
        _ = try await worker.render(data)
        let afterSecond = await worker.modelLoadCount

        XCTAssertEqual(afterFirst, 1, "the engine must be built exactly once")
        XCTAssertEqual(afterSecond, 1, "a second render rebuilt the engine")
    }

    /// Four 12 MP picks abandoned after 0.7 s, the shape that used to push
    /// phys_footprint from 215.4 MB to 427.1 MB here: each abandoned render kept
    /// a full-frame bitmap, a `CIContext` and its own pair of models alive while
    /// the next pick built the same again. The engine is warmed first, so what
    /// this measures is the stacking and not the one-time cost of the models.
    func testRepeatedAbandonedPicksDoNotStack() async throws {
        _ = try await StillInferenceWorker.shared.render(try Self.jpeg(320, 240))
        let data = try Self.jpeg(Self.twelveMegapixel.width, Self.twelveMegapixel.height)
        let before = Self.physFootprintMB()

        var tasks: [Task<RenderedFrame, Error>] = []
        for cycle in 1...4 {
            let worker = StillInferenceWorker.shared
            let task = Task.detached { try await worker.render(data) }
            tasks.append(task)
            try await Task.sleep(nanoseconds: 700_000_000)
            task.cancel()
            NSLog("TPD test: pick %d abandoned after 0.7 s, phys_footprint %.1f MB",
                  cycle, Self.physFootprintMB())
        }
        try await Task.sleep(nanoseconds: 5_000_000_000)
        let after = Self.physFootprintMB()
        NSLog("TPD test: phys_footprint %.1f MB -> %.1f MB over four abandoned picks",
              before, after)
        for task in tasks { _ = try? await task.value }

        XCTAssertLessThan(after - before, 100,
                          "four abandoned picks grew phys_footprint by \(after - before) MB — "
                              + "renders are stacking again")
    }

    // MARK: - The happy path still works

    /// The checks are cheap and they are also the easiest thing to put in the
    /// wrong place, so: an uncancelled render still returns the frame and the
    /// result measured from it, both at the upright size.
    func testAnUncancelledRenderStillProducesTheFrameAndItsResult() async throws {
        let spec = try TPDModelSpec.load(from: .main)
        let rendered = try await StillInferenceWorker.shared.render(try Self.jpeg(900, 675))

        XCTAssertEqual(rendered.image.width, 900)
        XCTAssertEqual(rendered.image.height, 675)
        XCTAssertEqual(rendered.result.frameSize, CGSize(width: 900, height: 675))
        XCTAssertEqual(rendered.result.keypoints.count, spec.numKeypoints)
        XCTAssertEqual(rendered.result.probabilities.reduce(0, +), 1, accuracy: 1e-3)
    }

    // MARK: - Fixtures

    /// What one render attempt did. The frame is dropped inside the task rather
    /// than carried out of it: only whether there was one matters here, and a
    /// `RenderedFrame` the test holds is a 12 MP bitmap kept alive for nothing.
    private struct Attempt: Sendable {
        let producedAFrame: Bool
        let cancelled: Bool
        let error: String
        /// Stamped inside the task the moment `render` returned or threw, so an
        /// interval between two of these contains no scheduling of the test's own.
        let finished: Date
    }

    /// A render on the shared worker, started detached and never throwing out of
    /// the task: the outcome is data to be compared, not a control flow.
    private static func attempt(_ data: Data) -> Task<Attempt, Never> {
        Task.detached {
            do {
                _ = try await StillInferenceWorker.shared.render(data)
                return Attempt(producedAFrame: true, cancelled: false, error: "", finished: Date())
            } catch {
                return Attempt(producedAFrame: false, cancelled: error is CancellationError,
                               error: "\(error)", finished: Date())
            }
        }
    }

    /// A JPEG of the requested size: a bright figure-shaped bar on grass green,
    /// so stage 1 has something to put a box around.
    private static func jpeg(_ width: Int, _ height: Int) throws -> Data {
        let space = CGColorSpaceCreateDeviceRGB()
        let context = try XCTUnwrap(CGContext(
            data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
            space: space,
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                | CGBitmapInfo.byteOrder32Little.rawValue))
        context.setFillColor(CGColor(red: 0.24, green: 0.29, blue: 0.22, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        context.setFillColor(CGColor(red: 0.92, green: 0.90, blue: 0.86, alpha: 1))
        context.fill(CGRect(x: Double(width) * 0.42, y: Double(height) * 0.2,
                            width: Double(width) * 0.15, height: Double(height) * 0.62))
        let image = try XCTUnwrap(context.makeImage())
        return try XCTUnwrap(CIContext().jpegRepresentation(
            of: CIImage(cgImage: image), colorSpace: space, options: [:]))
    }

    /// The number Instruments' Memory gauge shows and the one the OS kills on,
    /// read from `TASK_VM_INFO`. `TPDTests` is hosted by the `TPD` app, so this
    /// is the app's own footprint.
    private static func physFootprintMB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let status = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard status == KERN_SUCCESS else { return -1 }
        return Double(info.phys_footprint) / 1_048_576
    }
}
