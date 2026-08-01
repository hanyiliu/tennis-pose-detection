//  LiveViewModelConcurrencyTests.swift
//  The four concurrency defects the live loop shipped — every one caught by review, none by a test
//  — and the single distinction all four got wrong: a generation says whether a result may be
//  PUBLISHED. It does not say what a run OWNS (that is `producer`), and it does not say when the
//  shared drop gate may be RELEASED (that is the pass which set it).
//
//  Every case drives two overlapping `run()`s across one shared `FrameSource`, which is what
//  `.task(id:)` does at a model switch. Ordering comes from the model's own observable state, not
//  from sleeps: `run()` bumps `generation` first thing, so that counter moving proves the incoming
//  prologue ran, and `isInferring` closing proves a pass is in flight. The one step that cannot be
//  synchronised, and the two budgets everything waits on, say so where they are.

import CoreMedia
import CoreVideo
import Foundation
import XCTest

@testable import TPD

private struct Timeout: Error { let what: String }

/// `coldStart` covers the one genuinely slow, genuinely once step — the process's first Core ML
/// load, 35 s on this suite's own first clean run against ~2 s warm — and is far wider than any
/// measurement because it is a hang detector, not a latency target. `warmWait` is every per-case
/// wait, all of which run warm: an order of magnitude of slack, and a wedge still fails in seconds.
private let coldStart: TimeInterval = 300, warmWait: TimeInterval = 15

/// A producer shaped like the two real ones where the ordering argument lives: `lifecycle.begin()`
/// is the only thing that hands out a stream and a second `begin()` finishes the first's, so a run
/// that starts or stops a producer it does not own really does kill the live feed here, as on the
/// camera and the bundled clip. It also counts calls, keeps each run's token, and parks starts.
private final class ScriptedSource: FrameSource, @unchecked Sendable {
    /// Where the next `start()` suspends. `beforeOpening` leaves the previous run's stream live
    /// underneath the parked one; `afterOpening` parks a run whose own stream is already open, and
    /// which can therefore be superseded while it sleeps.
    enum Park { case beforeOpening, afterOpening }

    private let lifecycle = FrameSourceLifecycle()
    private let lock = NSLock()
    private var startCount = 0, openCount = 0, stopCount = 0
    private var issued: [UInt64] = []
    private var pending: Park?
    private var parked: CheckedContinuation<Void, Never>?

    var starts: Int { lock.withLock { startCount } }
    var opened: Int { lock.withLock { openCount } }
    var stops: Int { lock.withLock { stopCount } }
    var tokens: [UInt64] { lock.withLock { issued } }
    var isParked: Bool { lock.withLock { parked != nil } }

    /// The next `start()`, and only the next, suspends at `point`.
    func parkNextStart(at point: Park) { lock.withLock { pending = point } }

    func releaseParkedStart() {
        let waiting = lock.withLock { () -> CheckedContinuation<Void, Never>? in
            defer { parked = nil }
            return parked
        }
        waiting?.resume()
    }

    func start() async throws -> FrameStream {
        let point = lock.withLock { () -> Park? in
            startCount += 1
            defer { pending = nil }
            return pending
        }
        if point == .beforeOpening { await park() }
        let (stream, token) = lifecycle.begin()
        lock.withLock { openCount += 1; issued.append(token) }
        if point == .afterOpening { await park() }
        return stream
    }

    private func park() async {
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            lock.withLock { parked = continuation }
        }
    }

    func stop() { lock.withLock { stopCount += 1 }; lifecycle.stop() }

    /// Is the stream this token was issued for still the live one?
    func isLive(_ token: UInt64) -> Bool { lifecycle.isCurrent(token) }

    func push(_ frame: VideoFrame) { lifecycle.yield(frame) }
}

/// A frame of flat grey. Built rather than bundled — no binary belongs in the repo — and its only
/// job is to be a real `CVPixelBuffer` the engines will actually run on.
private func grey(_ width: Int = 160, _ height: Int = 120) -> VideoFrame {
    var buffer: CVPixelBuffer?
    CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA,
                        [kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary] as CFDictionary,
                        &buffer)
    let pixels = buffer!
    CVPixelBufferLockBaseAddress(pixels, [])
    if let base = CVPixelBufferGetBaseAddress(pixels) {
        memset(base, 128, CVPixelBufferGetBytesPerRow(pixels) * height)
    }
    CVPixelBufferUnlockBaseAddress(pixels, [])
    return VideoFrame(pixelBuffer: pixels, time: .zero)
}

@MainActor
final class LiveViewModelConcurrencyTests: XCTestCase {
    /// Pay the cold Core ML load once, here, where nothing is being timed. Before this the first
    /// case to run carried it inside its own per-case wait and went red for it: the suite's very
    /// first clean run failed a 30 s wait while the load underneath took 35 s — precisely the "just
    /// run it again" reflex these cases exist to remove. Both entries a switch moves between are
    /// warmed, through the real path: each one's first load and pass land in a timed wait below.
    private static var warmed = false

    override func setUp() async throws {
        try await super.setUp()
        guard !Self.warmed else { return }
        let source = ScriptedSource()
        let model = LiveViewModel(source: source)
        for entry in try switchable(model).prefix(2) {
            model.select(entry)
            let warm = Task { @MainActor in await model.run() }
            try await waitUntil("the first load of \(entry.id) and one pass through it",
                                timeout: coldStart, feeding: source,
                                diagnosing: model) { model.performance.completed >= 1 }
            warm.cancel()
            _ = await warm.value
        }
        Self.warmed = true
    }

    // MARK: - Defect 1: a superseded run must start nothing
    /// Run A parks in `await worker.load`, a switch supersedes it, run B takes over. When A wakes it
    /// must NOT reach `start()`: that opens a stream through the shared lifecycle, and opening one
    /// finishes whatever was live — the stream the run owning the screen sits in. A frozen preview.
    func testARunSupersededAcrossTheModelLoadStartsNoProducer() async throws {
        let source = ScriptedSource()
        let model = LiveViewModel(source: source)
        let entries = try switchable(model)

        let stale = Task { @MainActor in await model.run() }
        // `run()` is `@MainActor` and its first suspension is `worker.load`, which leaves main, so
        // spinning on the generation bump lands this in the very next main-actor turn: `starts == 0`
        // is checked while A is provably still in the load, not after a sleep long enough to finish.
        try await spinUntil("run A to enter run()") { self.generation(model) == 1 }
        XCTAssertEqual(source.starts, 0, "run A reached start() before it was superseded")
        XCTAssertNil(model.active, "run A is already past the model load")

        model.select(entries[1])  // the supersede, then what `.task(id:)` does to the outgoing run
        stale.cancel()
        let live = Task { @MainActor in await model.run() }
        _ = await stale.value  // A has now done everything it is ever going to do
        try await waitUntil("run B to publish a frame", feeding: source,
                            diagnosing: model) { model.performance.completed >= 1 }

        XCTAssertEqual(source.starts, 1, "a superseded run called start(); its begin() finishes "
                       + "the live run's stream and the feed dies || \(internals(model))")
        XCTAssertEqual(source.stops, 0, "a run that started no producer stopped one")
        XCTAssertEqual(model.active?.id, entries[1].id, "the wrong model is on screen")
        XCTAssertTrue(source.isLive(try XCTUnwrap(source.tokens.last)),
                      "the live run's stream was closed under it")

        // And it keeps going, rather than having been left holding a finished stream.
        let published = model.performance.completed
        try await waitUntil("run B to keep publishing", feeding: source,
                            diagnosing: model) { model.performance.completed > published }
        live.cancel()
        _ = await live.value
    }

    // MARK: - Defect 2: a superseded run must stop nothing it did not start
    /// The opposite mistake, and the fix for the one above. Run A gets *inside* `start()` — stream
    /// open — and parks; a switch supersedes it and run B opens its own, finishing A's. When A wakes
    /// it must leave the producer alone: `stop()` finishes whichever stream is live, and that is B's.
    func testARunSupersededInsideStartDoesNotTearDownTheLiveProducer() async throws {
        let source = ScriptedSource()
        let model = LiveViewModel(source: source)
        let entries = try switchable(model)

        source.parkNextStart(at: .afterOpening)
        let stale = Task { @MainActor in await model.run() }
        try await waitUntil("run A to park inside start() with its stream open") { source.isParked }
        XCTAssertEqual(source.opened, 1)

        model.select(entries[1])
        stale.cancel()
        let live = Task { @MainActor in await model.run() }
        try await waitUntil("run B to open its own stream") { source.opened == 2 }
        let liveToken = try XCTUnwrap(source.tokens.last)
        try await waitUntil("run B to publish a frame", feeding: source,
                            diagnosing: model) { model.performance.completed >= 1 }
        XCTAssertEqual(source.stops, 0, "something stopped the source before the stale run woke")

        source.releaseParkedStart()  // now let the superseded run out of start()
        _ = await stale.value

        XCTAssertEqual(source.stops, 0, "the superseded run stopped a producer it never owned")
        XCTAssertTrue(source.isLive(liveToken), "the superseded run finished the live stream")
        let published = model.performance.completed
        try await waitUntil("run B to keep publishing after the stale run woke", feeding: source,
                            diagnosing: model) { model.performance.completed > published }

        // The owner, and only the owner, tears the producer down when it ends.
        live.cancel()
        try await waitUntil("the owning run to stop the producer it started") { source.stops == 1 }
        _ = await live.value
    }

    // MARK: - Defect 3: a stale pass must not wedge the gate the incoming run needs
    /// One late loop body of a live, publishing run A spawns a pass carrying A's — by then stale —
    /// generation, having closed the drop gate. The user switches models. With a generation guard
    /// above the release the pass never reopens it: the incoming run drops every frame it is handed.
    func testALateStalePassMustNotWedgeTheIncomingRunsDropGate() async throws {
        let source = ScriptedSource()
        let (model, entries, outgoing) = try await liveRun(over: source)

        // The switch, with run B parked BEFORE it opens a stream: A's stream stays live and A's loop
        // keeps iterating under it — the interleaving a producer yielding off main makes reachable.
        source.parkNextStart(at: .beforeOpening)
        model.select(entries[1])
        let entered = try XCTUnwrap(generation(model))
        let incoming = Task { @MainActor in await model.run() }
        try await waitUntil("run B to park inside start()") { source.isParked }
        // `run()`'s first statement is `generation += 1`, so +1 is proof B's prologue ran — where the
        // compensating reset that used to paper over this lived — and the park proves B is past it
        // while holding no stream of its own.
        XCTAssertEqual(generation(model), entered + 1)
        XCTAssertEqual(model.performance.completed, 0, "select() should have reset the meter")

        // Exactly ONE late loop body of A, strictly after B's prologue.
        source.push(grey())
        try await waitUntil("run A's late loop body to take that frame",
                            diagnosing: model) { self.gate(model) == true }
        let released = await becomesTrue(within: warmWait) { self.gate(model) == false }

        source.releaseParkedStart()
        try await waitUntil("run B to open its own stream") { source.opened == 2 }
        _ = await outgoing.value

        try await waitUntil("run B to publish anything at all (the stale pass released its gate: "
                            + "\(released))", feeding: source,
                            diagnosing: model) { model.performance.completed >= 1 }
        XCTAssertEqual(model.active?.id, entries[1].id, "the newly selected model is not on screen")
        incoming.cancel()
        _ = await incoming.value
    }

    /// The control for the case above: the identical switch, identically parked and ordered, whose
    /// single difference is that no late loop body of A is arranged. It passes on every build, which
    /// is the point — it is what makes the failure above attributable to the stale pass.
    func testControlTheSameSwitchWithoutALateStalePassPublishesNormally() async throws {
        let source = ScriptedSource()
        let (model, entries, outgoing) = try await liveRun(over: source)

        source.parkNextStart(at: .beforeOpening)
        model.select(entries[1])
        let entered = try XCTUnwrap(generation(model))
        let incoming = Task { @MainActor in await model.run() }
        try await waitUntil("run B to park inside start()") { source.isParked }
        XCTAssertEqual(generation(model), entered + 1)
        // The one difference: nothing is pushed here, so A runs no late loop body.
        source.releaseParkedStart()
        try await waitUntil("run B to open its own stream") { source.opened == 2 }
        _ = await outgoing.value

        try await waitUntil("run B to publish a frame of its own", feeding: source,
                            diagnosing: model) { model.performance.completed >= 1 }
        XCTAssertEqual(model.active?.id, entries[1].id)
        incoming.cancel()
        _ = await incoming.value
    }

    // MARK: - Defect 4: a pass in flight across a switch belongs to the model it was measured on
    /// `select()` clears the frame and resets the meter; if it does not also bump the generation, a
    /// pass still in flight lands and undoes both — the old overlay republished under the newly
    /// chosen name, its timing in the new model's meter, the HUD quoting one cost beside another.
    func testAPassInFlightAcrossASwitchPublishesNothingUnderTheNewModel() async throws {
        let source = ScriptedSource()
        let (model, entries, outgoing) = try await liveRun(over: source)
        XCTAssertEqual(model.active?.id, entries[0].id)

        // One pass of A in flight, proved by the gate closing rather than assumed from a sleep.
        source.push(grey())
        try await waitUntil("run A's loop body to take a frame",
                            diagnosing: model) { self.gate(model) == true }

        // The switch, with nothing started in its place: what lands now belongs to the model left.
        model.select(entries[1])
        outgoing.cancel()

        // On a build that releases the gate unconditionally, `finish()`'s first statement is that
        // release, so observing it proves the stale pass landed. On one that leaks it that proof
        // never comes: this dwell stands in for it, weakening that proof — never the assertions.
        let landed = await becomesTrue(within: warmWait) { self.gate(model) == false }
        XCTAssertNil(model.frame, "the model just left republished its overlay under the new one "
                     + "(the stale pass was observed landing: \(landed))")
        XCTAssertEqual(model.performance.completed, 0,
                       "the old model's timing was recorded into the new model's meter")
        XCTAssertNil(model.active, "a model no run has loaded is being named as active")
        XCTAssertEqual(model.selected?.id, entries[1].id)
        _ = await outgoing.value
    }

    // MARK: - The same distinction at the other site: teardown
    /// Ownership-not-generation is asserted above at `run()`'s post-`start()` guard and nowhere at
    /// the `defer` — so re-merging the two concepts at the teardown site, which is exactly what
    /// `e512db3` shipped (`defer { if generation == mine { source.stop() } }`), passes every other
    /// case in this file. Both halves are needed, either alone being satisfied by the wrong rule: a
    /// run that claimed no producer stops nothing, and the run that did stops it exactly once.
    func testTeardownStopsOnlyTheProducerTheRunItselfClaimed() async throws {
        let source = ScriptedSource()
        let (model, entries, owner) = try await liveRun(over: source)
        let owned = try XCTUnwrap(source.tokens.last)

        // A run that never gets as far as claiming: superseded inside `worker.load`, above
        // `producer = claim`. Nothing suspends between the spin and the supersede, so it cannot
        // wake in the gap, and `starts` below makes a violation of that loud rather than silent.
        model.select(entries[1])
        let entered = try XCTUnwrap(generation(model))
        let unclaimed = Task { @MainActor in await model.run() }
        try await spinUntil("the second run to enter run()") { self.generation(model) == entered + 1 }
        model.select(entries[0])
        unclaimed.cancel()
        _ = await unclaimed.value

        XCTAssertEqual(source.starts, 1, "the second run reached start() — it was either not "
                       + "superseded inside the load, or claimed a producer anyway")
        XCTAssertEqual(source.stops, 0, "a run that claimed no producer tore one down")
        XCTAssertTrue(source.isLive(owned), "a run that never owned this stream closed it")

        // The other half: the claim, not the generation, says whose job teardown is. Two switches
        // have moved the generation past this run; it is still the only one that may stop this.
        owner.cancel()
        try await waitUntil("the claiming run to stop the producer it started",
                            diagnosing: model) { source.stops == 1 }
        _ = await owner.value
        XCTAssertEqual(source.stops, 1, "the producer was torn down more than once")
        XCTAssertFalse(source.isLive(owned), "the owning run left its producer running")
    }

    // MARK: - The shared invariant: `isInferring` is true iff a pass is in flight
    /// Half one — every set is paired with a release **by the pass that set it**. Publishing is a
    /// decision about a result; the gate is shared state, and a pass that declines to publish still
    /// hands it back. Nothing else here asserts that without also asserting what got published.
    func testAStalePassReleasesTheGateItSet() async throws {
        let source = ScriptedSource()
        let (model, entries, outgoing) = try await liveRun(over: source)

        source.push(grey())
        try await waitUntil("run A's loop body to take a frame",
                            diagnosing: model) { self.gate(model) == true }
        // Make that in-flight pass stale and start nothing in its place, so the gate can only be
        // reopened by the pass itself.
        model.select(entries[1])
        outgoing.cancel()

        try await waitUntil("the stale pass to release the gate it set",
                            diagnosing: model) { self.gate(model) == false }
        _ = await outgoing.value
    }

    /// Half two — no compensating reset anywhere. A fresh `run()` clearing the gate in its prologue
    /// is what made a leaked gate survivable most of the time, and it is wrong on its own terms: it
    /// opens the gate while a pass is still running, so the outgoing run's next frame starts a
    /// second concurrent pass — precisely what the gate exists to prevent.
    ///
    /// Ordered, not slept. Main-actor jobs run FIFO, so the task created below is queued ahead of
    /// this function's own continuation and the generation bump proves its prologue ran before the
    /// gate is read; the pass suspends into the worker's executor on its first await, so its
    /// `finish()` cannot be queued ahead of either. Residual assumption: one whole inference does
    /// not finish inside those two main-actor hops. A flake here means that is what broke.
    func testEnteringAFreshRunDoesNotClearAGateHeldByAPassInFlight() async throws {
        let source = ScriptedSource()
        let (model, entries, outgoing) = try await liveRun(over: source)

        source.push(grey(640, 480))  // deliberately dearer than the frames above: more margin
        try await spinUntil("run A's loop body to take the frame") { self.gate(model) == true }

        source.parkNextStart(at: .beforeOpening)
        model.select(entries[1])
        let entered = try XCTUnwrap(generation(model))
        let incoming = Task { @MainActor in await model.run() }
        try await spinUntil("run B to enter run() and execute its prologue") {
            self.generation(model) == entered + 1
        }
        XCTAssertEqual(gate(model), true, "entering run() reopened a drop gate that a pass still "
                       + "in flight is holding || \(internals(model))")

        try await waitUntil("run B to park inside start()") { source.isParked }
        source.releaseParkedStart()
        incoming.cancel()
        _ = await incoming.value
        outgoing.cancel()
        _ = await outgoing.value
    }

    // MARK: - Harness
    /// The boilerplate five of the cases share: a model over `source`, the entries a switch moves
    /// between, and run A live, publishing, and idle — no pass in flight, so the next frame pushed
    /// is certainly taken rather than dropped.
    private func liveRun(over source: ScriptedSource) async throws
        -> (model: LiveViewModel, entries: [TPDModelEntry], run: Task<Void, Never>) {
        let model = LiveViewModel(source: source)
        let entries = try switchable(model)
        let run = Task { @MainActor in await model.run() }
        try await waitUntil("run A to open its stream") { source.opened == 1 }
        try await waitUntil("run A to publish a frame", feeding: source,
                            diagnosing: model) { model.performance.completed >= 1 }
        // Open *and staying* open: the feeding above leaves a pass in flight behind it.
        for _ in 0..<40 {
            try await waitUntil("run A's drop gate to open",
                                diagnosing: model) { self.gate(model) == false }
            try await Task.sleep(nanoseconds: 200_000_000)
            if gate(model) == false { return (model, entries, run) }
        }
        XCTFail("run A's drop gate never stayed open || \(internals(model))")
        throw Timeout(what: "an idle run A")
    }

    /// The two entries a switch moves between, or a skip that names the fix. Counting the registry
    /// is not the guard this file needs: TPDModelRegistry.json is committed, so it is in every clone
    /// and always says three, while the `.mlpackage`s it names are git-ignored and absent from a
    /// fresh one — so that guard passed and each case then FAILED on a missing resource instead of
    /// skipping. Ask the host bundle for the compiled models themselves, which is what is required.
    private func switchable(_ model: LiveViewModel) throws -> [TPDModelEntry] {
        let entries = model.models
        try XCTSkipIf(entries.count < 2,
                      "TPDModelRegistry.json lists \(entries.count) model(s); a switch needs two")
        let absent = entries.prefix(2).flatMap(\.resources)
            .filter { Bundle.main.url(forResource: $0, withExtension: "mlmodelc") == nil }
        try XCTSkipIf(!absent.isEmpty, "\(absent.joined(separator: ", ")) not in this build — the "
                      + ".mlpackage bundles are git-ignored, so a fresh clone has none. Run "
                      + "`make export` from frontend/ios, then `make generate`.")
        return entries
    }

    /// Polls on the main actor — the same actor that mutates everything read here — optionally
    /// feeding the source while it waits, and fails loudly rather than hanging the suite.
    private func waitUntil(_ what: String, timeout: TimeInterval = warmWait,
                           feeding source: ScriptedSource? = nil,
                           diagnosing model: LiveViewModel? = nil,
                           _ condition: () -> Bool) async throws {
        let deadline = Date().addingTimeInterval(timeout)
        while !condition() {
            if Date() > deadline {
                XCTFail("timed out after \(Int(timeout)) s waiting for \(what)"
                        + (model.map { " || \(internals($0))" } ?? ""))
                throw Timeout(what: what)
            }
            source?.push(grey())
            try await Task.sleep(nanoseconds: 20_000_000)
        }
    }

    /// The same, without sleeping: the condition is re-read every main-actor turn, so it is observed
    /// in the turn after the change rather than a poll later. Only where that gap would be raced.
    private func spinUntil(_ what: String, turns: Int = 400_000,
                           _ condition: () -> Bool) async throws {
        for _ in 0..<turns {
            if condition() { return }
            await Task.yield()
        }
        XCTFail("spun \(turns) main-actor turns waiting for \(what)")
        throw Timeout(what: what)
    }

    /// Non-failing: did `condition` become true inside `timeout`?
    private func becomesTrue(within timeout: TimeInterval, _ condition: () -> Bool) async -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if condition() { return true }
            try? await Task.sleep(nanoseconds: 20_000_000)
        }
        return condition()
    }

    /// `isInferring` and `generation` are private, and reading them is the only way to prove *when*
    /// a pass is in flight instead of inferring it from a sleep — what made every earlier attempt at
    /// these cases timing-dependent. `Mirror` sees the `@Observable` macro's backing storage, so both
    /// spellings are accepted; a rename returns nil and every caller turns nil into a failure, never
    /// a silent pass. A `#if DEBUG` read-only view of the two would retire it; no app code is owned.
    private func peek<T>(_ model: LiveViewModel, _ name: String) -> T? {
        for child in Mirror(reflecting: model).children
        where child.label == name || child.label == "_" + name {
            return child.value as? T
        }
        return nil
    }

    private func gate(_ model: LiveViewModel) -> Bool? { peek(model, "isInferring") }
    private func generation(_ model: LiveViewModel) -> Int? { peek(model, "generation") }

    private func internals(_ model: LiveViewModel) -> String {
        "gate=\(String(describing: gate(model))) generation=\(String(describing: generation(model)))"
            + " completed=\(model.performance.completed) dropped=\(model.performance.dropped)"
            + " frame==nil: \(model.frame == nil) active=\(model.active?.id ?? "nil")"
            + " failure=\(model.failure?.title ?? "nil")"
    }
}
