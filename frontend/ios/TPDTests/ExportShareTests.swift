//  ExportShareTests.swift
//  The parts of Share that are pure logic: which frames the export gets for free from the
//  preview's cache, that add-only authorization gates the write, and that a failed save keeps
//  the finished file. The end-to-end run — a real clip through a real export into the camera
//  roll — is verified on the simulator; at ~7 s a frame it is not a suite anybody would run.

import CoreGraphics
import Photos
import XCTest
@testable import TPD

final class ExportShareTests: XCTestCase {
    private var scratch: [URL] = []

    override func tearDown() {
        for url in scratch { try? FileManager.default.removeItem(at: url) }
        scratch = []
        super.tearDown()
    }

    /// The headline saving, at the level where it can actually go wrong. A snapshot must key
    /// frames the way the cache that produced them did: keyed even slightly differently it is a
    /// dictionary that never hits, which costs minutes and looks like nothing at all. The tally
    /// beside it is what the card and the log report the saving with.
    @MainActor
    func testASnapshotServesExactlyTheFramesThePreviewCacheHeldAndTheTallyCountsThem() {
        let cache = ResultCache(frameRate: 30)
        cache.insert(Self.result(), at: cache.index(for: 0.2))
        let snapshot = ResultSnapshot(frameRate: cache.frameRate, entries: cache.snapshot)
        let tally = ExportTally()
        // Two times inside frame 6's period and one outside it, counted as the export counts.
        for seconds in [0.2, 0.21, 0.24] { tally.record(reused: snapshot.result(at: seconds) != nil) }

        XCTAssertEqual(snapshot.result(at: 0.2), Self.result())
        XCTAssertEqual(snapshot.result(at: 0.21), Self.result(), "a time inside frame 6 missed")
        XCTAssertNil(snapshot.result(at: 0.24), "a neighbouring frame was served frame 6")
        XCTAssertNil(ResultSnapshot().result(at: 0.2), "an empty snapshot served something")
        XCTAssertEqual(tally.value.reused, 2, "a snapshot hit was counted as an inference")
        XCTAssertEqual(tally.value.inferred, 1, "a miss was counted as reuse")
    }

    /// Add-only authorization gates the write, and `.limited` counts as a grant: under
    /// `.addOnly` it is one permission reported two ways. A refusal must not reach
    /// `performChanges` at all, and `requestAccess` must reach the same verdict alone — the
    /// export asks it up front, before spending minutes it would then have to throw away.
    func testTheSaverWritesOnlyOnceAddAccessIsGranted() async throws {
        for status in [PHAuthorizationStatus.authorized, .limited] {
            let written = Box<URL?>(nil)
            let saver = PhotoLibrarySaver(authorize: { status }, write: { written.value = $0 })
            try await saver.requestAccess()
            try await saver.save(Self.fixture)
            XCTAssertEqual(written.value, Self.fixture, "status \(status.rawValue) did not write")
        }
        for status in [PHAuthorizationStatus.denied, .restricted, .notDetermined] {
            let written = Box<URL?>(nil)
            let saver = PhotoLibrarySaver(authorize: { status }, write: { written.value = $0 })
            await assertThrows(.notAuthorized, "status \(status.rawValue)") {
                try await saver.requestAccess()
            }
            await assertThrows(.notAuthorized, "status \(status.rawValue)") {
                try await saver.save(Self.fixture)
            }
            XCTAssertNil(written.value, "status \(status.rawValue) still wrote to the library")
        }
        let refused = CocoaError(.fileNoSuchFile)
        await assertThrows(.failed(refused.localizedDescription), "a refused write") {
            try await PhotoLibrarySaver(authorize: { .authorized },
                                        write: { _ in throw refused }).save(Self.fixture)
        }
    }

    /// A save that fails for any reason other than authorization must keep the exported file:
    /// re-running `performChanges` is seconds and re-running the export is minutes. It is
    /// deleted only once the asset is really in the library. (An *authorization* refusal is
    /// caught before the export instead — granting it in Settings relaunches the app.)
    @MainActor
    func testAFailedSaveKeepsTheFileAndRetryingSavesAndCleansUp() async {
        let file = scratchURL()
        FileManager.default.createFile(atPath: file.path, contents: Data([0]))
        let healthy = Box(false)
        let model = ExportViewModel(saver: PhotoLibrarySaver(authorize: { .authorized }, write: { _ in
            if !healthy.value { throw CocoaError(.fileWriteOutOfSpace) }
        }))
        await model.store(file)

        guard case .failed = model.phase else {
            return XCTFail("a failed save surfaced \(model.phase), not a failure")
        }
        XCTAssertEqual(model.exported, file, "the export was thrown away on a failed save")
        XCTAssertTrue(FileManager.default.fileExists(atPath: file.path),
                      "the file a retry needs was deleted")

        healthy.value = true
        model.retrySave()
        for _ in 0..<50 where model.phase != .saved { try? await Task.sleep(nanoseconds: 20_000_000) }
        XCTAssertEqual(model.phase, .saved)
        XCTAssertNil(model.exported)
        XCTAssertFalse(FileManager.default.fileExists(atPath: file.path),
                       "a saved export left its temp copy behind")
    }

    // MARK: - Fixtures

    private static let fixture = URL(fileURLWithPath: "/tmp/tpd-overlay.mp4")

    private static func result() -> TPDResult {
        TPDResult(frameSize: CGSize(width: 160, height: 120),
                  bbox: CGRect(x: 30, y: 30, width: 70, height: 60),
                  keypoints: [TPDKeypoint(position: CGPoint(x: 60, y: 50), visibility: 2)],
                  probabilities: [0.9, 0.1], labels: ["serve", "volley"], bestIndex: 0)
    }

    private func scratchURL() -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("tpd-test-\(UUID().uuidString).mp4")
        scratch.append(url)
        return url
    }

    private func assertThrows(_ expected: PhotoLibrarySaveError, _ what: String,
                              file: StaticString = #filePath, line: UInt = #line,
                              _ body: () async throws -> Void) async {
        do {
            try await body()
            XCTFail("\(what) was accepted", file: file, line: line)
        } catch {
            XCTAssertEqual(error as? PhotoLibrarySaveError, expected, what, file: file, line: line)
        }
    }

    /// A mutable cell for the `@Sendable` closures the saver is built from.
    private final class Box<T>: @unchecked Sendable {
        var value: T
        init(_ value: T) { self.value = value }
    }
}
