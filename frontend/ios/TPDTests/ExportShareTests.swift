//  ExportShareTests.swift
//  The parts of Share that are pure logic: which frames the export gets for free from the
//  preview's cache, that add-only authorization gates the write, and that a refused save keeps
//  the finished file so a grant costs one tap rather than another export. The end-to-end run —
//  a real clip through a real export into the camera roll — is verified on the simulator; it
//  costs ~9.7 s a frame, which is not a suite anybody would run.

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
    /// `.addOnly` it is one permission reported two ways, and refusing it would block a user
    /// who has said yes. A refusal must not reach `performChanges` at all.
    func testTheSaverWritesOnlyOnceAddAccessIsGranted() async throws {
        for status in [PHAuthorizationStatus.authorized, .limited] {
            let written = Box<URL?>(nil)
            try await PhotoLibrarySaver(authorize: { status },
                                        write: { written.value = $0 }).save(Self.fixture)
            XCTAssertEqual(written.value, Self.fixture, "status \(status.rawValue) did not write")
        }
        for status in [PHAuthorizationStatus.denied, .restricted, .notDetermined] {
            let written = Box<URL?>(nil)
            let thrown = await error(PhotoLibrarySaver(authorize: { status },
                                                       write: { written.value = $0 }))
            XCTAssertEqual(thrown, .notAuthorized, "status \(status.rawValue) was accepted")
            XCTAssertNil(written.value, "status \(status.rawValue) still wrote to the library")
        }
        let refused = CocoaError(.fileNoSuchFile)
        let thrown = await error(PhotoLibrarySaver(authorize: { .authorized },
                                                   write: { _ in throw refused }))
        XCTAssertEqual(thrown, .failed(refused.localizedDescription))
    }

    /// The recovery this codebase has got wrong once already: "fix it in Settings" is honest
    /// only if coming back works. The exported file is kept on failure, so a retry is one
    /// `performChanges`; it is deleted only once the asset is really in the library.
    @MainActor
    func testARefusedSaveKeepsTheFileAndRetryingAfterAGrantSavesAndCleansUp() async {
        let file = scratchURL()
        FileManager.default.createFile(atPath: file.path, contents: Data([0]))
        let granted = Box(false)
        let model = ExportViewModel(saver: PhotoLibrarySaver(
            authorize: { granted.value ? .authorized : .denied }, write: { _ in }))
        await model.store(file)

        guard case .failed(let message) = model.phase else {
            return XCTFail("a denied save surfaced \(model.phase), not a failure")
        }
        XCTAssertTrue(message.contains("Settings"), "the denial message names no way out")
        XCTAssertEqual(model.exported, file, "the export was thrown away on a refused save")
        XCTAssertTrue(FileManager.default.fileExists(atPath: file.path),
                      "the file a retry needs was deleted")

        granted.value = true
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

    /// `nil` means the save was allowed through, which every caller here treats as a failure.
    private func error(_ saver: PhotoLibrarySaver) async -> PhotoLibrarySaveError? {
        do { try await saver.save(Self.fixture); return nil } catch {
            return error as? PhotoLibrarySaveError
        }
    }

    /// A mutable cell for the `@Sendable` closures the saver is built from.
    private final class Box<T>: @unchecked Sendable {
        var value: T
        init(_ value: T) { self.value = value }
    }
}
