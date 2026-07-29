//  PhotoLibrarySaver.swift
//  The step `VideoExporter` deliberately stops short of, and the only place in the app that
//  touches PHPhotoLibrary: a finished file in the temp directory -> an asset in the camera roll.

import Foundation
import Photos

enum PhotoLibrarySaveError: Error, Equatable, LocalizedError {
    case notAuthorized, failed(String)

    /// The denial message names the fix *and* promises the retry is cheap, because it is: the
    /// caller keeps the exported file, so coming back from Settings costs one tap.
    var errorDescription: String? {
        switch self {
        case .notAuthorized:
            return "TPD is not allowed to add to your photo library. Turn on Add Photos Only "
                + "for TPD in Settings › Privacy & Security › Photos, then tap Save again — the "
                + "overlaid video is still here, so nothing has to be analysed twice."
        case .failed(let why): return "The camera roll refused the file: \(why)"
        }
    }
}

/// Authorize, then write. Both halves are injected because the alternative is a type only
/// reachable by hand: the simulator offers each authorization prompt once per install, so a
/// denial is a branch no test could otherwise reach twice.
struct PhotoLibrarySaver: Sendable {
    var authorize: @Sendable () async -> PHAuthorizationStatus
    var write: @Sendable (URL) async throws -> Void

    /// **`.addOnly`**, matching the single Photos key in Info.plist. `.readWrite` would also
    /// require `NSPhotoLibraryUsageDescription`, and read access to save a file the app itself
    /// produced is access this app has no use for. The change request's placeholder goes
    /// unused for the same reason: there is no library to go looking for the asset in.
    static let live = PhotoLibrarySaver(
        authorize: { await PHPhotoLibrary.requestAuthorization(for: .addOnly) },
        write: { url in
            try await PHPhotoLibrary.shared().performChanges {
                PHAssetChangeRequest.creationRequestForAssetFromVideo(atFileURL: url)
            }
        })

    /// `.authorized` and `.limited` both permit an add — under add-only they are one grant
    /// reported two ways, and treating `.limited` as a refusal would block a user who said yes.
    /// The status is bound once: authorizing twice would put a second prompt on screen.
    func save(_ url: URL) async throws {
        let status = await authorize()
        guard status == .authorized || status == .limited else {
            throw PhotoLibrarySaveError.notAuthorized
        }
        do {
            try await write(url)
        } catch {
            throw PhotoLibrarySaveError.failed(error.localizedDescription)
        }
    }
}
