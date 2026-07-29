//  PhotoLibrarySaver.swift
//  The step `VideoExporter` deliberately stops short of, and the only place in the app that
//  touches PHPhotoLibrary: a finished file in the temp directory -> an asset in the camera roll.

import Foundation
import Photos

enum PhotoLibrarySaveError: Error, Equatable, LocalizedError {
    case notAuthorized, failed(String)

    /// The denial message names the fix and says what granting it costs, rather than promising
    /// an in-app retry: **iOS relaunches the app when a Photos switch changes in Settings**, as
    /// measured here — the process died the instant access was revoked.
    var errorDescription: String? {
        switch self {
        case .notAuthorized:
            return "TPD cannot add to your photo library, so there is nothing to export to. "
                + "Turn on Add Photos Only for TPD in Settings › Privacy & Security › Photos; "
                + "iOS restarts TPD when you do, then pick the clip and tap Share again."
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

    /// The authorization half alone, so the caller can ask **before** the export rather than
    /// after it: a refusal 100 s in has already cost the whole burn-in, and no in-app retry can
    /// hand that back because granting it in Settings relaunches the app. `.authorized` and
    /// `.limited` both permit an add — under add-only they are one grant reported two ways —
    /// and the status is bound once, since authorizing twice shows a second prompt.
    func requestAccess() async throws {
        let status = await authorize()
        guard status == .authorized || status == .limited else {
            throw PhotoLibrarySaveError.notAuthorized
        }
    }

    func save(_ url: URL) async throws {
        try await requestAccess()
        do {
            try await write(url)
        } catch {
            throw PhotoLibrarySaveError.failed(error.localizedDescription)
        }
    }
}
