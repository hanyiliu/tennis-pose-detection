//  PreviewViewModel.swift
//  One pick -> one frame on screen. Stills run their single inference here;
//  movies are copied to a file and handed to `VideoPreviewModel`.

import CoreImage
import CoreTransferable
import Foundation
import Observation
// PhotosUI declares `PhotosPickerItem` against SwiftUI; without SwiftUI in scope
// the type is invisible even with PhotosUI imported.
import PhotosUI
import SwiftUI

/// Decode, orient, predict and rasterize — the whole expensive half, off the
/// main actor, and **one instance for the whole app**.
///
/// It is a second actor rather than a reuse of the live view's `InferenceWorker`
/// because that one is fed capture `CVPixelBuffer`s and this path starts from
/// file bytes decoded here. What it is no longer is per-presentation. Every
/// `PhotoPreviewView` used to build its own worker and therefore its own
/// `TPDInferenceEngine`: a second copy of both Core ML models, a second
/// `CIContext` and a full-resolution bitmap, all still resident while the next
/// pick built its own set. Four picks abandoned after 0.7 s each held
/// phys_footprint at 367 MB against a 212 MB baseline, with the live capture
/// loop restarting underneath them. `shared` is now the only way in, so the
/// models load once and renders serialize instead of racing each other and the
/// camera for the same cores.
actor StillInferenceWorker {
    static let shared = StillInferenceWorker()

    /// Built inside `render`, never by `init`. An actor's `init` body runs
    /// synchronously in the **caller's** isolation domain rather than hopping to
    /// the actor, so `TPDInferenceEngine()` in a stored-property initializer
    /// performs both `MLModel(contentsOf:)` loads on whichever thread said
    /// `StillInferenceWorker()` — main — and freezes the UI. The live view
    /// already paid for that lesson; this is the same trap on a new path, and
    /// `shared` does not defuse it: a lazy global runs its initializer in
    /// whatever domain touched it first, which here is the main actor.
    private var engine: TPDInferenceEngine?
    /// Display rasterization only. Pointedly *not* the engine's context, whose
    /// colour management is switched off for numeric parity with the Python
    /// pipeline; those settings are for matching floats, not for looking right.
    /// Built lazily for the same isolation reason as `engine`.
    private var display: CIContext?
    /// How many times the engine has been built. One, forever — the observable
    /// half of "shared", and what a test can assert instead of trusting the
    /// `private init`.
    private(set) var modelLoadCount = 0

    private init() {}

    /// The whole job in one hop: file bytes in, a frame and the result measured
    /// **from that same frame** out. `Data` and `RenderedFrame` are the only
    /// things that cross the boundary; the `CIImage` never leaves this actor.
    ///
    /// **Cooperatively cancellable**, which is the point of the checks below.
    /// A caller's `try Task.checkCancellation()` before this call only ever
    /// caught an exit that happened before the hop; once inside, nothing looked
    /// again, so leaving the preview cancelled the task and bought nothing —
    /// a 12 MP still cancelled at 0.5 s still ran to completion 70 s later.
    /// Each stage is long enough to be worth not entering: the first check
    /// covers a render queued behind another one (the actor serializes, so it
    /// may have been waiting a whole render), then the model load, then
    /// decode + orientation, then inference — after which the only thing left
    /// is rasterizing a full-resolution bitmap for a screen that is gone.
    /// The stages themselves are not interruptible; `predict` is one call.
    func render(_ data: Data) throws -> RenderedFrame {
        try Task.checkCancellation()
        let display = display ?? CIContext()
        self.display = display
        let engine = try engine ?? makeEngine()
        self.engine = engine
        try Task.checkCancellation()
        // Orientation is applied by `upright`, before anything measures the
        // image. Everything downstream — bbox, keypoints, the bitmap on screen —
        // is expressed in this one upright space.
        let image = try PickedImage.upright(from: data)
        try Task.checkCancellation()
        let result = try engine.predict(frame: image)
        try Task.checkCancellation()
        guard let raster = display.createCGImage(image, from: image.extent) else {
            throw MediaPickerError.undecodableImage
        }
        // The only observable number this path produces. `frameSize` is the
        // orientation check in one value: a portrait photo whose EXIF tag was
        // ignored arrives here transposed, and that shows up as WxH swapped long
        // before the overlay looks wrong. The bbox beside it is directly
        // comparable to `norm_bbox_to_xyxy_pixels` on the same file.
        #if DEBUG
        NSLog("TPD still: frame %.0fx%.0f bbox xyxy %.0f %.0f %.0f %.0f",
              result.frameSize.width, result.frameSize.height, result.bbox.minX,
              result.bbox.minY, result.bbox.maxX, result.bbox.maxY)
        #endif
        return RenderedFrame(image: raster, result: result)
    }

    /// The video path's way in, here rather than in a new actor so a clip analysed
    /// after a photo reuses models already resident. It returns only the result;
    /// `FrameRasterizer` draws the picture. The second check is the last boundary
    /// there is — `predict` is one call and runs to the end.
    func analyse(_ frame: VideoFrame) throws -> TPDResult {
        try Task.checkCancellation()
        let engine = try engine ?? makeEngine()
        self.engine = engine
        try Task.checkCancellation()
        return try engine.predict(pixelBuffer: frame.pixelBuffer)
    }

    private func makeEngine() throws -> TPDInferenceEngine {
        let engine = try TPDInferenceEngine()
        modelLoadCount += 1
        // Cheap standing proof, the same one the live view keeps: if this load
        // ever migrates back into an initializer this prints YES.
        #if DEBUG
        NSLog("TPD still model load #%d ran on the main thread: %@",
              modelLoadCount, Thread.isMainThread ? "YES" : "NO")
        #endif
        return engine
    }
}

/// Photo Preview View state. `@MainActor` throughout, like `LiveViewModel`: the
/// only thing that leaves main is `StillInferenceWorker`.
@MainActor
@Observable
final class PreviewViewModel {
    /// What the screen is showing. One enum rather than a bag of flags, so
    /// "loading and also failed" is not a state that can be reached.
    enum Stage {
        case loading
        /// The picture and the result measured from it, published together so an
        /// overlay can never be drawn over a frame it does not describe.
        case still(RenderedFrame)
        /// A movie, copied out of the library; playback, scrubbing and per-frame
        /// analysis are `VideoPreviewModel`'s from here.
        case video(URL)
        case failed(String)
    }

    private(set) var stage: Stage = .loading
    /// Independent of the live view's copy on purpose: the toggles a user set
    /// over a still are about that still, and must not reach back and change
    /// what the camera screen shows underneath.
    var overlay = OverlayOptions()

    /// The app's one still worker, not this presentation's. A fresh view model
    /// per pick is still what makes `.task` a once-per-pick load; what it must
    /// not also mean is a fresh pair of Core ML models.
    private let worker = StillInferenceWorker.shared

    /// Runs exactly once per presentation, driven by the view's `.task`. Never
    /// throws: every failure becomes a state the screen can render.
    ///
    /// Both awaits are real suspensions off this actor — `loadTransferable` is
    /// nonisolated and does its I/O on the cooperative pool, and `render` hops to
    /// the worker's executor. Main is therefore free for the whole download and
    /// the whole three-stage pass, which matters more here than in the live view:
    /// a full-resolution 12 MP still is a much larger stage-1 resample than a
    /// capture frame, and an iCloud asset can take seconds to arrive.
    func load(_ media: PickedMedia) async {
        switch media.kind {
        case .image: await loadStill(media.item)
        case .video: await loadVideo(media.item)
        }
    }

    /// A movie is loaded as a **file**, not as `Data`: `AVURLAsset` needs a URL to
    /// seek in, and a 30 s 4K recording through memory would cost hundreds of MB.
    private func loadVideo(_ item: PhotosPickerItem) async {
        do {
            guard let movie = try await item.loadTransferable(type: PickedMovie.self) else {
                throw MediaPickerError.transferFailed
            }
            try Task.checkCancellation()
            stage = .video(movie.url)
        } catch is CancellationError { return } catch {
            stage = .failed(Self.message(for: error))
        }
    }

    private func loadStill(_ item: PhotosPickerItem) async {
        do {
            guard let data = try await item.loadTransferable(type: Data.self) else {
                throw MediaPickerError.transferFailed
            }
            // The user can exit while a big asset is still arriving; publishing
            // into a view that is on its way out is pointless work. `render`
            // keeps checking past this point — see it.
            try Task.checkCancellation()
            stage = .still(try await worker.render(data))
        } catch is CancellationError {
            return
        } catch {
            stage = .failed(Self.message(for: error))
        }
    }

    /// The engine's own errors already read as instructions ("run `make export`"),
    /// so they are passed through rather than rewritten.
    private static func message(for error: Error) -> String {
        (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
    }
}

/// A picked movie, on disk. The URL the picker hands the importer is valid only
/// for that call, so the copy is mandatory rather than cautious;
/// `VideoPreviewModel.stop()` deletes it again.
struct PickedMovie: Transferable {
    let url: URL

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(importedContentType: .movie) { received in
            let suffix = received.file.pathExtension.isEmpty ? "mov" : received.file.pathExtension
            let copy = FileManager.default.temporaryDirectory
                .appendingPathComponent("tpd-\(UUID().uuidString).\(suffix)")
            try FileManager.default.copyItem(at: received.file, to: copy)
            return PickedMovie(url: copy)
        }
    }
}
