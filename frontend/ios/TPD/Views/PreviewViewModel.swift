//  PreviewViewModel.swift
//  One pick -> one inference -> one frame on screen. The still path; video is
//  PR9/PR10 and stops at a placeholder here.

import CoreImage
import Foundation
import Observation
// PhotosUI declares `PhotosPickerItem` against SwiftUI; without SwiftUI in scope
// the type is invisible even with PhotosUI imported.
import PhotosUI
import SwiftUI

/// Decode, orient, predict and rasterize — the whole expensive half, off the
/// main actor.
///
/// It is a second actor rather than a reuse of the live view's `InferenceWorker`
/// for two reasons: that one belongs to `LiveViewModel` and is held for the
/// app's lifetime, while this one is created with the preview and dies with it,
/// so a still's models are given back when the user exits; and this path starts
/// from a `CIImage` decoded here, not from a capture `CVPixelBuffer`. What is
/// *not* different is the load discipline — see below.
actor StillInferenceWorker {
    /// Built inside `render`, never by `init`. An actor's `init` body runs
    /// synchronously in the **caller's** isolation domain rather than hopping to
    /// the actor, so `TPDInferenceEngine()` in a stored-property initializer
    /// performs both `MLModel(contentsOf:)` loads on whichever thread said
    /// `StillInferenceWorker()` — main — and freezes the UI. The live view
    /// already paid for that lesson; this is the same trap on a new path.
    private var engine: TPDInferenceEngine?
    /// Display rasterization only. Pointedly *not* the engine's context, whose
    /// colour management is switched off for numeric parity with the Python
    /// pipeline; those settings are for matching floats, not for looking right.
    /// Built lazily for the same isolation reason as `engine`.
    private var display: CIContext?

    /// The whole job in one hop: file bytes in, a frame and the result measured
    /// **from that same frame** out. `Data` and `RenderedFrame` are the only
    /// things that cross the boundary; the `CIImage` never leaves this actor.
    func render(_ data: Data) throws -> RenderedFrame {
        let display = display ?? CIContext()
        self.display = display
        let engine = try engine ?? TPDInferenceEngine()
        self.engine = engine
        // Orientation is applied by `upright`, before anything measures the
        // image. Everything downstream — bbox, keypoints, the bitmap on screen —
        // is expressed in this one upright space.
        let image = try PickedImage.upright(from: data)
        let result = try engine.predict(frame: image)
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
        /// A movie was picked. Deliberately inert in this PR — see the view.
        case video
        case failed(String)
    }

    private(set) var stage: Stage = .loading
    /// Independent of the live view's copy on purpose: the toggles a user set
    /// over a still are about that still, and must not reach back and change
    /// what the camera screen shows underneath.
    var overlay = OverlayOptions()

    private let worker = StillInferenceWorker()

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
        case .video: stage = .video
        }
    }

    private func loadStill(_ item: PhotosPickerItem) async {
        do {
            guard let data = try await item.loadTransferable(type: Data.self) else {
                throw MediaPickerError.transferFailed
            }
            // The user can exit while a big asset is still arriving; publishing
            // into a view that is on its way out is pointless work.
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
