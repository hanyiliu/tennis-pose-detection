//  FramePreview.swift
//  Draws the frame ourselves instead of hanging an AVCaptureVideoPreviewLayer
//  behind the overlay. That is deliberate: the preview layer applies its own
//  gravity/mirroring/rotation that the overlay would then have to re-derive, and
//  it shows nothing at all in the Simulator, where frames come from a file. One
//  bitmap drawn by us collapses the whole mapping to `FrameFit`, and device and
//  Simulator become the same code path.

import CoreGraphics
import SwiftUI

/// Where a frame lands inside a view: the single frame-space -> view-space
/// transform in the app. `FramePreview` positions the bitmap with it and
/// `OverlayCanvas` applies it numerically, so the picture and the boxes drawn on
/// top of it cannot disagree — there is only one aspect-fit to get wrong.
struct FrameFit: Equatable {
    /// Uniform, so the aspect ratio survives; `.fit`, so nothing is cropped away.
    let scale: CGFloat
    /// Top-left of the letterboxed picture inside the view, in view points.
    let origin: CGPoint

    init(frame: CGSize, view: CGSize) {
        let width = max(frame.width, 1), height = max(frame.height, 1)
        scale = max(min(view.width / width, view.height / height), 0)
        origin = CGPoint(x: (view.width - width * scale) / 2,
                         y: (view.height - height * scale) / 2)
    }

    /// Frame pixels (top-left origin, the space `TPDResult` is expressed in) ->
    /// view points. SwiftUI's y also grows downward, so there is no flip here.
    func map(_ point: CGPoint) -> CGPoint {
        CGPoint(x: origin.x + point.x * scale, y: origin.y + point.y * scale)
    }

    func map(_ rect: CGRect) -> CGRect {
        CGRect(origin: map(rect.origin),
               size: CGSize(width: rect.width * scale, height: rect.height * scale))
    }
}

/// Renders exactly the bitmap the engine measured — `LiveViewModel` publishes the
/// image and the result it produced together, so the overlay is never drawn over
/// a newer frame than the one it describes.
struct FramePreview: View {
    let image: CGImage?

    var body: some View {
        GeometryReader { proxy in
            if let image {
                let size = CGSize(width: image.width, height: image.height)
                let fit = FrameFit(frame: size, view: proxy.size)
                Image(decorative: image, scale: 1, orientation: .up)
                    .resizable()
                    .interpolation(.medium)
                    .frame(width: size.width * fit.scale, height: size.height * fit.scale)
                    .position(x: proxy.size.width / 2, y: proxy.size.height / 2)
            }
        }
    }
}
