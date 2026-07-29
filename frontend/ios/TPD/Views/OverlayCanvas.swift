//  OverlayCanvas.swift
//  Everything the model predicted, drawn over the preview in one Canvas pass.

import SwiftUI

/// Stroke/dot/pill drawing for one `TPDResult`. It re-derives no geometry: the
/// keypoints already arrive in full-frame pixels (`TPDInferenceEngine` walks them
/// back out through `LetterboxMath.letterboxToFrame`), so the only transform left
/// is `FrameFit`, shared with `FramePreview`.
struct OverlayCanvas: View {
    let result: TPDResult?
    let options: OverlayOptions

    private let dotRadius: CGFloat = 4

    var body: some View {
        Canvas { context, size in
            guard let result, result.frameSize.width > 0, result.frameSize.height > 0 else { return }
            let fit = FrameFit(frame: result.frameSize, view: size)
            let box = fit.map(result.bbox)

            if options.boundingBox {
                context.stroke(Path(roundedRect: box, cornerRadius: 4),
                               with: .color(.yellow), lineWidth: 2.5)
            }
            if options.keypoints {
                // `drawableKeypoints` is the backend's `visibility <= 0` skip; a
                // channel the heatmap never fired on must not be drawn at (0, 0).
                for keypoint in result.drawableKeypoints {
                    let centre = fit.map(keypoint.position)
                    let dot = CGRect(x: centre.x - dotRadius, y: centre.y - dotRadius,
                                     width: dotRadius * 2, height: dotRadius * 2)
                    context.fill(Path(ellipseIn: dot), with: .color(.cyan))
                    context.stroke(Path(ellipseIn: dot),
                                   with: .color(.black.opacity(0.6)), lineWidth: 1)
                }
            }
            if let caption = caption(for: result) {
                draw(caption, in: &context, above: box, bounds: size)
            }
        }
        // Purely decorative: taps belong to the controls underneath it.
        .allowsHitTesting(false)
    }

    /// Two independent toggles, one pill — so turning the class off leaves a bare
    /// percentage rather than an empty badge.
    private func caption(for result: TPDResult) -> String? {
        var parts: [String] = []
        if options.label { parts.append(result.label) }
        if options.confidence { parts.append(String(format: "%.0f%%", result.confidence * 100)) }
        return parts.isEmpty ? nil : parts.joined(separator: "   ")
    }

    /// Sits on the box's top edge, and flips below it when the box is against the
    /// top of the screen, so the reading never leaves the view.
    private func draw(_ caption: String, in context: inout GraphicsContext,
                      above box: CGRect, bounds: CGSize) {
        // The fill is explicit: `resolve` bakes in the *environment's* colour,
        // which is white here, and white on the yellow pill is unreadable.
        let text = context.resolve(Text(caption).font(.system(size: 15, weight: .semibold))
            .foregroundColor(.black))
        let measured = text.measure(in: bounds)
        let pill = CGSize(width: measured.width + 20, height: measured.height + 10)
        let x = min(max(6, box.minX), max(6, bounds.width - pill.width - 6))
        var y = box.minY - pill.height - 6
        if y < 6 { y = min(box.minY + 6, max(6, bounds.height - pill.height - 6)) }
        let rect = CGRect(origin: CGPoint(x: x, y: y), size: pill)
        context.fill(Path(roundedRect: rect, cornerRadius: pill.height / 2), with: .color(.yellow))
        context.draw(text, at: CGPoint(x: rect.midX, y: rect.midY), anchor: .center)
    }
}
