//  OverlayCanvas.swift
//  Everything the model predicted, drawn over the preview in one Canvas pass.

import SwiftUI

/// The live rasterizer, and *only* a rasterizer: it derives no geometry of its own. Every
/// position — the mapped box, the visible keypoints, the caption and where its pill sits — comes
/// out of `OverlayGeometry`, the same value `OverlayRenderer` burns into exported frames. Before
/// that the two paths computed the same arithmetic twice, which agreed only by transcription; a
/// divergence would have shown up as a screenshot that did not match the exported video, which is
/// nobody's first suspicion. `OverlayParityTests` renders one result through both and compares.
struct OverlayCanvas: View {
    let result: TPDResult?
    let options: OverlayOptions
    /// Live view points, the units the overlay's constants were chosen in; the exporter passes
    /// `.fitting(width:)` instead, because 2.5 pt of stroke is a hairline across 1080 px.
    var style = OverlayStyle()

    var body: some View {
        Canvas { context, size in
            guard let result else { return }
            OverlayCanvas.draw(OverlayGeometry(result: result, options: options,
                                               size: size, style: style),
                               in: &context, style: style)
        }
        // Purely decorative: taps belong to the controls underneath it.
        .allowsHitTesting(false)
    }

    /// The SwiftUI twin of `OverlayRenderer.draw(_:in:style:)` — same geometry, same palette, and
    /// nothing left but the drawing calls, which are the one thing a `Canvas` and a `CGContext`
    /// genuinely cannot share. The colours are read off `OverlayRenderer` rather than named
    /// (`.yellow`, `.cyan`) so the two files cannot be repainted apart either.
    static func draw(_ geometry: OverlayGeometry, in context: inout GraphicsContext,
                     style: OverlayStyle) {
        if let box = geometry.box {
            context.stroke(Path(roundedRect: box, cornerRadius: style.boxCornerRadius),
                           with: .color(Color(cgColor: OverlayRenderer.boxColor)),
                           lineWidth: style.lineWidth)
        }
        for dot in geometry.dots {
            let rect = CGRect(x: dot.center.x - dot.radius, y: dot.center.y - dot.radius,
                              width: dot.radius * 2, height: dot.radius * 2)
            context.fill(Path(ellipseIn: rect),
                         with: .color(Color(cgColor: OverlayRenderer.dotColor)))
            context.stroke(Path(ellipseIn: rect),
                           with: .color(Color(cgColor: OverlayRenderer.dotOutlineColor)),
                           lineWidth: max(style.lineWidth / 2.5, 1))
        }
        guard let pill = geometry.pill else { return }
        context.fill(Path(roundedRect: pill.rect, cornerRadius: pill.rect.height / 2),
                     with: .color(Color(cgColor: OverlayRenderer.boxColor)))
        // The fill is explicit: `resolve` bakes in the *environment's* colour, which is white
        // here, and white on the yellow pill is unreadable.
        let text = context.resolve(Text(pill.text)
            .font(.system(size: pill.fontSize, weight: .semibold))
            .foregroundColor(Color(cgColor: OverlayRenderer.captionColor)))
        context.draw(text, at: CGPoint(x: pill.rect.midX, y: pill.rect.midY), anchor: .center)
    }
}
