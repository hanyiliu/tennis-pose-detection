//  OverlayRenderer.swift
//  The overlay's GEOMETRY, and the CoreGraphics rasterizer the exporter draws with.
//  `OverlayGeometry` is the one place box/dot/pill placement is computed: two drawing paths drift,
//  and the drift stays invisible until someone diffs a screenshot against an exported frame. Both
//  rasterizers — this one and `OverlayCanvas` — now consume the same value, and
//  `OverlayParityTests` renders a result through both and compares where the pixels landed.

import CoreGraphics
import CoreText
import Foundation
import UIKit

/// Weights for one overlay pass — never positions, which come from `FrameFit`. The live view draws
/// into view points on a ~390 pt phone and the exporter into the video's pixels, so at 1080 wide
/// the live constants would be hairlines under tiny type. `fitting` never scales *down*.
struct OverlayStyle: Equatable, Sendable {
    var scale: CGFloat = 1
    var lineWidth: CGFloat { 2.5 * scale }
    var dotRadius: CGFloat { 4 * scale }
    var fontSize: CGFloat { 15 * scale }
    var boxCornerRadius: CGFloat { 4 * scale }
    var margin: CGFloat { 6 * scale }
    var pillPadding: CGSize { CGSize(width: 20 * scale, height: 10 * scale) }
    /// 390 is the canvas width `OverlayCanvas`'s constants were chosen against.
    static func fitting(width: CGFloat) -> OverlayStyle { .init(scale: max(1, width / 390)) }
}

/// Everything the overlay draws, in destination coordinates and holding no drawing API: a
/// `Canvas` and a `CGContext` share one computation, and tests assert placement without pixels.
struct OverlayGeometry: Equatable {
    struct Dot: Equatable { var center: CGPoint; var radius: CGFloat }
    struct Pill: Equatable { var rect: CGRect; var text: String; var fontSize: CGFloat }
    /// `nil` with its toggle off; the pill still hangs off it, so hiding the outline moves nothing.
    var box: CGRect?
    var dots: [Dot] = []
    var pill: Pill?
    /// `measure` defaults to CoreText for **both** rasterizers, so the pill is sized once: SwiftUI
    /// resolves its own `Text` to draw the caption, never to place it. It stays injectable only so
    /// tests can assert placement against arithmetic instead of against a font's metrics.
    init(result: TPDResult, options: OverlayOptions, size: CGSize,
         style: OverlayStyle = OverlayStyle(),
         measure: (String, CGFloat) -> CGSize = OverlayRenderer.measure) {
        guard result.frameSize.width > 0, result.frameSize.height > 0 else { return }
        let fit = FrameFit(frame: result.frameSize, view: size)
        let bounds = result.bbox.map { fit.map($0) }
        if options.boundingBox { box = bounds }
        if options.keypoints {
            // The backend's `visibility <= 0` skip: a channel that never fired must not be drawn.
            dots = result.drawableKeypoints.map {
                Dot(center: fit.map($0.position), radius: style.dotRadius)
            }
        }
        guard let caption = Self.caption(for: result, options) else { return }
        let measured = measure(caption, style.fontSize)
        let pillSize = CGSize(width: measured.width + style.pillPadding.width,
                              height: measured.height + style.pillPadding.height)
        let margin = style.margin
        // A classifier has no box to hang the caption off, and the top-left it would fall back
        // to is under the diagnostics pill. Centred: this reading is all such a model produces.
        guard let bounds else {
            pill = Pill(rect: CGRect(origin: CGPoint(x: (size.width - pillSize.width) / 2,
                                                     y: (size.height - pillSize.height) / 2),
                                     size: pillSize), text: caption, fontSize: style.fontSize)
            return
        }
        let x = min(max(margin, bounds.minX), max(margin, size.width - pillSize.width - margin))
        var y = bounds.minY - pillSize.height - margin
        // Flips below a box against the top of the frame, so the reading stays in shot.
        if y < margin {
            y = min(bounds.minY + margin, max(margin, size.height - pillSize.height - margin))
        }
        pill = Pill(rect: CGRect(origin: CGPoint(x: x, y: y), size: pillSize),
                    text: caption, fontSize: style.fontSize)
    }

    /// Two toggles, one pill — class off leaves a bare percentage, not an empty badge.
    static func caption(for result: TPDResult, _ options: OverlayOptions) -> String? {
        var parts: [String] = []
        if options.label { parts.append(result.label) }
        if options.confidence { parts.append(String(format: "%.0f%%", result.confidence * 100)) }
        return parts.isEmpty ? nil : parts.joined(separator: "   ")
    }
}

/// Rasterizes an `OverlayGeometry`. The context must already be **top-left-origin** — the space
/// `TPDResult` is expressed in, which is what `VideoExporter` flips its bitmap context into;
/// forgetting that mirrors the overlay rather than crashing. The palette is sRGB literals because
/// `LiveCameraView` pins dark mode, and a literal needs no trait collection off main.
enum OverlayRenderer {
    static let boxColor = CGColor(srgbRed: 1, green: 0.839, blue: 0.039, alpha: 1)
    static let dotColor = CGColor(srgbRed: 0.392, green: 0.824, blue: 1, alpha: 1)
    static let dotOutlineColor = CGColor(srgbRed: 0, green: 0, blue: 0, alpha: 0.6)
    static let captionColor = CGColor(srgbRed: 0, green: 0, blue: 0, alpha: 1)

    /// The caption's size in the destination's own units — `OverlayGeometry`'s default measure,
    /// and therefore the metric the live canvas sizes its pill with too.
    static func measure(_ caption: String, _ points: CGFloat) -> CGSize {
        let text = typeset(caption, points)
        return CGSize(width: text.width, height: text.ascent + text.descent)
    }

    static func draw(_ result: TPDResult, options: OverlayOptions, in context: CGContext,
                     size: CGSize, style: OverlayStyle? = nil) {
        let style = style ?? .fitting(width: size.width)
        draw(OverlayGeometry(result: result, options: options, size: size, style: style),
             in: context, style: style)
    }

    static func draw(_ geometry: OverlayGeometry, in context: CGContext, style: OverlayStyle) {
        context.saveGState()
        defer { context.restoreGState() }
        if let box = geometry.box {
            context.setStrokeColor(boxColor); context.setLineWidth(style.lineWidth)
            context.addPath(CGPath(roundedRect: box, cornerWidth: style.boxCornerRadius,
                                   cornerHeight: style.boxCornerRadius, transform: nil))
            context.strokePath()
        }
        context.setLineWidth(max(style.lineWidth / 2.5, 1))
        for dot in geometry.dots {
            let rect = CGRect(x: dot.center.x - dot.radius, y: dot.center.y - dot.radius,
                              width: dot.radius * 2, height: dot.radius * 2)
            context.setFillColor(dotColor); context.fillEllipse(in: rect)
            context.setStrokeColor(dotOutlineColor); context.strokeEllipse(in: rect)
        }
        guard let pill = geometry.pill else { return }
        context.setFillColor(boxColor)
        context.addPath(CGPath(roundedRect: pill.rect, cornerWidth: pill.rect.height / 2,
                               cornerHeight: pill.rect.height / 2, transform: nil))
        context.fillPath()
        let text = typeset(pill.text, pill.fontSize)
        // The CTM's y points down here, so the glyphs need the opposite flip to come out upright.
        context.textMatrix = CGAffineTransform(scaleX: 1, y: -1)
        context.textPosition = CGPoint(x: pill.rect.midX - text.width / 2,
                                       y: pill.rect.midY + (text.ascent - text.descent) / 2)
        CTLineDraw(text.line, context)
    }

    private static func typeset(_ caption: String, _ points: CGFloat)
        -> (line: CTLine, width: CGFloat, ascent: CGFloat, descent: CGFloat) {
        let line = CTLineCreateWithAttributedString(NSAttributedString(string: caption, attributes: [
            .font: UIFont.systemFont(ofSize: points, weight: .semibold),
            .foregroundColor: UIColor(cgColor: captionColor)]))
        var ascent: CGFloat = 0, descent: CGFloat = 0
        let width = CGFloat(CTLineGetTypographicBounds(line, &ascent, &descent, nil))
        return (line, width, ascent, descent)
    }
}
