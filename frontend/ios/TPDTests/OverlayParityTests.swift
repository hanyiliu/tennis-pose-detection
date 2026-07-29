import CoreGraphics
import SwiftUI
import XCTest
@testable import TPD

/// The live overlay and the exported overlay, each rendered for real and then compared as numbers.
/// `OverlayCanvas` (SwiftUI `Canvas`, driven here through `ImageRenderer`, so its actual `body`
/// runs) and `OverlayRenderer` (`CGContext`, in the exporter's own flipped bitmap) share exactly
/// one thing: `OverlayGeometry`. Nothing here reads that value — the comparison is on where the
/// colours landed — so a rasterizer that starts doing placement arithmetic of its own fails here
/// instead of silently making screenshots disagree with exported video.
@MainActor
final class OverlayParityTests: XCTestCase {
    /// 320x480 against a 640x480 result: scale 0.5 plus a 120 pt letterbox band, so nothing is an
    /// identity transform and a dropped `FrameFit` on either side shows up immediately.
    private let size = CGSize(width: 320, height: 480)
    private let options = OverlayOptions()

    func testTheLiveCanvasAndTheExporterPutEveryElementInTheSamePlace() throws {
        let live = try raster(of: liveImage(result()))
        let exported = try raster(of: exportedImage(result()))
        for (name, colour) in [("yellow (box and pill)", Self.isYellow),
                               ("cyan (keypoints)", Self.isCyan)] {
            let a = Self.features(live, colour), b = Self.features(exported, colour)
            // Both paths really drew the thing, so an all-black raster cannot pass by matching.
            XCTAssertGreaterThan(a.count, 60, "the live pass barely drew any \(name)")
            XCTAssertGreaterThan(b.count, 60, "the export pass barely drew any \(name)")
            XCTAssertEqual(a.bounds.minX, b.bounds.minX, accuracy: 1.5, name)
            XCTAssertEqual(a.bounds.minY, b.bounds.minY, accuracy: 1.5, name)
            XCTAssertEqual(a.bounds.maxX, b.bounds.maxX, accuracy: 1.5, name)
            XCTAssertEqual(a.bounds.maxY, b.bounds.maxY, accuracy: 1.5, name)
            // The centroid is the sensitive one: it moves for a shift the union box absorbs.
            XCTAssertEqual(a.centre.x, b.centre.x, accuracy: 1.0, name)
            XCTAssertEqual(a.centre.y, b.centre.y, accuracy: 1.0, name)
            // Area is the loosest of the three on purpose: two antialiasers disagree along every
            // edge, and the keypoint dots are almost all edge. Placement is what is being asserted.
            XCTAssertEqual(Double(a.count), Double(b.count),
                           accuracy: Double(a.count) * 0.35, name)
        }
    }

    /// The above only means something if it can tell a real drift from an antialiasing
    /// difference. So: the export path re-rendered from a geometry nudged by a few points — the
    /// size of one stray constant in one rasterizer — has to move the live path's numbers by more
    /// than the 1.0 pt the parity assertion allows. Without this the tolerances above are unproven.
    func testTheComparisonNoticesADriftOfAFewPoints() throws {
        var drifted = OverlayGeometry(result: result(), options: options, size: size)
        var pill = try XCTUnwrap(drifted.pill)
        pill.rect = pill.rect.offsetBy(dx: 0, dy: 4)
        drifted.pill = pill
        drifted.box = try XCTUnwrap(drifted.box).offsetBy(dx: 6, dy: 0)
        let honest = Self.features(try raster(of: liveImage(result())), Self.isYellow)
        let nudged = Self.features(try raster(of: rasterize { OverlayRenderer.draw(
            drifted, in: $0, style: OverlayStyle()) }), Self.isYellow)
        XCTAssertGreaterThan(abs(honest.centre.x - nudged.centre.x), 1.0, "a 6 pt box shift hid")
        XCTAssertGreaterThan(abs(honest.centre.y - nudged.centre.y), 1.0, "a 4 pt pill shift hid")
    }

    // MARK: - The two rendering paths

    /// The live path through its real `body` — `ImageRenderer` runs the `Canvas` closure.
    private func liveImage(_ subject: TPDResult) throws -> CGImage {
        let renderer = ImageRenderer(content: OverlayCanvas(result: subject, options: options)
            .frame(width: size.width, height: size.height)
            .background(Color.black))
        renderer.scale = 1
        let image = try XCTUnwrap(renderer.cgImage)
        XCTAssertEqual(image.width, Int(size.width), "ImageRenderer resized the canvas")
        return image
    }

    /// The export path in the exporter's own drawing environment: a bitmap flipped to top-left
    /// origin exactly as `VideoExporter` flips a decoded frame before handing it over.
    private func exportedImage(_ subject: TPDResult) throws -> CGImage {
        try rasterize { OverlayRenderer.draw(subject, options: options, in: $0, size: size,
                                             style: OverlayStyle()) }
    }

    private func rasterize(_ body: (CGContext) -> Void) throws -> CGImage {
        let space = try XCTUnwrap(CGColorSpace(name: CGColorSpace.sRGB))
        let context = try XCTUnwrap(CGContext(
            data: nil, width: Int(size.width), height: Int(size.height), bitsPerComponent: 8,
            bytesPerRow: 0, space: space, bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                | CGBitmapInfo.byteOrder32Little.rawValue))
        context.setFillColor(gray: 0, alpha: 1)
        context.fill(CGRect(origin: .zero, size: size))
        context.translateBy(x: 0, y: size.height)
        context.scaleBy(x: 1, y: -1)
        body(context)
        return try XCTUnwrap(context.makeImage())
    }

    // MARK: - Reducing a raster to numbers

    private struct Raster { var bytes: [UInt8]; var width: Int; var height: Int }
    private struct Features { var bounds = CGRect.null; var centre = CGPoint.zero; var count = 0 }

    /// Both images through one sRGB BGRA buffer, so the two renderers' colours are comparable.
    private func raster(of image: CGImage) throws -> Raster {
        let width = Int(size.width), height = Int(size.height)
        let space = try XCTUnwrap(CGColorSpace(name: CGColorSpace.sRGB))
        var bytes = [UInt8](repeating: 0, count: width * height * 4)
        try bytes.withUnsafeMutableBytes { raw in
            let context = try XCTUnwrap(CGContext(
                data: raw.baseAddress, width: width, height: height, bitsPerComponent: 8,
                bytesPerRow: width * 4, space: space,
                bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                    | CGBitmapInfo.byteOrder32Little.rawValue))
            context.draw(image, in: CGRect(origin: .zero, size: self.size))
        }
        return Raster(bytes: bytes, width: width, height: height)
    }

    /// Bounding box, centroid and area of every pixel of one overlay colour.
    private static func features(_ raster: Raster, _ matches: (UInt8, UInt8, UInt8) -> Bool)
        -> Features {
        var out = Features(), sumX = 0.0, sumY = 0.0
        for y in 0..<raster.height {
            for x in 0..<raster.width {
                let at = (y * raster.width + x) * 4  // BGRA, little-endian
                guard matches(raster.bytes[at + 2], raster.bytes[at + 1], raster.bytes[at])
                else { continue }
                out.bounds = out.bounds.union(CGRect(x: x, y: y, width: 1, height: 1))
                sumX += Double(x); sumY += Double(y); out.count += 1
            }
        }
        if out.count > 0 {
            out.centre = CGPoint(x: sumX / Double(out.count), y: sumY / Double(out.count))
        }
        return out
    }

    /// `OverlayRenderer`'s palette, thresholded loosely enough to survive two antialiasers.
    private static let isYellow: (UInt8, UInt8, UInt8) -> Bool = { $0 > 200 && $1 > 150 && $2 < 100 }
    private static let isCyan: (UInt8, UInt8, UInt8) -> Bool = { $0 < 160 && $1 > 150 && $2 > 200 }

    private func result() -> TPDResult {
        TPDResult(frameSize: CGSize(width: 640, height: 480),
                  bbox: CGRect(x: 100, y: 60, width: 200, height: 300),
                  keypoints: [(CGPoint(x: 150, y: 100), 0.9), (CGPoint(x: 220, y: 260), 0.2),
                              (CGPoint(x: 190, y: 140), 0.7), (CGPoint(x: 260, y: 330), 0.5),
                              (CGPoint(x: 120, y: 300), 0.4), (CGPoint(x: 100, y: 60), -0.4)]
                      .map { TPDKeypoint(position: $0.0, visibility: $0.1) },
                  probabilities: [0.05, 0.1, 0.8, 0.05],
                  labels: ["forehand", "backhand", "serve", "ready"], bestIndex: 2)
    }
}
