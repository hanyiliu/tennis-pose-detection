import CoreGraphics
import XCTest
@testable import TPD

/// Locks the geometry several review rounds converged on: the `<= 1.5` "looks normalized"
/// heuristic, ties-to-even integer rounding, every clamp, and the **clamped** letterbox scale
/// that must never upscale a sub-output crop. Every expectation is hand-computed from
/// `norm_bbox_to_xyxy_pixels` / `_letterbox_xy_to_bbox_xy`, not read off the Swift.
final class LetterboxMathTests: XCTestCase {

    // MARK: - normBBoxToPixels

    /// Max |component| is 0.5, so the heuristic fires: 0.25*640=160, 0.5*480=240,
    /// 0.5*640=320, 0.25*480=120. A full-frame normalized box covers the frame exactly.
    func testNormalizedBoxIsScaledByFrameSize() {
        XCTAssertEqual(pixels(0.25, 0.5, 0.5, 0.25, 640, 480),
                       CGRect(x: 160, y: 240, width: 320, height: 120))
        XCTAssertEqual(pixels(0, 0, 1, 1, 640, 480), CGRect(x: 0, y: 0, width: 640, height: 480))
    }

    /// Max |component| is 300, so the box is already in pixels and passes through untouched.
    /// Scaling here would multiply a pixel box by the frame size.
    func testPixelSpaceBoxSkipsTheHeuristic() {
        XCTAssertEqual(pixels(100, 50, 200, 300, 640, 480),
                       CGRect(x: 100, y: 50, width: 200, height: 300))
    }

    /// The boundary is inclusive: w=1.5 scales (10,10,150,20, x2 clamping to 100), one ulp
    /// above it does not and leaves a 2x1 box at the origin.
    func testHeuristicBoundaryIsInclusiveAtOnePointFive() {
        XCTAssertEqual(pixels(0.1, 0.1, 1.5, 0.2, 100, 100),
                       CGRect(x: 10, y: 10, width: 90, height: 20))
        XCTAssertEqual(pixels(0.1, 0.1, 1.5000001, 0.2, 100, 100),
                       CGRect(x: 0, y: 0, width: 2, height: 1))
    }

    /// Python's `round()` breaks ties to even: 2.5 -> 2, 1.5 -> 2, and the far edges
    /// 12.5 -> 12 / 11.5 -> 12. Away-from-zero rounding would give x=3 and maxX=13.
    func testEdgesRoundTiesToEvenNotAwayFromZero() {
        XCTAssertEqual(pixels(2.5, 1.5, 10, 10, 100, 100),
                       CGRect(x: 2, y: 2, width: 10, height: 10))
    }

    /// A zero-area box still has to yield a croppable rect (x2 >= x1+1, y2 >= y1+1), and a
    /// box past the far edge only survives as a 1x1 at (W-1, H-1) — that is x1 clamped to
    /// W-1 and y1 to H-1, then x2/y2 clamped to W/H.
    func testDegenerateAndOutOfFrameBoxesClampToOnePixel() {
        XCTAssertEqual(pixels(10, 20, 0, 0, 640, 480), CGRect(x: 10, y: 20, width: 1, height: 1))
        XCTAssertEqual(pixels(1000, 900, 50, 50, 640, 480),
                       CGRect(x: 639, y: 479, width: 1, height: 1))
    }

    /// A negative origin clamps to 0 without translating the box, so the far edge stays put
    /// and the rect narrows — 100 wide in, 50 wide out.
    func testNegativeOriginClampsWithoutTranslating() {
        XCTAssertEqual(pixels(-50, -30, 100, 60, 640, 480),
                       CGRect(x: 0, y: 0, width: 50, height: 30))
    }

    /// The Python raises on a non-finite bbox; on device that would kill the capture session,
    /// so each edge collapses to its lower bound instead.
    func testNonFiniteBoxDegradesToOnePixelInsteadOfTrapping() {
        XCTAssertEqual(pixels(.nan, .nan, .nan, .nan, 640, 480),
                       CGRect(x: 0, y: 0, width: 1, height: 1))
        XCTAssertEqual(pixels(.infinity, -.infinity, .nan, 1, 640, 480),
                       CGRect(x: 0, y: 0, width: 1, height: 1))
    }

    /// Whatever stage 1 emits, the crop must be integral, at least 1x1 and inside the frame —
    /// that is what makes it safe to hand to `CIContext.render`.
    func testEveryOutputIsAnIntegralInFrameCrop() {
        let values: [Double] = [-9e9, -1.7, -0.5, 0, 0.3, 0.5, 1.0, 1.5, 1.5001, 37.5, 5000, .nan]
        for (width, height) in [(640, 480), (1920, 1080), (1, 1), (13, 7)] {
            for x in values {
                for w in values {
                    let rect = pixels(x, 0.3, w, 1.0, width, height)
                    let note = "\(x)/\(w) in \(width)x\(height) gave \(rect)"
                    XCTAssertEqual(rect.minX, rect.minX.rounded(), note)
                    XCTAssertEqual(rect.width, rect.width.rounded(), note)
                    XCTAssertTrue(rect.width >= 1 && rect.height >= 1, note)
                    XCTAssertTrue(rect.minX >= 0 && rect.minY >= 0, note)
                    XCTAssertTrue(rect.maxX <= CGFloat(width) && rect.maxY <= CGFloat(height), note)
                }
            }
        }
    }

    // MARK: - letterboxParams

    /// THE regression behind the sub-128 class flips: a crop that already fits is pasted at
    /// native size, never stretched. scale is exactly 1 and 64x48 centres at (32, 40).
    func testCropSmallerThanOutputIsNotUpscaled() {
        let params = LetterboxMath.letterboxParams(for: CGSize(width: 64, height: 48), output: 128)
        XCTAssertEqual(params.scale, 1)
        XCTAssertEqual(params.padX, 32)
        XCTAssertEqual(params.padY, 40)
        for width in 1...128 {
            for height in stride(from: 1, through: 128, by: 7) {
                XCTAssertEqual(
                    LetterboxMath.letterboxParams(for: CGSize(width: width, height: height),
                                                  output: 128).scale, 1,
                    "upscaled a \(width)x\(height) crop")
            }
        }
    }

    /// Above the output the unclamped fit applies: 256 -> 0.5 unpadded, 400x200 -> 0.32
    /// padded on y, 200x400 -> 0.32 padded on x.
    func testLargerCropsUseTheFittingScale() {
        let square = LetterboxMath.letterboxParams(for: CGSize(width: 256, height: 256), output: 128)
        XCTAssertEqual(square.scale, 0.5, accuracy: 1e-12)
        XCTAssertEqual(square.padX, 0, accuracy: 1e-12)
        let wide = LetterboxMath.letterboxParams(for: CGSize(width: 400, height: 200), output: 128)
        XCTAssertEqual(wide.scale, 0.32, accuracy: 1e-12)
        XCTAssertEqual(wide.padX, 0, accuracy: 1e-12)
        XCTAssertEqual(wide.padY, 32, accuracy: 1e-12)
        let tall = LetterboxMath.letterboxParams(for: CGSize(width: 200, height: 400), output: 128)
        XCTAssertEqual(tall.padX, 32, accuracy: 1e-12)
        XCTAssertEqual(tall.padY, 0, accuracy: 1e-12)
    }

    // MARK: - letterboxToFrame

    /// The point of sharing `letterboxParams` between the directions: forward then reverse
    /// lands back on the same pixel. Square, wide, tall, exactly-output and sub-output crops,
    /// at a grid of points and a non-zero crop origin.
    func testLetterboxToFrameInvertsTheForwardTransform() {
        for crop in [CGRect(x: 40, y: 90, width: 200, height: 200),
                     CGRect(x: 0, y: 0, width: 400, height: 150),
                     CGRect(x: 7, y: 3, width: 150, height: 400),
                     CGRect(x: 11, y: 13, width: 128, height: 128),
                     CGRect(x: 500, y: 250, width: 64, height: 48)] {
            let params = LetterboxMath.letterboxParams(for: crop.size, output: 128)
            for step in 0...8 {
                let fraction = CGFloat(step) / 8
                let local = CGPoint(x: (crop.width - 1) * fraction, y: (crop.height - 1) * fraction)
                let forward = CGPoint(x: local.x * params.scale + params.padX,
                                      y: local.y * params.scale + params.padY)
                let back = LetterboxMath.letterboxToFrame(point: forward, bbox: crop, output: 128)
                XCTAssertEqual(back.x, crop.minX + local.x, accuracy: 1e-6, "x on \(crop)")
                XCTAssertEqual(back.y, crop.minY + local.y, accuracy: 1e-6, "y on \(crop)")
            }
        }
    }

    /// Anything the decoder can emit (it normalizes to 0...1, and out-of-range values must
    /// clamp) has to land inside the crop, because the overlay draws it in frame pixels. A
    /// degenerate crop divides by `max(side, 1)` rather than by zero.
    func testMappedKeypointsStayInsideTheCrop() {
        let crop = CGRect(x: 30, y: 60, width: 90, height: 240)
        for step in -2...10 {
            let normalized = CGFloat(step) / 8
            let point = LetterboxMath.letterboxToFrame(
                point: LetterboxMath.denormalizeKeypoint(x: normalized, y: normalized, output: 128),
                bbox: crop, output: 128)
            XCTAssertTrue((crop.minX...(crop.maxX - 1)).contains(point.x), "x \(point.x)")
            XCTAssertTrue((crop.minY...(crop.maxY - 1)).contains(point.y), "y \(point.y)")
        }
        XCTAssertEqual(LetterboxMath.letterboxToFrame(point: CGPoint(x: 64, y: 64),
                                                      bbox: CGRect(x: 5, y: 6, width: 0, height: 0),
                                                      output: 128), CGPoint(x: 5, y: 6))
    }

    /// The divisor is `size - 1`, from `normalize_keypoints_xy` — half of a 128 px input is
    /// 63.5, not 64.
    func testDenormalizeUsesSizeMinusOne() {
        XCTAssertEqual(LetterboxMath.denormalizeKeypoint(x: 0.5, y: 1, output: 128),
                       CGPoint(x: 63.5, y: 127))
        XCTAssertEqual(LetterboxMath.denormalizeKeypoint(x: 1, y: 0, output: 1), CGPoint(x: 1, y: 0))
    }

    private func pixels(_ x: Double, _ y: Double, _ w: Double, _ h: Double,
                        _ width: Int, _ height: Int) -> CGRect {
        LetterboxMath.normBBoxToPixels(x: x, y: y, width: w, height: h,
                                       frameWidth: width, frameHeight: height)
    }
}
