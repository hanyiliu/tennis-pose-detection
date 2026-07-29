import XCTest
@testable import TPD

/// Byte-level and structural cover for the Pillow port. The exact bytes below are traced by
/// hand through `precompute_coeffs` + `ImagingResampleHorizontal_8bpc`, because "close
/// enough" resampling is exactly what moved the stage-1 crop by up to 89 px.
final class TPDResampleTests: XCTestCase {

    // MARK: - Byte-level

    /// 2x2 -> 1x1 bilinear. Both taps normalize to 0.5, i.e. `0.5 * 2^22 + 0.5` truncated =
    /// 2097152, and the accumulator starts at `1 << 21`. Row 0: (1+10+20)*2097152 >> 22 = 15
    /// (15.5 truncated). Row 1: (1+30+40)*2097152 >> 22 = 35. Vertical: (1+15+35)*2097152
    /// >> 22 = 25. The true mean of 10/20/30/40 is also 25, so the passes agree here.
    func testTwoByTwoBilinearDownscaleIsByteExact() {
        let out = TPDResample.resized(gray(2, 2, [10, 20, 30, 40]), to: 1, 1, TPDResample.bilinear)
        XCTAssertEqual(out.width, 1)
        XCTAssertEqual(out.height, 1)
        XCTAssertEqual(out.pixels, [25, 25, 25, 255])
    }

    /// The same shape with 0/0/0/255 lands on 64, not on the exact mean 63.75 and not on 63.
    /// That gap is the 8-bit intermediate between the horizontal and vertical passes: row 0
    /// truncates to 0, row 1 to 128, and (1+0+128)*2097152 >> 22 = 64. Collapsing the two
    /// passes into one, or keeping the intermediate at higher precision, breaks this.
    func testEightBitIntermediateBetweenPassesIsLoadBearing() {
        let out = TPDResample.resized(gray(2, 2, [0, 0, 0, 255]), to: 1, 1, TPDResample.bilinear)
        XCTAssertEqual(out.pixels, [64, 64, 64, 255])
    }

    // MARK: - Structural invariants

    func testResizedReturnsTheRequestedDimensions() {
        for (width, height) in [(128, 128), (256, 256), (37, 91), (1, 5)] {
            let out = TPDResample.resized(constant(200, 150, 12), to: width, height,
                                          TPDResample.bilinear)
            XCTAssertEqual(out.width, width)
            XCTAssertEqual(out.height, height)
            XCTAssertEqual(out.pixels.count, width * height * 4)
        }
    }

    /// Normalized taps sum to one, so a flat image survives a downscale unchanged. Both code
    /// paths: 200 -> 128 skips the `reduce` pre-pass (factor 0), 512 -> 128 takes it (factor
    /// 2). A bug in the reciprocal division or the tap normalization shows up as +-1 drift.
    func testDownscaleOfAConstantImageStaysConstant() {
        for source in [200, 512] {
            let out = TPDResample.resized(constant(source, source, 137), to: 128, 128,
                                          TPDResample.lanczos, reducingGap: 2)
            for (x, y) in [(0, 0), (63, 64), (127, 127), (0, 127), (127, 0)] {
                XCTAssertEqual(color(out, x, y), [137, 137, 137, 255],
                               "drifted at \(x),\(y) from \(source)")
            }
        }
    }

    // MARK: - Never upscale

    /// `PIL.thumbnail`'s early return, and the reason `LetterboxMath.Params.scale` is
    /// clamped: a crop that already fits keeps its own size on both axes.
    func testThumbnailNeverUpscales() {
        XCTAssertEqual(TPDResample.thumbnailSize(width: 64, height: 48, output: 128).width, 64)
        XCTAssertEqual(TPDResample.thumbnailSize(width: 64, height: 48, output: 128).height, 48)
        for width in stride(from: 1, through: 128, by: 3) {
            for height in stride(from: 1, through: 128, by: 5) {
                let size = TPDResample.thumbnailSize(width: width, height: height, output: 128)
                XCTAssertEqual(size.width, width, "upscaled width of \(width)x\(height)")
                XCTAssertEqual(size.height, height, "upscaled height of \(width)x\(height)")
            }
        }
    }

    /// Above the output the fit is aspect-preserving: 256x128 -> 128x64, 128x256 -> 64x128,
    /// and neither axis ever exceeds the output or the source.
    func testThumbnailFitsInsideTheOutputWithoutExceedingTheSource() {
        XCTAssertEqual(TPDResample.thumbnailSize(width: 256, height: 128, output: 128).height, 64)
        XCTAssertEqual(TPDResample.thumbnailSize(width: 128, height: 256, output: 128).width, 64)
        for width in stride(from: 1, through: 600, by: 17) {
            for height in stride(from: 1, through: 600, by: 23) {
                let size = TPDResample.thumbnailSize(width: width, height: height, output: 128)
                XCTAssertLessThanOrEqual(size.width, min(width, 128), "\(width)x\(height)")
                XCTAssertLessThanOrEqual(size.height, min(height, 128), "\(width)x\(height)")
                XCTAssertGreaterThanOrEqual(size.width, 1)
                XCTAssertGreaterThanOrEqual(size.height, 1)
            }
        }
    }

    // MARK: - letterboxed

    /// A 64x48 crop is pasted at native size into a 128x128 canvas: content at x 32...95 and
    /// y 40...87, opaque black outside. Upscaled, the canvas would be full-bleed instead.
    func testLetterboxCentresASubOutputCropOnBlackWithoutUpscaling() {
        let out = TPDResample.letterboxed(constant(64, 48, 200), to: 128)
        XCTAssertEqual(out.width, 128)
        XCTAssertEqual(out.height, 128)
        XCTAssertEqual(color(out, 32, 40), [200, 200, 200, 255])
        XCTAssertEqual(color(out, 95, 87), [200, 200, 200, 255])
        for (x, y) in [(0, 0), (127, 127), (31, 40), (96, 40), (32, 39), (32, 88)] {
            XCTAssertEqual(color(out, x, y), [0, 0, 0, 255], "expected padding at \(x),\(y)")
        }
    }

    /// The padding lands on the minor axis and nowhere else: 256x128 fits as 128x64 with 32
    /// black rows above and below, 128x256 as 64x128 with 32 black columns either side.
    func testPaddingGoesOnTheMinorAxisOnly() {
        let wide = TPDResample.letterboxed(constant(256, 128, 200), to: 128)
        let tall = TPDResample.letterboxed(constant(128, 256, 200), to: 128)
        for other in [0, 63, 127] {
            for edge in [0, 31, 127] {
                XCTAssertEqual(color(wide, other, edge), [0, 0, 0, 255], "wide \(other),\(edge)")
                XCTAssertEqual(color(tall, edge, other), [0, 0, 0, 255], "tall \(edge),\(other)")
            }
            for inside in [32, 95] {
                XCTAssertEqual(color(wide, other, inside), [200, 200, 200, 255])
                XCTAssertEqual(color(tall, inside, other), [200, 200, 200, 255])
            }
        }
    }

    // MARK: - Helpers

    private func gray(_ width: Int, _ height: Int, _ values: [UInt8]) -> TPDResample.Bitmap {
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        for index in 0..<(width * height) {
            pixels[index * 4] = values[index]
            pixels[index * 4 + 1] = values[index]
            pixels[index * 4 + 2] = values[index]
            pixels[index * 4 + 3] = 255
        }
        return TPDResample.Bitmap(pixels: pixels, width: width, height: height)
    }

    private func constant(_ width: Int, _ height: Int, _ value: UInt8) -> TPDResample.Bitmap {
        gray(width, height, [UInt8](repeating: value, count: width * height))
    }

    private func color(_ bitmap: TPDResample.Bitmap, _ x: Int, _ y: Int) -> [UInt8] {
        let base = (y * bitmap.width + x) * 4
        return Array(bitmap.pixels[base..<base + 4])
    }
}
