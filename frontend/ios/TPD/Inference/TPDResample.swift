import CoreGraphics
import Foundation

/// The two resamples the pipeline needs, as a **port of Pillow's resampler**
/// (`src/libImaging/Resample.c`) rather than a CoreImage approximation: `PIL.Image.resize`
/// stretches its filter support by the downscale factor, and that stretch — not the filter
/// shape — is what no CoreImage filter reproduces. Over 24 1920x1080 frames of court lines,
/// net mesh and text, pushed through the real `TPDBBox.mlpackage` and
/// `norm_bbox_to_xyxy_pixels`, `CILanczosScaleTransform` moved the stage-1 crop rect by a
/// median of 34 px (max 89) and matched PIL on 0/24 frames; this port is byte-identical on
/// all 24. See frontend/ios/README.md for the stage-2 residual.
enum TPDResample {
    /// Tightly packed interleaved BGRA8 — what `CIContext.render(toBitmap:)` writes and Core
    /// ML's image input takes. Resampling is per channel, so the extra alpha changes nothing.
    struct Bitmap { var pixels: [UInt8]; let width: Int; let height: Int }

    /// A Pillow `ImagingFilter`: `weight` is zero outside `support`.
    struct Filter { let support: Double; let weight: @Sendable (Double) -> Double }

    /// `Image.Resampling.BILINEAR` — a triangle. Stage 1 uses it because the checkpoint is
    /// trained through it (`bbox_train.py`) and the backend infers through it.
    static let bilinear = Filter(support: 1) { x in max(1 - abs(x), 0) }

    /// `Image.Resampling.LANCZOS`, which `letterbox_resize` uses for stage 2. Spelled as two
    /// `sinc` calls rather than the algebraically equal closed form so the last bit matches.
    static let lanczos = Filter(support: 3) { x in
        x >= -3 && x < 3 ? sinc(x) * sinc(x / 3) : 0
    }

    private static func sinc(_ x: Double) -> Double {
        x == 0 ? 1 : sin(.pi * x) / (.pi * x)
    }

    /// Pillow's `PRECISION_BITS`: taps are fixed point with 22 fraction bits, and the
    /// accumulator starts at half an LSB so the final shift rounds.
    private static let bits = 22

    /// `precompute_coeffs` + `normalize_coeffs_8bpc`, for a whole-image box.
    private static func taps(_ inSize: Int, _ outSize: Int, _ filter: Filter)
        -> (k: [Int32], first: [Int], count: [Int], stride: Int) {
        let scale = Double(inSize) / Double(outSize)
        let filterScale = max(scale, 1)
        let support = filter.support * filterScale
        let stride = Int(support.rounded(.up)) * 2 + 1
        var k = [Int32](repeating: 0, count: outSize * stride)
        var first = [Int](repeating: 0, count: outSize), count = first
        for out in 0..<outSize {
            let center = (Double(out) + 0.5) * scale
            // Both casts truncate toward zero in the C, as `Int(_: Double)` does here.
            let lo = max(Int(center - support + 0.5), 0)
            let n = max(min(Int(center + support + 0.5), inSize) - lo, 0)
            var weights = [Double](repeating: 0, count: n), total = 0.0
            for i in 0..<n {
                weights[i] = filter.weight((Double(lo + i) - center + 0.5) / filterScale)
                total += weights[i]
            }
            for i in 0..<n {
                // The +-400 clamp only stops `Int32(_:)` trapping where the C wraps; a
                // normalized tap never leaves +-1.3.
                let w = total == 0 ? 0 : min(max(weights[i] / total, -400), 400)
                k[out * stride + i] = Int32(w * Double(1 << bits) + (w < 0 ? -0.5 : 0.5))
            }
            (first[out], count[out]) = (lo, n)
        }
        return (k, first, count, stride)
    }

    /// `ImagingResampleHorizontal_8bpc`, writing transposed so that running it twice is
    /// Pillow's horizontal-then-vertical pair — including the 8-bit intermediate between
    /// them, which is load-bearing for byte equality.
    private static func pass(_ image: Bitmap, _ outWidth: Int, _ filter: Filter) -> Bitmap {
        let (k, first, count, stride) = taps(image.width, outWidth, filter)
        var dst = [UInt8](repeating: 0, count: outWidth * image.height * 4)
        image.pixels.withUnsafeBufferPointer { source in
        k.withUnsafeBufferPointer { kernel in
        dst.withUnsafeMutableBufferPointer { out in
            for y in 0..<image.height {
                for x in 0..<outWidth {
                    let base = (y * image.width + first[x]) * 4, kBase = x * stride
                    // The four channels share a tap, so one SIMD lane each. `&+`/`&*` are
                    // the C's wrapping int; a tap set peaks near 255 * 1.2 * 2^22, inside Int32.
                    var acc = SIMD4<Int32>(repeating: 1 << (bits - 1))
                    for i in 0..<count[x] {
                        let p = base + i * 4
                        let pixel = SIMD4<Int32>(
                            Int32(source[p]), Int32(source[p + 1]),
                            Int32(source[p + 2]), Int32(source[p + 3]))
                        acc &+= pixel &* SIMD4<Int32>(repeating: kernel[kBase + i])
                    }
                    let o = (x * image.height + y) * 4
                    for channel in 0..<4 { out[o + channel] = UInt8(clamping: acc[channel] >> bits) }
                }
            }
        } } }
        return Bitmap(pixels: dst, width: image.height, height: outWidth)
    }

    /// `PIL.Image.resize(size, filter)`. An unchanged axis still runs `pass`, whose taps
    /// then collapse to an exact copy.
    static func resized(_ image: Bitmap, to width: Int, _ height: Int, _ filter: Filter) -> Bitmap {
        pass(pass(image, width, filter), height, filter)
    }

    /// `letterbox_resize`: LANCZOS fit into `output` x `output`, centered on black, on
    /// `LetterboxMath`'s **unclamped** scale (README divergence 2) — so a sub-128-px crop is
    /// upscaled where `PIL.thumbnail` would paste it at native size.
    static func letterboxed(_ image: Bitmap, to output: Int) -> Bitmap {
        let params = LetterboxMath.letterboxParams(
            for: CGSize(width: image.width, height: image.height), output: output)
        let width = min(max(Int(params.scaledWidth.rounded()), 1), output)
        let height = min(max(Int(params.scaledHeight.rounded()), 1), output)
        let scaled = resized(image, to: width, height, lanczos)
        var pixels = [UInt8](repeating: 0, count: output * output * 4)
        for alpha in stride(from: 3, to: pixels.count, by: 4) { pixels[alpha] = 255 }
        let left = (output - width) / 2, top = (output - height) / 2
        for row in 0..<height {
            let to = ((top + row) * output + left) * 4
            pixels.replaceSubrange(to ..< to + width * 4,
                                   with: scaled.pixels[row * width * 4 ..< (row + 1) * width * 4])
        }
        return Bitmap(pixels: pixels, width: output, height: output)
    }
}
