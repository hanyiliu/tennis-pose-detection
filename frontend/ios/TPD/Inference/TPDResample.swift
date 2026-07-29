import CoreGraphics
import Foundation

/// The two resamples the pipeline needs, as a **port of Pillow's resampler**
/// (`src/libImaging/Resample.c` + `Reduce.c`) rather than a CoreImage approximation:
/// `PIL.Image.resize` stretches its filter support by the downscale factor, and that stretch —
/// not the filter shape — is what no CoreImage filter reproduces. Over 24 1920x1080 frames of
/// court lines, net mesh and text, pushed through the real `TPDBBox.mlpackage` and
/// `norm_bbox_to_xyxy_pixels`, `CILanczosScaleTransform` moved the stage-1 crop rect by a
/// median of 34 px (max 89) and matched PIL on 0/24 frames; this port is byte-identical on
/// all 24. Stage 2 is byte-identical to `letterbox_resize` at every crop size measured — see
/// frontend/ios/README.md.
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

    /// A `PIL.Image.resize` `box`: a half-open source span, in source pixels, that need not
    /// land on pixel boundaries. Whole-image resizes pass `0..<size`; after a `reduce`
    /// pre-pass the span is the old box divided by the integer factor, so it is fractional.
    typealias Span = (lo: Double, hi: Double)

    /// `precompute_coeffs` + `normalize_coeffs_8bpc` over the source span `box`.
    private static func taps(_ inSize: Int, _ box: Span, _ outSize: Int, _ filter: Filter)
        -> (k: [Int32], first: [Int], count: [Int], stride: Int) {
        // `_imaging.c` parses the box with the `f` format and `precompute_coeffs` takes
        // `float in0, in1`, so a fractional post-reduce span is single precision *before*
        // the double arithmetic starts. Skipping this narrowing moved 6 of 46 sweep sizes
        // off byte-identity by 1/255.
        let (lo, hi) = (Double(Float(box.lo)), Double(Float(box.hi)))
        let scale = (hi - lo) / Double(outSize)
        let filterScale = max(scale, 1)
        let support = filter.support * filterScale
        let stride = Int(support.rounded(.up)) * 2 + 1
        // The C multiplies by a precomputed `1.0 / filterscale` rather than dividing per
        // tap. Kept in that form to match; no sweep here has shown a size where dividing
        // instead changes a byte, so treat it as fidelity to the source, not a fix.
        let inverse = 1 / filterScale
        var k = [Int32](repeating: 0, count: outSize * stride)
        var first = [Int](repeating: 0, count: outSize), count = first
        for out in 0..<outSize {
            let center = lo + (Double(out) + 0.5) * scale
            // Both casts truncate toward zero in the C, as `Int(_: Double)` does here.
            let lo = max(Int(center - support + 0.5), 0)
            let n = max(min(Int(center + support + 0.5), inSize) - lo, 0)
            var weights = [Double](repeating: 0, count: n), total = 0.0
            for i in 0..<n {
                weights[i] = filter.weight((Double(lo + i) - center + 0.5) * inverse)
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
    /// them, which is load-bearing for byte equality. The C narrows the intermediate to the
    /// rows the vertical taps touch; keeping them all changes no output pixel.
    private static func pass(_ image: Bitmap, _ outWidth: Int, _ filter: Filter, _ box: Span)
        -> Bitmap {
        let (k, first, count, stride) = taps(image.width, box, outWidth, filter)
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

    /// `Image._get_safe_box`: the integer region a later `reduce` must keep so the resample
    /// still sees every source pixel its filter support reaches. Kept faithful rather than
    /// folded away, but be aware it cannot bite here: `resized` only ever passes the whole
    /// image, so the expansion clips straight back to `(0, 0, width, height)` — verified on
    /// all 1940817 crop sizes in `1..1399` squared with a side over 128 px, 0 exceptions. No
    /// sweep case can distinguish it; it exists so the `reduce` origin subtraction below stays
    /// right if a real box is ever plumbed through.
    private static func safeBox(_ image: Bitmap, _ width: Int, _ height: Int, _ filter: Filter,
                                _ x: Span, _ y: Span) -> (x0: Int, y0: Int, x1: Int, y1: Int) {
        let support = filter.support - 0.5
        let padX = support * (x.hi - x.lo) / Double(width)
        let padY = support * (y.hi - y.lo) / Double(height)
        return (max(Int(x.lo - padX), 0), max(Int(y.lo - padY), 0),
                min(Int((x.hi + padX).rounded(.up)), image.width),
                min(Int((y.hi + padY).rounded(.up)), image.height))
    }

    /// `Image.reduce` / `ImagingReduceNxN` + `ImagingReduceCorners`: an integer box average,
    /// rounded through the C's reciprocal trick. Edge boxes are clipped, and their smaller
    /// pixel count is exactly what `ImagingReduceCorners` divides by, so one loop covers both.
    private static func reduced(_ image: Bitmap, _ xScale: Int, _ yScale: Int,
                                _ box: (x0: Int, y0: Int, x1: Int, y1: Int)) -> Bitmap {
        let outWidth = (box.x1 - box.x0 + xScale - 1) / xScale
        let outHeight = (box.y1 - box.y0 + yScale - 1) / yScale
        var dst = [UInt8](repeating: 0, count: outWidth * outHeight * 4)
        for y in 0..<outHeight {
            let y0 = box.y0 + y * yScale, y1 = min(y0 + yScale, box.y1)
            for x in 0..<outWidth {
                let x0 = box.x0 + x * xScale, x1 = min(x0 + xScale, box.x1)
                let n = (x1 - x0) * (y1 - y0)
                // Mirrors `division_UINT32(n, 8)`. The C computes this in float; a brute
                // force over n in 1..200000 found float and double give the same truncated
                // multiplier for every n, so the narrowing is kept only to match the C
                // literally, not because any measurement requires it.
                let multiplier = UInt32(Float(1 << 30) * 4 / Float(256 * n))
                var acc = SIMD4<UInt32>(repeating: UInt32(n / 2))
                for yy in y0..<y1 {
                    for xx in x0..<x1 {
                        let p = (yy * image.width + xx) * 4
                        acc &+= SIMD4<UInt32>(UInt32(image.pixels[p]), UInt32(image.pixels[p + 1]),
                                              UInt32(image.pixels[p + 2]), UInt32(image.pixels[p + 3]))
                    }
                }
                let o = (y * outWidth + x) * 4
                for c in 0..<4 { dst[o + c] = UInt8(truncatingIfNeeded: (acc[c] &* multiplier) >> 24) }
            }
        }
        return Bitmap(pixels: dst, width: outWidth, height: outHeight)
    }

    /// `PIL.Image.resize(size, filter, reducing_gap:)`. An unchanged axis still runs `pass`,
    /// whose taps then collapse to an exact copy. `reducingGap` is `nil` for stage 1 —
    /// `torchvision.transforms.Resize` calls `img.resize` without it — and `2` for stage 2,
    /// `PIL.Image.thumbnail`'s default.
    static func resized(_ image: Bitmap, to width: Int, _ height: Int, _ filter: Filter,
                        reducingGap: Double? = nil) -> Bitmap {
        var image = image
        var x: Span = (0, Double(image.width)), y: Span = (0, Double(image.height))
        if let gap = reducingGap {
            // `int(...) or 1` in the Python: a zero factor means "no reduction on this axis".
            let fx = max(Int((x.hi - x.lo) / Double(width) / gap), 1)
            let fy = max(Int((y.hi - y.lo) / Double(height) / gap), 1)
            if fx > 1 || fy > 1 {
                let safe = safeBox(image, width, height, filter, x, y)
                image = reduced(image, fx, fy, safe)
                x = ((x.lo - Double(safe.x0)) / Double(fx), (x.hi - Double(safe.x0)) / Double(fx))
                y = ((y.lo - Double(safe.y0)) / Double(fy), (y.hi - Double(safe.y0)) / Double(fy))
            }
        }
        return pass(pass(image, width, filter, x), height, filter, y)
    }

    /// `PIL.Image.thumbnail`'s `preserve_aspect_ratio`, including `round_aspect`: the minor
    /// axis is whichever of `floor`/`ceil` lands closer to the source aspect, ties to `floor`,
    /// never below 1. Note the minor-axis key minimizes `|aspect - out/n|`, which is *not*
    /// `|out/aspect - n|`, so this is not "round with a tie rule": measured against `round()`
    /// over the 1940817 crop sizes in `1..1399` squared with a side over 128 px, they disagree
    /// on 9342 (0.481%) — 635 exact `.5` ties plus 8707 from the key. Each disagreement is 1 px
    /// on the minor axis, which shifts every pasted pixel. Returning the input size unchanged
    /// is thumbnail's early return — it **never upscales**, and that is the clamped scale.
    static func thumbnailSize(width: Int, height: Int, output: Int) -> (width: Int, height: Int) {
        guard output < width || output < height else { return (width, height) }
        let out = Double(output), aspect = Double(width) / Double(height)
        // Python compares `x / y >= aspect` with `x == y == output`, i.e. `width <= height`.
        if width <= height {
            return (roundAspect(out * aspect) { abs(aspect - Double($0) / out) }, output)
        }
        return (output, roundAspect(out / aspect) { $0 == 0 ? 0 : abs(aspect - out / Double($0)) })
    }

    private static func roundAspect(_ number: Double, _ key: (Int) -> Double) -> Int {
        let low = Int(number.rounded(.down)), high = Int(number.rounded(.up))
        return max(key(low) <= key(high) ? low : high, 1)
    }

    /// `letterbox_resize`: LANCZOS thumbnail into `output` x `output`, centered on black.
    /// The fit is `PIL.thumbnail`'s, so a crop that already fits is pasted at native size
    /// rather than upscaled — the **clamped** scale, matching both the deployed backend and
    /// what `keypoint_train.py` fed the model. See `LetterboxMath.Params.scale`.
    static func letterboxed(_ image: Bitmap, to output: Int) -> Bitmap {
        let size = thumbnailSize(width: image.width, height: image.height, output: output)
        let scaled = size.width == image.width && size.height == image.height
            ? image : resized(image, to: size.width, size.height, lanczos, reducingGap: 2)
        var pixels = [UInt8](repeating: 0, count: output * output * 4)
        for alpha in stride(from: 3, to: pixels.count, by: 4) { pixels[alpha] = 255 }
        let left = (output - size.width) / 2, top = (output - size.height) / 2
        for row in 0..<size.height {
            let to = ((top + row) * output + left) * 4
            pixels.replaceSubrange(to ..< to + size.width * 4,
                                   with: scaled.pixels[row * size.width * 4 ..< (row + 1) * size.width * 4])
        }
        return Bitmap(pixels: pixels, width: output, height: output)
    }
}
