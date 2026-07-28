import CoreGraphics
import Foundation

/// Pure geometry shared by the two model stages. Ports, which must stay bit-comparable:
///
/// - `normBBoxToPixels`  <- `preprocessing/pil_preprocessing.py::norm_bbox_to_xyxy_pixels`
/// - `letterboxToFrame`  <- `backend/api/utils/image_outputs.py::_letterbox_xy_to_bbox_xy`
/// - `letterboxParams`   <- the forward half of the same map
///
/// Imports only Foundation/CoreGraphics on purpose: that lets it compile and *run*
/// against the macOS SDK, which is the only way to verify it numerically while Xcode.app
/// is missing. Do not "simplify" the rounding or the clamps.
enum LetterboxMath {

    /// Forward letterbox transform: source pixels -> square model input.
    struct Params: Equatable, Sendable {
        /// `min(out / width, out / height)`, deliberately **unclamped** — a crop smaller
        /// than the model input is upscaled. `PIL.Image.thumbnail` never upscales, but
        /// the training targets and the reverse map both assume the unclamped scale, so
        /// this side follows them. See frontend/ios/README.md, "Decided divergences".
        let scale: CGFloat
        let padX: CGFloat
        let padY: CGFloat
        let scaledWidth: CGFloat
        let scaledHeight: CGFloat
    }

    /// Letterbox parameters for fitting `size` into an `output` x `output` square.
    static func letterboxParams(for size: CGSize, output: Int) -> Params {
        let width = max(size.width, 1)
        let height = max(size.height, 1)
        let out = CGFloat(output)
        let scale = min(out / width, out / height)
        let scaledWidth = width * scale
        let scaledHeight = height * scale
        return Params(
            scale: scale,
            padX: (out - scaledWidth) / 2,
            padY: (out - scaledHeight) / 2,
            scaledWidth: scaledWidth,
            scaledHeight: scaledHeight
        )
    }

    /// Reverse map: a point in letterboxed model space -> full-frame pixel coordinates.
    ///
    /// Mirror of `_letterbox_xy_to_bbox_xy`, including its clamp of the crop-local point
    /// to `0...(bbox side - 1)` before the crop origin is added back.
    static func letterboxToFrame(point: CGPoint, bbox: CGRect, output: Int) -> CGPoint {
        // Python takes `max(x2 - x1, 1)`, so a degenerate bbox still divides by 1.
        let bboxWidth = max(bbox.width, 1)
        let bboxHeight = max(bbox.height, 1)
        let params = letterboxParams(for: CGSize(width: bboxWidth, height: bboxHeight), output: output)
        // `max(scale, 1e-6)` guards the divide exactly as the Python does.
        let scale = max(params.scale, 1e-6)
        let x = clamp((point.x - params.padX) / scale, 0, bboxWidth - 1)
        let y = clamp((point.y - params.padY) / scale, 0, bboxHeight - 1)
        return CGPoint(x: bbox.minX + x, y: bbox.minY + y)
    }

    /// Undo the `/(out - 1)` normalization the fused stage 2+3 graph applies to the
    /// heatmap argmax, giving a point in letterboxed model pixels.
    ///
    /// The divisor is `out - 1`, not `out` — it comes from
    /// `preprocessing/tensor_preprocessing.py::normalize_keypoints_xy`.
    static func denormalizeKeypoint(x: CGFloat, y: CGFloat, output: Int) -> CGPoint {
        let span = CGFloat(max(output - 1, 1))
        return CGPoint(x: x * span, y: y * span)
    }

    /// Stage-1 output (normalized `x, y, w, h`) -> integer pixel crop rect.
    ///
    /// Faithful port of `norm_bbox_to_xyxy_pixels`: the `<= 1.5` "looks normalized"
    /// heuristic, `int(round(...))` on all four edges, and both clamp pairs. The
    /// rounding is load-bearing — it is why a 3e-4 difference in the normalized bbox can
    /// still move the crop by a whole pixel.
    static func normBBoxToPixels(
        x: Double, y: Double, width: Double, height: Double,
        frameWidth: Int, frameHeight: Int
    ) -> CGRect {
        let frameW = max(frameWidth, 1)
        let frameH = max(frameHeight, 1)
        var x = x, y = y, w = width, h = height

        let magnitude = max(max(abs(x), abs(y)), max(abs(w), abs(h)))
        if magnitude <= 1.5 {
            x *= Double(frameW)
            y *= Double(frameH)
            w *= Double(frameW)
            h *= Double(frameH)
        }

        let x1 = roundedClamped(x, 0, frameW - 1)
        let y1 = roundedClamped(y, 0, frameH - 1)
        let x2 = roundedClamped(x + w, x1 + 1, frameW)
        let y2 = roundedClamped(y + h, y1 + 1, frameH)
        return CGRect(x: x1, y: y1, width: x2 - x1, height: y2 - y1)
    }

    /// `max(lower, min(int(round(value)), upper))`.
    ///
    /// `.toNearestOrEven` is not a stylistic choice: Python's `round()` breaks ties to
    /// even, and `1.5 -> 2` / `2.5 -> 2` shows up on real bbox edges. Non-finite input
    /// collapses to `lower` rather than trapping in `Int(_:)`; the Python would raise, and
    /// on device a NaN bbox must not take the app down mid-capture.
    private static func roundedClamped(_ value: Double, _ lower: Int, _ upper: Int) -> Int {
        guard value.isFinite else { return lower }
        let rounded = value.rounded(.toNearestOrEven)
        if rounded <= Double(lower) { return lower }
        if rounded >= Double(upper) { return upper }
        return Int(rounded)
    }

    private static func clamp(_ value: CGFloat, _ lower: CGFloat, _ upper: CGFloat) -> CGFloat {
        guard value.isFinite else { return lower }
        return min(max(value, lower), max(lower, upper))
    }
}
