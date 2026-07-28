import CoreGraphics
import Foundation

/// Pure geometry shared by the two stages. Ports that must stay bit-comparable:
/// `normBBoxToPixels` <- `pil_preprocessing.norm_bbox_to_xyxy_pixels`, `letterboxToFrame`
/// <- `image_outputs._letterbox_xy_to_bbox_xy`, `letterboxParams` <- its forward half.
/// Foundation/CoreGraphics only, so it compiles and *runs* against the macOS SDK — the only
/// way to verify it while Xcode.app is missing. Do not "simplify" the rounding or clamps.
enum LetterboxMath {
    /// Forward transform: source pixels -> square model input.
    struct Params: Equatable, Sendable {
        /// `min(1, min(out / width, out / height))` — **clamped**, because `PIL.thumbnail`
        /// never upscales and that is the convention the model was actually fed:
        /// `train/keypoint_train.py:220` letterboxes the training *input* through
        /// `letterbox_resize`. Only the target heatmaps (`tensor_preprocessing.py:120`) use
        /// an unclamped scale, so training is self-inconsistent below 128 px — but the input
        /// convention is unambiguous and the backend shares it. See
        /// frontend/ios/README.md, "Decided divergences".
        let scale: CGFloat
        let padX: CGFloat, padY: CGFloat
        let scaledWidth: CGFloat, scaledHeight: CGFloat
    }

    /// Letterbox parameters for fitting `size` into an `output` x `output` square.
    static func letterboxParams(for size: CGSize, output: Int) -> Params {
        let width = max(size.width, 1), height = max(size.height, 1)
        let out = CGFloat(output)
        let scale = min(1, min(out / width, out / height))
        let scaledWidth = width * scale, scaledHeight = height * scale
        return Params(scale: scale, padX: (out - scaledWidth) / 2, padY: (out - scaledHeight) / 2,
                      scaledWidth: scaledWidth, scaledHeight: scaledHeight)
    }

    /// Reverse map: letterboxed model space -> full-frame pixels. Mirrors
    /// `_letterbox_xy_to_bbox_xy`, including its clamp of the crop-local point to
    /// `0...(bbox side - 1)` before the crop origin is added back — but **not** its scale.
    /// The backend inverts with the unclamped `min(out/w, out/h)`, which does not undo the
    /// forward transform its own `letterbox_resize` applied, so it mis-places keypoints on
    /// sub-128 crops. Sharing `letterboxParams` here keeps the inverse a true inverse. This
    /// path only draws overlays — it can never change the predicted class — so being correct
    /// is worth more than being bug-compatible.
    static func letterboxToFrame(point: CGPoint, bbox: CGRect, output: Int) -> CGPoint {
        // Python takes `max(x2 - x1, 1)`, so a degenerate bbox still divides by 1, and
        // `max(scale, 1e-6)` guards the divide.
        let bboxWidth = max(bbox.width, 1), bboxHeight = max(bbox.height, 1)
        let params = letterboxParams(for: CGSize(width: bboxWidth, height: bboxHeight), output: output)
        let scale = max(params.scale, 1e-6)
        let x = clamp((point.x - params.padX) / scale, 0, bboxWidth - 1)
        let y = clamp((point.y - params.padY) / scale, 0, bboxHeight - 1)
        return CGPoint(x: bbox.minX + x, y: bbox.minY + y)
    }

    /// Undo the `/(out - 1)` the fused stage 2+3 graph applies to the heatmap argmax,
    /// giving a point in letterboxed model pixels. The divisor is `out - 1`, not `out`;
    /// it comes from `tensor_preprocessing.normalize_keypoints_xy`.
    static func denormalizeKeypoint(x: CGFloat, y: CGFloat, output: Int) -> CGPoint {
        let span = CGFloat(max(output - 1, 1))
        return CGPoint(x: x * span, y: y * span)
    }

    /// Stage-1 output (normalized `x, y, w, h`) -> integer pixel crop rect. Faithful port
    /// of `norm_bbox_to_xyxy_pixels`: the `<= 1.5` "looks normalized" heuristic,
    /// `int(round(...))` on all four edges, and both clamp pairs. The rounding is
    /// load-bearing — a 3e-4 shift in the normalized bbox can move the crop a whole pixel.
    static func normBBoxToPixels(
        x: Double, y: Double, width: Double, height: Double, frameWidth: Int, frameHeight: Int
    ) -> CGRect {
        let frameW = max(frameWidth, 1), frameH = max(frameHeight, 1)
        var x = x, y = y, w = width, h = height
        if max(max(abs(x), abs(y)), max(abs(w), abs(h))) <= 1.5 {
            x *= Double(frameW); y *= Double(frameH)
            w *= Double(frameW); h *= Double(frameH)
        }
        let x1 = roundedClamped(x, 0, frameW - 1)
        let y1 = roundedClamped(y, 0, frameH - 1)
        let x2 = roundedClamped(x + w, x1 + 1, frameW)
        let y2 = roundedClamped(y + h, y1 + 1, frameH)
        return CGRect(x: x1, y: y1, width: x2 - x1, height: y2 - y1)
    }

    /// `max(lower, min(int(round(value)), upper))`. `.toNearestOrEven` is not a stylistic
    /// choice: Python's `round()` breaks ties to even, and `1.5 -> 2` / `2.5 -> 2` shows up
    /// on real bbox edges. Non-finite input collapses to `lower` rather than trapping in
    /// `Int(_:)` — the Python raises, but on device a NaN bbox must not kill the capture.
    private static func roundedClamped(_ value: Double, _ lower: Int, _ upper: Int) -> Int {
        guard value.isFinite else { return lower }
        let rounded = value.rounded(.toNearestOrEven)
        if rounded <= Double(lower) { return lower }
        return rounded >= Double(upper) ? upper : Int(rounded)
    }

    private static func clamp(_ value: CGFloat, _ lower: CGFloat, _ upper: CGFloat) -> CGFloat {
        guard value.isFinite else { return lower }
        return min(max(value, lower), max(lower, upper))
    }
}
