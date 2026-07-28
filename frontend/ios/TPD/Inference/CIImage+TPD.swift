import CoreGraphics
import CoreImage
import CoreImage.CIFilterBuiltins
import Foundation

/// The two resamples the pipeline needs, in CoreImage. Closest available equivalent to
/// PIL rather than a bit-for-bit port: `CILanczosScaleTransform` is *antialiased* (its
/// filter support grows with the reduction factor), the property that makes
/// `PIL.Image.resize` differ from naive sampling. A plain `transformed(by:)` would take
/// roughly one source pixel in seven going 1920 -> 256 and alias badly. Stage 2's Python
/// side already uses LANCZOS (`letterbox_resize`), so that call site matches outright.
extension CIImage {

    /// Squash the whole frame into `output` x `output`, **aspect ratio not preserved** —
    /// stage 1 is trained on a distorted full frame (`model_service.predict`).
    func tpdSquashed(to output: Int) -> CIImage {
        let source = tpdAtOrigin()
        let out = CGFloat(output)
        let scaleY = out / max(source.extent.height, 1)
        let scaleX = out / max(source.extent.width, 1)
        return source.tpdScaled(scaleY: scaleY, aspectRatio: scaleX / scaleY)
    }

    /// Crop to `rect`, given in **top-left-origin pixel coordinates** (the space
    /// `LetterboxMath.normBBoxToPixels` returns), and re-origin the result at (0, 0).
    /// CoreImage's y axis points up, so the rect is flipped against this image's extent
    /// first; padding is symmetric, so nothing downstream cares about row order.
    func tpdCropped(toFramePixels rect: CGRect) -> CIImage {
        let flipped = CGRect(
            x: extent.minX + rect.minX,
            y: extent.maxY - rect.maxY,
            width: max(rect.width, 1),
            height: max(rect.height, 1))
        return cropped(to: flipped)
            .transformed(by: CGAffineTransform(translationX: -flipped.minX, y: -flipped.minY))
    }

    /// Fit into an `output` x `output` square with centered black padding, using the
    /// **unclamped** scale from `LetterboxMath` (small crops are upscaled).
    func tpdLetterboxed(to output: Int) -> CIImage {
        let source = tpdAtOrigin()
        let params = LetterboxMath.letterboxParams(for: source.extent.size, output: output)
        let scaled = source.tpdScaled(scaleY: params.scale, aspectRatio: 1)
        let placed = scaled.transformed(by: CGAffineTransform(
            translationX: params.padX - scaled.extent.minX,
            y: params.padY - scaled.extent.minY))
        let canvas = CGRect(x: 0, y: 0, width: CGFloat(output), height: CGFloat(output))
        // Compositing over opaque black is what actually creates the padding: rendering
        // only writes the image's own extent, so the margins would otherwise keep
        // whatever was in the destination buffer from the previous frame.
        return placed.cropped(to: canvas).composited(over: CIImage(color: .black).cropped(to: canvas))
    }

    private func tpdAtOrigin() -> CIImage {
        extent.origin == .zero
            ? self
            : transformed(by: CGAffineTransform(translationX: -extent.minX, y: -extent.minY))
    }

    private func tpdScaled(scaleY: CGFloat, aspectRatio: CGFloat) -> CIImage {
        let filter = CIFilter.lanczosScaleTransform()
        filter.inputImage = self
        filter.scale = Float(scaleY)
        filter.aspectRatio = Float(aspectRatio)
        return filter.outputImage ?? self
    }
}
