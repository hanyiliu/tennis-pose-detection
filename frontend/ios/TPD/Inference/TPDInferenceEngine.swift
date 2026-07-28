import CoreGraphics
import CoreImage
import CoreML
import CoreVideo
import Foundation

/// The on-device pipeline: full frame -> bbox -> cropped/letterboxed player -> class.
/// Stages 2 and 3 are one Core ML graph (`TPDPoseNet`) with the heatmap argmax decode and
/// the `/(size - 1)` normalization baked in, so the only work left in Swift is geometry.
/// CoreImage only rasterizes; every resample is `TPDResample`'s PIL port, never a CIFilter.
///
/// Models load **generically** through `MLModel(contentsOf:)` and are read by output name,
/// not through Xcode's generated wrappers: Xcode is not installed on the development host,
/// so generated code could not be verified here. The names are fixed export-side
/// (`tools/export_coreml.py`) and a mismatch throws rather than miscompiles.
///
/// Not `Sendable` on purpose — it owns `MLModel`s, a `CIContext` and two reusable pixel
/// buffers that `predict` writes. One instance per isolation domain, calls serialized.
final class TPDInferenceEngine {
    let spec: TPDModelSpec
    let labels: [String]

    private let bboxModel: MLModel, poseModel: MLModel
    private let context: CIContext
    private let bboxBuffer: CVPixelBuffer, poseBuffer: CVPixelBuffer

    init(bundle: Bundle = .main, configuration: MLModelConfiguration = MLModelConfiguration()) throws {
        spec = try TPDModelSpec.load(from: bundle)
        labels = try TPDLabels.load(from: bundle, expecting: spec.numClasses)
        bboxModel = try Self.loadModel(spec.bboxModelResource, bundle, configuration)
        poseModel = try Self.loadModel(spec.poseModelResource, bundle, configuration)
        // Colour management off: the Python pipeline reads sRGB bytes and divides by 255 with
        // no profile conversion, so a CoreImage colourspace convert would shift every input.
        context = CIContext(options: [.workingColorSpace: NSNull(), .cacheIntermediates: false])
        bboxBuffer = try Self.makeBuffer(spec.bboxInputSize)
        poseBuffer = try Self.makeBuffer(spec.keypointInputSize)
    }

    func predict(pixelBuffer: CVPixelBuffer) throws -> TPDResult {
        try predict(frame: CIImage(cvPixelBuffer: pixelBuffer))
    }

    func predict(frame: CIImage) throws -> TPDResult {
        let extent = frame.extent
        guard !extent.isInfinite, !extent.isEmpty, extent.width >= 1, extent.height >= 1 else {
            throw TPDInferenceError.invalidFrame("extent \(extent) has no finite pixel area")
        }
        let frameWidth = Int(extent.width.rounded()), frameHeight = Int(extent.height.rounded())

        // Stage 1 — the FULL frame squashed to a square, aspect ratio deliberately lost.
        let full = bitmap(of: frame, CGRect(x: 0, y: 0, width: frameWidth, height: frameHeight))
        write(TPDResample.resized(full, to: spec.bboxInputSize, spec.bboxInputSize,
                                  TPDResample.bilinear), into: bboxBuffer)
        let raw = try floats(from: try run(bboxModel, on: bboxBuffer), named: "bbox", count: 4)
        let bbox = LetterboxMath.normBBoxToPixels(
            x: Double(raw[0]), y: Double(raw[1]), width: Double(raw[2]), height: Double(raw[3]),
            frameWidth: frameWidth, frameHeight: frameHeight)

        // Stages 2+3 — crop the ORIGINAL frame (never the squashed one) and letterbox it.
        write(TPDResample.letterboxed(bitmap(of: frame, bbox), to: spec.keypointInputSize),
              into: poseBuffer)
        let pose = try run(poseModel, on: poseBuffer)
        // No softmax here: PoseClassificationModel.forward() already applied one and it is
        // inside the exported graph. The backend's second softmax is a documented divergence
        // (README) that flattens confidences without moving the argmax.
        let probabilities = try floats(from: pose, named: "probs", count: spec.numClasses)
        let decoded = try floats(from: pose, named: "keypoints", count: spec.numKeypoints * 3)

        let size = spec.keypointInputSize
        var keypoints: [TPDKeypoint] = []
        keypoints.reserveCapacity(spec.numKeypoints)
        for channel in 0..<spec.numKeypoints {
            let base = channel * 3
            // x, y arrive normalized by (size - 1): undo that, then walk back out through
            // the letterbox padding and scale.
            let letterboxed = LetterboxMath.denormalizeKeypoint(
                x: CGFloat(decoded[base]), y: CGFloat(decoded[base + 1]), output: size)
            let point = LetterboxMath.letterboxToFrame(point: letterboxed, bbox: bbox, output: size)
            keypoints.append(TPDKeypoint(position: point, visibility: decoded[base + 2]))
        }

        var bestIndex = -1, best = -Float.infinity
        for (index, probability) in probabilities.enumerated() where probability > best {
            (best, bestIndex) = (probability, index)
        }
        return TPDResult(
            frameSize: CGSize(width: frameWidth, height: frameHeight), bbox: bbox,
            keypoints: keypoints, probabilities: probabilities, labels: labels, bestIndex: bestIndex)
    }

    /// Rasterize `rect` — **top-left-origin pixels**, the space `normBBoxToPixels` returns —
    /// into a packed top-down BGRA bitmap. CoreImage's y axis points up, so the rect is flipped
    /// against the extent first; `colorSpace: nil` is the note in `init`.
    private func bitmap(of image: CIImage, _ rect: CGRect) -> TPDResample.Bitmap {
        let width = max(Int(rect.width), 1), height = max(Int(rect.height), 1)
        let bounds = CGRect(x: image.extent.minX + rect.minX, y: image.extent.maxY - rect.maxY,
                            width: CGFloat(width), height: CGFloat(height))
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        pixels.withUnsafeMutableBytes {
            context.render(image, toBitmap: $0.baseAddress!, rowBytes: width * 4,
                           bounds: bounds, format: .BGRA8, colorSpace: nil)
        }
        return TPDResample.Bitmap(pixels: pixels, width: width, height: height)
    }

    /// Row by row: `CVPixelBufferGetBytesPerRow` is aligned padding, not always `width * 4`.
    private func write(_ bitmap: TPDResample.Bitmap, into buffer: CVPixelBuffer) {
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return }
        let rowBytes = CVPixelBufferGetBytesPerRow(buffer), length = bitmap.width * 4
        bitmap.pixels.withUnsafeBytes { source in
            for row in 0..<bitmap.height {
                memcpy(base + row * rowBytes, source.baseAddress! + row * length, length)
            }
        }
    }

    private func run(_ model: MLModel, on buffer: CVPixelBuffer) throws -> MLFeatureProvider {
        // "image" is the input name every export writes; see convert() in export_coreml.py.
        try model.prediction(from: MLDictionaryFeatureProvider(
            dictionary: ["image": MLFeatureValue(pixelBuffer: buffer)]))
    }

    private func floats(from provider: MLFeatureProvider, named name: String, count: Int) throws -> [Float] {
        guard let array = provider.featureValue(for: name)?.multiArrayValue else {
            throw TPDInferenceError.missingOutput(name)
        }
        guard array.dataType == .float32, array.count == count else {
            throw TPDInferenceError.unexpectedOutput(
                name, reason: "\(array.count) x dtype \(array.dataType.rawValue), "
                    + "expected \(count) x float32")
        }
        return array.withUnsafeBufferPointer(ofType: Float.self) { Array($0) }
    }

    /// Xcode compiles a bundled `.mlpackage` into `<name>.mlmodelc` at build time.
    private static func loadModel(
        _ resource: String, _ bundle: Bundle, _ configuration: MLModelConfiguration
    ) throws -> MLModel {
        guard let url = bundle.url(forResource: resource, withExtension: "mlmodelc") else {
            throw TPDInferenceError.resourceMissing("\(resource).mlmodelc")
        }
        do { return try MLModel(contentsOf: url, configuration: configuration) } catch {
            throw TPDInferenceError.modelLoadFailed(resource: resource, reason: "\(error)")
        }
    }

    /// One buffer per stage, allocated once and rewritten every frame; 32BGRA is what Core
    /// ML requires for an RGB image input.
    private static func makeBuffer(_ size: Int) throws -> CVPixelBuffer {
        var buffer: CVPixelBuffer?
        let attributes: [CFString: Any] = [
            kCVPixelBufferIOSurfacePropertiesKey: [CFString: Any]() as CFDictionary,
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA,
        ]
        let status = CVPixelBufferCreate(kCFAllocatorDefault, size, size,
                                         kCVPixelFormatType_32BGRA, attributes as CFDictionary, &buffer)
        guard status == kCVReturnSuccess, let buffer else {
            throw TPDInferenceError.pixelBufferAllocationFailed(code: status, size: size)
        }
        return buffer
    }
}
