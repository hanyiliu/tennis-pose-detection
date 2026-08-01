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
final class TPDInferenceEngine: TPDEngine {
    let entry: TPDModelEntry
    let spec: TPDModelSpec
    let labels: [String]

    private let bboxModel: MLModel, poseModel: MLModel
    private let context: CIContext
    private let bboxBuffer: CVPixelBuffer, poseBuffer: CVPixelBuffer

    init(entry: TPDModelEntry, bundle: Bundle = .main,
         configuration: MLModelConfiguration = MLModelConfiguration()) throws {
        self.entry = entry
        spec = try TPDModelSpec.load(from: bundle, resource: entry.specResource)
        labels = entry.labels
        guard labels.count == spec.numClasses else {
            throw ModelRegistry.malformed(
                "'\(entry.id)': \(labels.count) labels for a \(spec.numClasses)-class head")
        }
        bboxModel = try EngineIO.loadModel(spec.bboxModelResource, bundle, configuration)
        poseModel = try EngineIO.loadModel(spec.poseModelResource, bundle, configuration)
        // Colour management off: the Python pipeline reads sRGB bytes and divides by 255 with
        // no profile conversion, so a CoreImage colourspace convert would shift every input.
        context = CIContext(options: [.workingColorSpace: NSNull(), .cacheIntermediates: false])
        bboxBuffer = try EngineIO.makeBuffer(spec.bboxInputSize)
        poseBuffer = try EngineIO.makeBuffer(spec.keypointInputSize)
    }

    func predict(frame: CIImage) throws -> TPDResult {
        let pixels = try frameSize(of: frame)
        let frameWidth = Int(pixels.width), frameHeight = Int(pixels.height)

        // Stage 1 — the FULL frame squashed to a square, aspect ratio deliberately lost.
        let full = EngineIO.bitmap(of: frame, CGRect(origin: .zero, size: pixels), context)
        EngineIO.write(TPDResample.resized(full, to: spec.bboxInputSize, spec.bboxInputSize,
                                           TPDResample.bilinear), into: bboxBuffer)
        let raw = try EngineIO.floats(from: try run(bboxModel, on: bboxBuffer),
                                      named: "bbox", count: 4)
        let bbox = LetterboxMath.normBBoxToPixels(
            x: Double(raw[0]), y: Double(raw[1]), width: Double(raw[2]), height: Double(raw[3]),
            frameWidth: frameWidth, frameHeight: frameHeight)

        // Stages 2+3 — crop the ORIGINAL frame (never the squashed one) and letterbox it.
        EngineIO.write(TPDResample.letterboxed(EngineIO.bitmap(of: frame, bbox, context),
                                               to: spec.keypointInputSize), into: poseBuffer)
        let pose = try run(poseModel, on: poseBuffer)
        // No softmax here, now enforced by `entry.result`: this entry's output.type is
        // `probabilities`, because forward() already applied one inside the exported graph.
        let probabilities = try EngineIO.floats(from: pose, named: entry.output.name,
                                                count: spec.numClasses)
        let decoded = try EngineIO.floats(from: pose, named: "keypoints",
                                          count: spec.numKeypoints * 3)

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

        return try entry.result(from: probabilities, frameSize: pixels,
                                bbox: bbox, keypoints: keypoints)
    }

    private func run(_ model: MLModel, on buffer: CVPixelBuffer) throws -> MLFeatureProvider {
        try model.prediction(from: MLDictionaryFeatureProvider(
            dictionary: [entry.input.name: MLFeatureValue(pixelBuffer: buffer)]))
    }
}

/// The plumbing, now shared with `ClassifierEngine`; left where it was private to the class.
enum EngineIO {
    /// Rasterize `rect` — **top-left-origin pixels**, the space `normBBoxToPixels` returns —
    /// into a packed top-down BGRA bitmap. CoreImage's y axis points up, so the rect is flipped
    /// against the extent first; `colorSpace: nil` is the note in `init`.
    static func bitmap(of image: CIImage, _ rect: CGRect,
                       _ context: CIContext) -> TPDResample.Bitmap {
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
    static func write(_ bitmap: TPDResample.Bitmap, into buffer: CVPixelBuffer) {
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

    static func floats(from provider: MLFeatureProvider, named name: String, count: Int) throws -> [Float] {
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
    static func loadModel(
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
    static func makeBuffer(_ size: Int) throws -> CVPixelBuffer {
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

    static func softmax(_ raw: [Float]) -> [Float] {
        guard let peak = raw.max() else { return [] }
        let weights = raw.map { expf($0 - peak) }
        let total = weights.reduce(0, +)
        return total > 0 ? weights.map { $0 / total } : raw
    }
}
