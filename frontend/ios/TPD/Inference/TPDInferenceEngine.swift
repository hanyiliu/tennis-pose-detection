import CoreGraphics
import CoreImage
import CoreML
import CoreVideo
import Foundation

/// The on-device pipeline: full frame -> bbox -> cropped/letterboxed player -> class.
///
/// Stages 2 and 3 are one Core ML graph (`TPDPoseNet`) with the heatmap argmax decode and
/// the `/(size - 1)` normalization baked in, so the only work left in Swift is geometry.
///
/// Models are loaded **generically** through `MLModel(contentsOf:)` and read by output
/// name, not through Xcode's generated wrapper classes: Xcode is not installed on the
/// development host, so generated code could not be verified here. The names are fixed
/// export-side (`tools/export_coreml.py`) and a mismatch throws rather than miscompiles.
///
/// Not `Sendable` on purpose — it owns `MLModel`s, a `CIContext` and two reusable pixel
/// buffers that `predict` writes. Own one instance per isolation domain and serialize
/// calls to it (the live view drops frames while a call is in flight).
final class TPDInferenceEngine {
    let spec: TPDModelSpec
    let labels: [String]

    private let bboxModel: MLModel
    private let poseModel: MLModel
    private let context: CIContext
    private let bboxBuffer: CVPixelBuffer
    private let poseBuffer: CVPixelBuffer

    init(bundle: Bundle = .main, configuration: MLModelConfiguration = MLModelConfiguration()) throws {
        spec = try TPDModelSpec.load(from: bundle)
        labels = try TPDLabels.load(from: bundle, expecting: spec.numClasses)
        bboxModel = try Self.loadModel(spec.bboxModelResource, bundle, configuration)
        poseModel = try Self.loadModel(spec.poseModelResource, bundle, configuration)
        // Color management off. The Python pipeline reads sRGB bytes and divides by 255
        // with no profile conversion, and the exported models fold that 1/255 into the
        // graph; letting CoreImage convert colorspaces would silently shift every input.
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
        let frameWidth = Int(extent.width.rounded())
        let frameHeight = Int(extent.height.rounded())

        // Stage 1 — the FULL frame squashed to a square, aspect ratio deliberately lost.
        render(frame.tpdSquashed(to: spec.bboxInputSize), into: bboxBuffer, size: spec.bboxInputSize)
        let raw = try floats(from: try run(bboxModel, on: bboxBuffer), named: "bbox", count: 4)
        let bbox = LetterboxMath.normBBoxToPixels(
            x: Double(raw[0]), y: Double(raw[1]), width: Double(raw[2]), height: Double(raw[3]),
            frameWidth: frameWidth, frameHeight: frameHeight)

        // Stages 2+3 — crop the ORIGINAL frame (never the squashed one) and letterbox it.
        let crop = frame.tpdCropped(toFramePixels: bbox).tpdLetterboxed(to: spec.keypointInputSize)
        render(crop, into: poseBuffer, size: spec.keypointInputSize)
        let pose = try run(poseModel, on: poseBuffer)
        // No softmax here: PoseClassificationModel.forward() already applied one, and it is
        // exported inside the graph. The backend's second softmax is a documented
        // divergence (frontend/ios/README.md) — it flattens confidences without moving
        // the argmax.
        let probabilities = try floats(from: pose, named: "probs", count: spec.numClasses)
        let decoded = try floats(from: pose, named: "keypoints", count: spec.numKeypoints * 3)

        var keypoints: [TPDKeypoint] = []
        keypoints.reserveCapacity(spec.numKeypoints)
        for channel in 0..<spec.numKeypoints {
            let base = channel * 3
            // x and y arrive normalized by (size - 1); undo that, then walk back through
            // the letterbox padding and scale into full-frame pixels.
            let letterboxed = LetterboxMath.denormalizeKeypoint(
                x: CGFloat(decoded[base]), y: CGFloat(decoded[base + 1]),
                output: spec.keypointInputSize)
            let position = LetterboxMath.letterboxToFrame(
                point: letterboxed, bbox: bbox, output: spec.keypointInputSize)
            keypoints.append(TPDKeypoint(position: position, visibility: decoded[base + 2]))
        }

        var bestIndex = -1
        var best = -Float.infinity
        for (index, probability) in probabilities.enumerated() where probability > best {
            best = probability
            bestIndex = index
        }

        return TPDResult(
            frameSize: CGSize(width: frameWidth, height: frameHeight),
            bbox: bbox,
            keypoints: keypoints,
            probabilities: probabilities,
            labels: labels,
            bestIndex: bestIndex)
    }

    // MARK: - Core ML plumbing

    private func run(_ model: MLModel, on buffer: CVPixelBuffer) throws -> MLFeatureProvider {
        // "image" is the input name every export writes; see convert() in export_coreml.py.
        let input = try MLDictionaryFeatureProvider(
            dictionary: ["image": MLFeatureValue(pixelBuffer: buffer)])
        return try model.prediction(from: input)
    }

    private func floats(from provider: MLFeatureProvider, named name: String, count: Int) throws -> [Float] {
        guard let array = provider.featureValue(for: name)?.multiArrayValue else {
            throw TPDInferenceError.missingOutput(name)
        }
        guard array.dataType == .float32 else {
            throw TPDInferenceError.unexpectedOutput(name, reason: "dtype \(array.dataType.rawValue), expected float32")
        }
        guard array.count == count else {
            throw TPDInferenceError.unexpectedOutput(name, reason: "\(array.count) values, expected \(count)")
        }
        return array.withUnsafeBufferPointer(ofType: Float.self) { Array($0) }
    }

    private func render(_ image: CIImage, into buffer: CVPixelBuffer, size: Int) {
        let bounds = CGRect(x: 0, y: 0, width: CGFloat(size), height: CGFloat(size))
        // colorSpace nil: pass the sample values through untouched, as above.
        context.render(image, to: buffer, bounds: bounds, colorSpace: nil)
    }

    private static func loadModel(
        _ resource: String, _ bundle: Bundle, _ configuration: MLModelConfiguration
    ) throws -> MLModel {
        // Xcode compiles a bundled .mlpackage into <name>.mlmodelc at build time.
        guard let url = bundle.url(forResource: resource, withExtension: "mlmodelc") else {
            throw TPDInferenceError.resourceMissing("\(resource).mlmodelc")
        }
        do {
            return try MLModel(contentsOf: url, configuration: configuration)
        } catch {
            throw TPDInferenceError.modelLoadFailed(
                resource: resource, reason: String(describing: error))
        }
    }

    /// One buffer per stage, allocated once and rewritten every frame — a per-frame
    /// `CVPixelBufferCreate` is a measurable cost at 30 fps. 32BGRA is the format Core ML
    /// requires for an RGB image input.
    private static func makeBuffer(_ size: Int) throws -> CVPixelBuffer {
        var buffer: CVPixelBuffer?
        let attributes: [CFString: Any] = [
            kCVPixelBufferIOSurfacePropertiesKey: [CFString: Any]() as CFDictionary,
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA,
        ]
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, size, size, kCVPixelFormatType_32BGRA,
            attributes as CFDictionary, &buffer)
        guard status == kCVReturnSuccess, let buffer else {
            throw TPDInferenceError.pixelBufferAllocationFailed(code: status, size: size)
        }
        return buffer
    }
}
