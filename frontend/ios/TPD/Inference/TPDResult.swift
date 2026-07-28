import CoreGraphics
import Foundation

/// One decoded keypoint, already mapped back to **full-frame pixel coordinates**
/// (top-left origin), so overlays never redo the letterbox math.
struct TPDKeypoint: Sendable, Equatable {
    let position: CGPoint
    /// The raw maximum heatmap logit for this channel — deliberately not a sigmoid and
    /// not bounded to 0...1. `heatmaps_to_keypoints` returns exactly this, and the
    /// backend overlay treats `<= 0` as "do not draw".
    let visibility: Float

    var isDrawable: Bool { visibility > 0 }
}

/// Everything one frame of inference produces. Value type, `Sendable`, so it can cross
/// isolation boundaries from the engine to a view model without copying pixel data.
struct TPDResult: Sendable, Equatable {
    /// Size of the frame the bbox and keypoints are expressed in.
    let frameSize: CGSize
    /// Stage-1 crop rect, in frame pixels, after the integer rounding and clamping.
    let bbox: CGRect
    /// One entry per keypoint channel, in checkpoint channel order.
    let keypoints: [TPDKeypoint]
    /// Class probabilities aligned with `labels`. Already softmaxed by the model.
    let probabilities: [Float]
    /// Class names from `TPDLabels.json`, in checkpoint order.
    let labels: [String]
    /// Index of the highest probability, or -1 if there were no classes.
    let bestIndex: Int

    var label: String { labels.indices.contains(bestIndex) ? labels[bestIndex] : "unknown" }
    var confidence: Float { probabilities.indices.contains(bestIndex) ? probabilities[bestIndex] : 0 }
    /// The subset an overlay should draw, matching the backend's `visibility <= 0` skip.
    var drawableKeypoints: [TPDKeypoint] { keypoints.filter(\.isDrawable) }
}

/// Contract written next to the `.mlpackage`s by `tools/export_coreml.py`. Nothing here
/// may be hardcoded in Swift: the checkpoint decides the input sizes, the keypoint count
/// and the class count, and a re-export that changes any of them must change behaviour
/// without a code edit.
struct TPDModelSpec: Decodable, Sendable, Equatable {
    let bboxInputSize: Int
    let keypointInputSize: Int
    let numKeypoints: Int
    let numClasses: Int
    /// `.mlpackage` file names. Xcode compiles a bundled package to `<name>.mlmodelc`, so
    /// the runtime resource name is the base name without the extension.
    let bboxModel: String
    let poseModel: String

    var bboxModelResource: String { (bboxModel as NSString).deletingPathExtension }
    var poseModelResource: String { (poseModel as NSString).deletingPathExtension }

    static func load(from bundle: Bundle) throws -> TPDModelSpec {
        let spec: TPDModelSpec = try decode(resource: "TPDModelSpec", from: bundle)
        guard spec.bboxInputSize > 0, spec.keypointInputSize > 0,
              spec.numKeypoints > 0, spec.numClasses > 0,
              !spec.bboxModelResource.isEmpty, !spec.poseModelResource.isEmpty
        else {
            throw TPDInferenceError.malformedResource(
                "TPDModelSpec.json",
                reason: "sizes and counts must be positive and model names non-empty, got "
                    + "bbox \(spec.bboxInputSize), keypoint \(spec.keypointInputSize), "
                    + "keypoints \(spec.numKeypoints), classes \(spec.numClasses)")
        }
        return spec
    }
}

/// `TPDLabels.json`: `{"labels": [...]}`, normalized lowercase names in checkpoint order.
struct TPDLabels: Decodable, Sendable, Equatable {
    let labels: [String]

    /// Loads the names and checks them against the classifier width from the spec — a
    /// mismatch means the two sidecars came from different exports.
    static func load(from bundle: Bundle, expecting numClasses: Int) throws -> [String] {
        let decoded: TPDLabels = try decode(resource: "TPDLabels", from: bundle)
        guard decoded.labels.count == numClasses else {
            throw TPDInferenceError.malformedResource(
                "TPDLabels.json",
                reason: "\(decoded.labels.count) labels for \(numClasses) classes")
        }
        return decoded.labels
    }
}

private func decode<T: Decodable>(resource: String, from bundle: Bundle) throws -> T {
    let name = resource + ".json"
    guard let url = bundle.url(forResource: resource, withExtension: "json") else {
        throw TPDInferenceError.resourceMissing(name)
    }
    do {
        return try JSONDecoder().decode(T.self, from: Data(contentsOf: url))
    } catch {
        throw TPDInferenceError.malformedResource(name, reason: String(describing: error))
    }
}

/// Every way the engine can fail, typed so callers can tell a bad build (missing or
/// mismatched resources) from a runtime problem.
enum TPDInferenceError: Error, Equatable, LocalizedError {
    case resourceMissing(String)
    case malformedResource(String, reason: String)
    case modelLoadFailed(resource: String, reason: String)
    case pixelBufferAllocationFailed(code: Int32, size: Int)
    case invalidFrame(String)
    case missingOutput(String)
    case unexpectedOutput(String, reason: String)

    var errorDescription: String? {
        switch self {
        case .resourceMissing(let name):
            return "\(name) is not in the app bundle. Run `make export` in frontend/ios, "
                + "then `make generate`."
        case .malformedResource(let name, let reason):
            return "\(name) is malformed: \(reason)"
        case .modelLoadFailed(let resource, let reason):
            return "could not load \(resource).mlmodelc: \(reason)"
        case .pixelBufferAllocationFailed(let code, let size):
            return "CVPixelBufferCreate failed with \(code) for a \(size)x\(size) buffer"
        case .invalidFrame(let reason):
            return "frame cannot be used for inference: \(reason)"
        case .missingOutput(let name):
            return "model produced no output named '\(name)'"
        case .unexpectedOutput(let name, let reason):
            return "output '\(name)' does not match the spec: \(reason)"
        }
    }
}
