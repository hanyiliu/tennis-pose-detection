//  TPDEngine.swift
//  The interface both kinds of model satisfy, the one dispatch, and the smaller of the two
//  engines — one image, four numbers. The pipeline keeps its own file.

import CoreImage
import CoreML
import CoreVideo
import Foundation

/// One frame in, one `TPDResult` out, whichever model is loaded. A classifier says it localized
/// nothing by returning `bbox == nil` and no keypoints, never a zero rect — that draws, as a box at
/// the origin. Class, not struct: implementations own `MLModel`s and pixel buffers `predict`
/// rewrites, so they are not `Sendable`; one per domain, which the worker's actor gives them.
protocol TPDEngine: AnyObject {
    var entry: TPDModelEntry { get }
    func predict(frame: CIImage) throws -> TPDResult
}

extension TPDEngine {
    func predict(pixelBuffer: CVPixelBuffer) throws -> TPDResult {
        try predict(frame: CIImage(cvPixelBuffer: pixelBuffer))
    }

    /// An infinite or empty extent reaches Core ML as a crash rather than as an error.
    func frameSize(of frame: CIImage) throws -> CGSize {
        let extent = frame.extent
        guard !extent.isInfinite, !extent.isEmpty, extent.width >= 1, extent.height >= 1 else {
            throw TPDInferenceError.invalidFrame("extent \(extent) has no finite pixel area") }
        return CGSize(width: extent.width.rounded(), height: extent.height.rounded())
    }
}

/// The only model dispatch in the app, and the reason no model id appears anywhere in Swift.
func makeEngine(_ entry: TPDModelEntry, bundle: Bundle = .main,
                configuration: MLModelConfiguration) throws -> any TPDEngine {
    switch entry.kind {
    case .pipeline: return try TPDInferenceEngine(entry: entry, bundle: bundle,
                                                  configuration: configuration)
    case .classifier: return try ClassifierEngine(entry: entry, bundle: bundle,
                                                  configuration: configuration)
    }
}

/// Driven entirely by its registry entry — input size, feature names, softmax-or-not and the
/// class names all come from `entry` — which is why nothing here knows which ResNet it holds.
final class ClassifierEngine: TPDEngine {
    let entry: TPDModelEntry
    private let model: MLModel
    private let context: CIContext
    private let buffer: CVPixelBuffer

    init(entry: TPDModelEntry, bundle: Bundle = .main,
         configuration: MLModelConfiguration = MLModelConfiguration()) throws {
        guard entry.kind == .classifier, let resource = entry.resources.first else {
            throw ModelRegistry.malformed("'\(entry.id)' is not a single-package classifier") }
        self.entry = entry
        model = try Self.loadModel(resource, bundle, configuration)
        // Colour management off, as in the pipeline: the reference transform reads sRGB bytes.
        context = CIContext(options: [.workingColorSpace: NSNull(), .cacheIntermediates: false])
        buffer = try Self.makeBuffer(entry.input.size)
    }

    func predict(frame: CIImage) throws -> TPDResult {
        let pixels = try frameSize(of: frame)
        let side = entry.input.size
        // The WHOLE frame squashed to a square, aspect ratio deliberately lost — torchvision's
        // `T.Resize((side, side))`, i.e. the PIL bilinear resample `TPDResample.bilinear` ports
        // byte-exactly and `parity_check.py` measured these two packages against. Plain 0–255 RGB
        // out of it: normalization is the graph's first op, so doing it here does it twice. And
        // `entry.result` softmaxes iff output.type says logits, with this entry's OWN labels.
        let full = bitmap(of: frame, CGRect(origin: .zero, size: pixels), context)
        write(TPDResample.resized(full, to: side, side, TPDResample.bilinear), into: buffer)
        let output = try run(model, on: buffer)
        return try entry.result(from: floats(from: output, named: entry.output.name,
                                            count: entry.labels.count), frameSize: pixels)
    }
}
