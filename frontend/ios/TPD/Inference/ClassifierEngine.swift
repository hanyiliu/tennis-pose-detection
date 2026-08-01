//  ClassifierEngine.swift
//  One image -> four numbers. No detection, no keypoints, and no pretending otherwise.

import CoreGraphics
import CoreImage
import CoreML
import Foundation

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
            throw ModelRegistry.malformed("'\(entry.id)' is not a single-package classifier")
        }
        self.entry = entry
        model = try EngineIO.loadModel(resource, bundle, configuration)
        // Colour management off, as in the pipeline: the reference transform reads sRGB bytes.
        context = CIContext(options: [.workingColorSpace: NSNull(), .cacheIntermediates: false])
        buffer = try EngineIO.makeBuffer(entry.input.size)
    }

    func predict(frame: CIImage) throws -> TPDResult {
        let pixels = try frameSize(of: frame)
        let side = entry.input.size
        // The WHOLE frame squashed to a square, aspect ratio deliberately lost — torchvision's
        // `T.Resize((side, side))`, i.e. the PIL bilinear resample `TPDResample.bilinear` ports
        // byte-exactly and that `parity_check.py` measured these two packages against.
        let full = EngineIO.bitmap(of: frame, CGRect(origin: .zero, size: pixels), context)
        EngineIO.write(TPDResample.resized(full, to: side, side, TPDResample.bilinear), into: buffer)
        // Plain 0–255 RGB: normalization is the graph's first op, so doing it here does it
        // twice. `entry.result` softmaxes iff output.type says logits, names classes with this
        // entry's OWN labels, and reports neither a bbox nor keypoints.
        let output = try model.prediction(from: MLDictionaryFeatureProvider(
            dictionary: [entry.input.name: MLFeatureValue(pixelBuffer: buffer)]))
        return try entry.result(
            from: EngineIO.floats(from: output, named: entry.output.name, count: entry.labels.count),
            frameSize: pixels)
    }
}
