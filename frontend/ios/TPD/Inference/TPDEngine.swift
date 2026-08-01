//  TPDEngine.swift
//  The one interface both kinds of model satisfy.

import CoreImage
import CoreML
import CoreVideo
import Foundation

/// One frame in, one `TPDResult` out, whichever model is loaded. A classifier says it localized
/// nothing by returning `bbox == nil` and no keypoints, never a zero rect — that draws, as a box
/// at the origin. Class, not struct: implementations own `MLModel`s and pixel buffers `predict`
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
            throw TPDInferenceError.invalidFrame("extent \(extent) has no finite pixel area")
        }
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
