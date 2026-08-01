import CoreGraphics
import CoreML
import CoreVideo
import Foundation
import XCTest
@testable import TPD

/// The first time the whole pipeline runs on a simulator. `TPDTests` is hosted by the `TPD`
/// app — XcodeGen turns `dependencies: [TPD]` in project.yml into TEST_HOST/BUNDLE_LOADER —
/// so `Bundle.main` here is the **app** bundle and already carries TPDLabels.json,
/// TPDModelSpec.json and the compiled `.mlmodelc`s. Nothing is copied into the test bundle.
final class TPDInferenceEngineTests: XCTestCase {
    private let bundle = Bundle.main

    // MARK: - Sidecars

    /// The four numbers Swift is forbidden from hardcoding, read back from the JSON that
    /// `tools/export_coreml.py` writes. `hiddenDim` is not part of `TPDModelSpec`, so it is
    /// read from the raw object — a re-export that changes it must be visible here.
    func testModelSpecCarriesTheDocumentedShape() throws {
        let spec = try TPDModelSpec.load(from: bundle)
        XCTAssertEqual(spec.numClasses, 4)
        XCTAssertEqual(spec.numKeypoints, 18)
        XCTAssertEqual(spec.bboxInputSize, 256)
        XCTAssertEqual(spec.keypointInputSize, 128)
        XCTAssertEqual(spec.bboxModelResource, "TPDBBox")
        XCTAssertEqual(spec.poseModelResource, "TPDPoseNet")

        let url = try XCTUnwrap(bundle.url(forResource: "TPDModelSpec", withExtension: "json"))
        let raw = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        XCTAssertEqual(raw["hiddenDim"] as? Int, 384)
    }

    /// Labels are checked against the classifier width from the spec, so a mismatched pair of
    /// sidecars is a load failure rather than a silently wrong class name.
    func testLabelsParseAndAreCheckedAgainstTheClassifierWidth() throws {
        let spec = try TPDModelSpec.load(from: bundle)
        let labels = try TPDLabels.load(from: bundle, expecting: spec.numClasses)
        XCTAssertEqual(labels.count, 4)
        XCTAssertEqual(Set(labels).count, labels.count, "duplicate label in \(labels)")
        XCTAssertFalse(labels.contains(where: \.isEmpty))
        XCTAssertThrowsError(try TPDLabels.load(from: bundle, expecting: spec.numClasses + 1))
    }

    func testCompiledModelsAreInTheHostBundle() throws {
        for resource in try ModelRegistry.load(from: bundle).models.flatMap(\.resources) {
            XCTAssertNotNil(bundle.url(forResource: resource, withExtension: "mlmodelc"),
                            "\(resource).mlmodelc missing — run `make export` then `make generate`")
        }
    }

    /// `input.name`, `output.name` and `input.size` are only claims until a frame goes through
    /// the compiled package. Summing to 1 also pins the softmax `output.type: logits` asks for.
    func testEachClassifierRunsAFrameThroughItsOwnPackage() throws {
        let pixels = try frame(320, 240) { x, y in x > y ? 220 : 40 }
        for entry in try ModelRegistry.load(from: bundle).models where entry.kind == .classifier {
            let got = try ClassifierEngine(entry: entry, bundle: bundle).predict(pixelBuffer: pixels)
            let saw = "\(entry.id) said \(got.label) from \(got.probabilities)"
            XCTAssertEqual(got.probabilities.count, 4, saw)
            XCTAssertEqual(got.probabilities.reduce(0, +), 1, accuracy: 1e-4, saw)  // NaN fails too
            XCTAssertTrue(entry.labels.contains(got.label), saw + " — not one of \(entry.labels)")
        }
    }

    // MARK: - predict

    /// Runs both stages on a synthetic 640x480 frame with a bright figure-shaped bar on grey,
    /// and asserts every invariant the result contract promises.
    func testPredictProducesAWellFormedResult() throws {
        let engine = try pipelineEngine()
        let result = try engine.predict(pixelBuffer: try frame(640, 480) { x, y in
            (200...320).contains(x) && (80...400).contains(y) ? 235 : 60
        })
        let box = try XCTUnwrap(result.bbox, "a pipeline always localizes")

        XCTAssertEqual(result.frameSize, CGSize(width: 640, height: 480))
        XCTAssertEqual(result.probabilities.count, engine.spec.numClasses)
        XCTAssertEqual(result.keypoints.count, engine.spec.numKeypoints)
        XCTAssertEqual(result.labels, engine.labels)
        print("TPD probs \(result.probabilities) -> \(result.label) bbox \(box)")

        // Exactly ONE softmax. Summing to 1 is necessary but not sufficient — a second
        // softmax also sums to 1 — so the max is also bounded. Feeding a probability vector
        // back through softmax caps the largest entry at e / (e + n - 1), because the widest
        // possible input gap is 1. Anything above that could not have been softmaxed twice.
        let total = result.probabilities.reduce(0, +)
        XCTAssertEqual(total, 1, accuracy: 1e-3, "probabilities \(result.probabilities)")
        let peak = try XCTUnwrap(result.probabilities.max())
        let doubleSoftmaxCeiling = Float(M_E / (M_E + Double(engine.spec.numClasses - 1)))
        XCTAssertGreaterThan(peak, doubleSoftmaxCeiling,
                             "flat distribution \(result.probabilities) — a second softmax "
                                + "cannot push any class above \(doubleSoftmaxCeiling)")

        XCTAssertTrue(engine.labels.contains(result.label), "\(result.label) not in \(engine.labels)")
        XCTAssertEqual(result.confidence, peak)
        XCTAssertTrue(engine.spec.numClasses > result.bestIndex && result.bestIndex >= 0)

        // The crop is a real in-frame rect, and every keypoint is inside it — which also
        // makes each coordinate lie in 0...1 once normalized by the frame size.
        XCTAssertTrue(box.minX >= 0 && box.minY >= 0, "\(box)")
        XCTAssertTrue(box.maxX <= 640 && box.maxY <= 480, "\(box)")
        XCTAssertTrue(box.width >= 1 && box.height >= 1, "\(box)")
        for (index, keypoint) in result.keypoints.enumerated() {
            let point = keypoint.position
            XCTAssertTrue(point.x.isFinite && point.y.isFinite, "keypoint \(index) not finite")
            XCTAssertTrue(keypoint.visibility.isFinite, "visibility \(index) not finite")
            XCTAssertTrue((box.minX...box.maxX).contains(point.x),
                          "keypoint \(index) x \(point.x) outside \(box)")
            XCTAssertTrue((box.minY...box.maxY).contains(point.y),
                          "keypoint \(index) y \(point.y) outside \(box)")
            XCTAssertTrue((0...1).contains(point.x / 640) && (0...1).contains(point.y / 480))
        }
        // The overlay must skip non-positive visibility, the way the backend does.
        XCTAssertEqual(result.drawableKeypoints.count,
                       result.keypoints.filter { $0.visibility > 0 }.count)
    }

    /// An all-black frame is the input that used to overflow fp16 in stage 3. Nothing in the
    /// result may be NaN or infinite.
    func testBlackFrameProducesNoNaNOrInfinity() throws {
        let engine = try pipelineEngine()
        let result = try engine.predict(pixelBuffer: try frame(320, 240) { _, _ in 0 })
        let box = try XCTUnwrap(result.bbox)

        for (index, probability) in result.probabilities.enumerated() {
            XCTAssertTrue(probability.isFinite, "probability \(index) is \(probability)")
        }
        XCTAssertEqual(result.probabilities.reduce(0, +), 1, accuracy: 1e-3,
                       "black frame probabilities \(result.probabilities)")
        for value in [box.minX, box.minY, box.width, box.height] {
            XCTAssertTrue(value.isFinite, "bbox \(box) is not finite")
        }
        for (index, keypoint) in result.keypoints.enumerated() {
            XCTAssertTrue(keypoint.position.x.isFinite && keypoint.position.y.isFinite,
                          "black-frame keypoint \(index) is \(keypoint.position)")
            XCTAssertTrue(keypoint.visibility.isFinite,
                          "black-frame visibility \(index) is \(keypoint.visibility)")
        }
        XCTAssertTrue(engine.labels.contains(result.label))
    }

    private func pipelineEngine() throws -> TPDInferenceEngine {
        try TPDInferenceEngine(entry: try ModelRegistry.load(from: bundle).first(of: .pipeline))
    }

    /// Greyscale BGRA at an arbitrary size; `luma` is indexed in top-left-origin pixels.
    private func frame(_ width: Int, _ height: Int,
                       _ luma: (Int, Int) -> UInt8) throws -> CVPixelBuffer {
        var buffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA,
            [kCVPixelBufferIOSurfacePropertiesKey: [CFString: Any]() as CFDictionary] as CFDictionary,
            &buffer)
        let pixelBuffer = try XCTUnwrap(buffer, "CVPixelBufferCreate failed with \(status)")
        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }
        let base = try XCTUnwrap(CVPixelBufferGetBaseAddress(pixelBuffer))
            .assumingMemoryBound(to: UInt8.self)
        let rowBytes = CVPixelBufferGetBytesPerRow(pixelBuffer)
        for y in 0..<height {
            for x in 0..<width {
                let value = luma(x, y), offset = y * rowBytes + x * 4
                base[offset] = value; base[offset + 1] = value
                base[offset + 2] = value; base[offset + 3] = 255
            }
        }
        return pixelBuffer
    }
}
