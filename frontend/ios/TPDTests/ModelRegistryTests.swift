import CoreGraphics
import Foundation
import XCTest
@testable import TPD

final class ModelRegistryTests: XCTestCase {
    private let bundle = Bundle.main

    func testMalformedEntriesAreRejectedWithASpecificReason() throws {
        let url = try XCTUnwrap(bundle.url(forResource: ModelRegistry.resource, withExtension: "json"))
        let good = try XCTUnwrap(String(data: try Data(contentsOf: url), encoding: .utf8))
        let cases = [
            ("unknown output convention", "\"type\": \"logits\"", "\"type\": \"sigmoid\""),
            ("unknown kind", "\"kind\": \"classifier\"", "\"kind\": \"detector\""),
            ("duplicate labels", "\"backhand\", \"ready_position\"", "\"backhand\", \"backhand\""),
            ("unbaked normalization", "BakedIn\": true", "BakedIn\": false"),
            ("future schema", "\"schemaVersion\": 1", "\"schemaVersion\": 99"),
        ]
        for (name, find, replace) in cases {
            let broken = good.replacingOccurrences(of: find, with: replace)
            XCTAssertNotEqual(broken, good, "\(name): the fixture no longer contains '\(find)'")
            XCTAssertThrowsError(try ModelRegistry.decode(Data(broken.utf8)), name) { error in
                let said = (error as? TPDInferenceError)?.errorDescription ?? "\(error)"
                XCTAssertTrue(said.contains("TPDModelRegistry.json") && said.count > 40, "\(name): \(said)")
            }
        }
        XCTAssertThrowsError(try ModelRegistry.decode(Data("{\"models\": []}".utf8)))
    }

    func testSoftmaxIsAppliedOnlyWhereTheEntrySaysLogits() throws {
        let registry = try ModelRegistry.load(from: bundle)
        let classifier = try registry.first(of: .classifier)
        XCTAssertEqual(classifier.output.type, .logits)
        let softmaxed = try classifier.result(from: [4, 1, 0, -2], frameSize: .zero).probabilities
        XCTAssertEqual(softmaxed.reduce(0, +), 1, accuracy: 1e-5)
        XCTAssertEqual(softmaxed[0], 0.9341, accuracy: 1e-3)  // e^4 / (e^4+e^1+e^0+e^-2)
        let pipeline = try registry.first(of: .pipeline)
        XCTAssertEqual(pipeline.output.type, .probabilities)
        let probabilities: [Float] = [0.7, 0.2, 0.06, 0.04]
        XCTAssertEqual(try pipeline.result(from: probabilities, frameSize: .zero).probabilities,
                       probabilities, "a probability vector must pass through untouched")
    }

    /// **The one that matters.** The pose-split ResNet puts forehand at index 0 where the others
    /// put backhand, so one vector decoded through both must come back under two names — a shared
    /// list would agree here and be wrong on screen. Then the honest "no geometry".
    func testTheSameVectorDecodesToDifferentNamesUnderDifferentLabelOrders() throws {
        let classifiers = try ModelRegistry.load(from: bundle).models.filter { $0.kind == .classifier }
        XCTAssertEqual(classifiers.count, 2)
        let (first, second) = (classifiers[0], classifiers[1])
        XCTAssertNotEqual(first.labels, second.labels, "the fixture no longer disagrees on order")
        let logits: [Float] = [9, 1, 0, -1]  // peaks at the index the two entries name apart
        let left = try first.result(from: logits, frameSize: CGSize(width: 8, height: 6))
        let right = try second.result(from: logits, frameSize: .zero)
        XCTAssertEqual([left.bestIndex, right.bestIndex], [0, 0])
        XCTAssertEqual(left.confidence, right.confidence, accuracy: 1e-6)
        XCTAssertNotEqual(left.label, right.label,
                          "both models named index 0 '\(left.label)' — the decode is using one "
                              + "shared label list, not each entry's own")
        XCTAssertEqual(Set([left.label, right.label]), ["backhand", "forehand"])
        XCTAssertNil(left.bbox); XCTAssertTrue(left.keypoints.isEmpty)
        let geometry = OverlayGeometry(result: left, options: OverlayOptions(),
                                       size: CGSize(width: 300, height: 200))
        XCTAssertTrue(geometry.box == nil && geometry.dots.isEmpty, "a classifier drew geometry")
        let pill = try XCTUnwrap(geometry.pill).rect  // centred, not at an absent rect's origin
        XCTAssertEqual(pill.midY, 100, accuracy: 0.5)
    }
}
