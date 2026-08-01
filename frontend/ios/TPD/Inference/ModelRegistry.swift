//  ModelRegistry.swift
//  TPDModelRegistry.json, decoded — the app's entire idea of what a model is.

import CoreGraphics
import Foundation

/// One shipped model, as `tools/export_comparison.py` writes it. Three fields can never be
/// assumed in Swift, because each is silent when wrong: `output.type` (a second softmax keeps the
/// argmax but flattens every confidence), `labels` (per model — the pose-split ResNet puts forehand
/// at 0 where the others put backhand), and `labelOrder.status`, which the picker owes the user.
struct TPDModelEntry: Decodable, Sendable, Equatable, Identifiable {
    enum Kind: String, Decodable, Sendable { case pipeline, classifier }
    enum OutputType: String, Decodable, Sendable { case logits, probabilities }
    enum OrderStatus: String, Decodable, Sendable { case derived, assumed }

    /// `normalizationBakedIn`: the mean/std divide is the graph's first op, so Swift hands over
    /// plain 0–255 RGB and the registry's own mean/std stay documentation, left undecoded here.
    struct Input: Decodable, Sendable, Equatable { let name: String, size: Int, normalizationBakedIn: Bool }
    struct Output: Decodable, Sendable, Equatable { let name: String; let type: OutputType }
    struct LabelOrder: Decodable, Sendable, Equatable { let status: OrderStatus, source: String }
    struct Build: Decodable, Sendable, Equatable { let pipelineSpec: String?, precision: String }
    struct Source: Decodable, Sendable, Equatable { let checkpoint: String, sha256: String }
    let id: String, displayName: String
    let kind: Kind, packages: [String]
    let input: Input, output: Output
    let labels: [String], labelOrder: LabelOrder
    let build: Build, source: Source
    /// `.mlpackage`s are git-ignored, so this is a screenshot's only way to say which export.
    var provenance: String { "\(source.checkpoint) · \(source.sha256.prefix(8)) · \(build.precision)" }
    var resources: [String] { packages.map { ($0 as NSString).deletingPathExtension } }
    var specResource: String { ((build.pipelineSpec ?? "") as NSString).deletingPathExtension }
    var producesGeometry: Bool { kind == .pipeline }
    var shortName: String { String(displayName.prefix { $0 != "(" }).trimmingCharacters(in: .whitespaces) }

    /// **The one place a raw output vector becomes a named prediction.** Both engines return
    /// through here, so neither the softmax decision nor the label lookup can drift from the
    /// registry, and a test drives it without Core ML. `bestIndex` is the FIRST peak — a tie
    /// keeps the earlier class — or -1 for an all-NaN vector, which reads as "unknown".
    func result(from raw: [Float], frameSize: CGSize,
                bbox: CGRect? = nil, keypoints: [TPDKeypoint] = []) throws -> TPDResult {
        guard raw.count == labels.count else { throw TPDInferenceError.unexpectedOutput(
            output.name, reason: "\(raw.count) values for \(labels.count) labels in '\(id)'") }
        // Softmax, shifted by the peak before exp so a large logit cannot overflow to inf.
        let peak = raw.max() ?? 0, weights = raw.map { expf($0 - peak) }, total = weights.reduce(0, +)
        let probabilities = output.type != .logits || total <= 0 ? raw : weights.map { $0 / total }
        let bestIndex = probabilities.firstIndex(of: probabilities.max() ?? .nan) ?? -1
        return TPDResult(frameSize: frameSize, bbox: bbox, keypoints: keypoints,
                         probabilities: probabilities, labels: labels, bestIndex: bestIndex)
    }

    func validate() throws {
        let problems = [
            input.size > 0 ? nil : "input.size \(input.size) is not positive",
            input.normalizationBakedIn ? nil
                : "normalization is not baked into the graph, and no engine here applies it",
            labels.count > 1 && Set(labels).count == labels.count && !labels.contains(where: \.isEmpty)
                ? nil : "labels \(labels) must be two or more distinct, non-empty names",
            !resources.isEmpty && !resources.contains(where: \.isEmpty)
                && (kind != .classifier || resources.count == 1)
                ? nil : "a \(kind.rawValue) cannot be built from packages \(packages)",
            kind == .pipeline && specResource.isEmpty ? "a pipeline needs build.pipelineSpec" : nil,
        ].compactMap { $0 }
        guard problems.isEmpty else { throw ModelRegistry.malformed(
            "'\(id)': " + problems.joined(separator: "; ")) }
    }
}

struct ModelRegistry: Decodable, Sendable, Equatable {
    static let resource = "TPDModelRegistry"
    static let supportedSchemaVersion = 1

    let schemaVersion: Int
    let models: [TPDModelEntry]
    var first: TPDModelEntry { models[0] }

    func first(of kind: TPDModelEntry.Kind) throws -> TPDModelEntry {
        guard let entry = models.first(where: { $0.kind == kind }) else { throw Self.malformed(
            "lists no \(kind.rawValue) model; have \(models.map(\.id))") }
        return entry
    }

    static func load(from bundle: Bundle = .main) throws -> ModelRegistry {
        guard let url = bundle.url(forResource: resource, withExtension: "json") else {
            throw TPDInferenceError.resourceMissing(resource + ".json") }
        return try decode(try Data(contentsOf: url))
    }

    /// Split from `load` so a test can hand it a deliberately broken document.
    static func decode(_ data: Data) throws -> ModelRegistry {
        let registry: ModelRegistry
        do { registry = try JSONDecoder().decode(ModelRegistry.self, from: data) }
        catch { throw malformed("\(error)") }
        guard registry.schemaVersion == supportedSchemaVersion else { throw malformed(
            "schemaVersion \(registry.schemaVersion), not \(supportedSchemaVersion) — "
                + "this build cannot read it") }
        guard !registry.models.isEmpty else { throw malformed("lists no models") }
        var seen = Set<String>()
        for entry in registry.models {
            try entry.validate()
            guard seen.insert(entry.id).inserted else { throw malformed("duplicate id '\(entry.id)'") }
        }
        return registry
    }

    static func malformed(_ reason: String) -> TPDInferenceError {
        .malformedResource(resource + ".json", reason: reason)
    }
}
