//  DiagnosticsHUD.swift
//  The always-on rate readout over the live feed, and the panel it opens.

import CoreML
import SwiftUI

/// Top-left companion to `ToggleBar`: same SF Symbol vocabulary and the same
/// yellow "this is on" accent, but material-backed rather than opaque, so it
/// stays legible over a bright court without hiding any of it.
///
/// **Which fps.** Capture, display and inference run at wildly different rates
/// here — roughly 30, 30 and 0.1 in the simulator — so an unlabelled number
/// would be a guess. The one worth a permanent readout is inference: it is the
/// app's actual throughput and the only one the user can feel. It is spelled out
/// next to the number, and never abbreviated to a bare "fps".
struct DiagnosticsHUD: View {
    let stats: PerformanceMeter.Snapshot
    /// From the configuration the engine was really built with, so this reports
    /// what the app asked Core ML for rather than a re-derived default. `nil`
    /// until the models have loaded.
    let computeUnits: MLComputeUnits?

    /// Survives relaunch. A debugging affordance that resets every launch is one
    /// nobody uses.
    @AppStorage("diagnostics.advanced") private var advanced = false

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            readout
            if advanced { panel }
        }
        .foregroundStyle(.white)
    }

    private var readout: some View {
        Button { withAnimation(.snappy(duration: 0.2)) { advanced.toggle() } } label: {
            HStack(spacing: 5) {
                Image(systemName: "speedometer").foregroundStyle(advanced ? .yellow : .white)
                Text(Self.rate(stats.fps)).fontWeight(.semibold).monospacedDigit()
                Text("inference fps").foregroundStyle(.white.opacity(0.7))
                Image(systemName: advanced ? "chevron.up" : "chevron.down")
                    .imageScale(.small).foregroundStyle(.white.opacity(0.7))
            }
            .font(.caption)
            .padding(.horizontal, 10)
            .padding(.vertical, 7)
            .background(.ultraThinMaterial, in: Capsule())
            .overlay(Capsule().strokeBorder(.white.opacity(0.18)))
        }
        .buttonStyle(.plain)
        .accessibilityLabel("Inference rate")
        .accessibilityValue(stats.fps == nil ? "measuring" : "\(Self.rate(stats.fps)) per second")
        .accessibilityHint(advanced ? "Hides the diagnostics panel" : "Shows the diagnostics panel")
    }

    /// Everything someone debugging on-device asks for, in one column. Only real
    /// measurements and real build metadata — no derived score, no invented
    /// health indicator.
    private var panel: some View {
        VStack(alignment: .leading, spacing: 5) {
            row("inference rate", stats.fps.map { "\(Self.rate($0)) fps" } ?? "measuring…")
            row("recent min / max", stats.fps == nil ? "—"
                : "\(Self.rate(stats.minFPS)) / \(Self.rate(stats.maxFPS)) fps")
            row("stages 1–3", stats.latest.map { Self.seconds($0.model) } ?? "—")
            row("display raster", stats.latest.map { Self.seconds($0.raster) } ?? "—")
            row("total per frame", stats.latest.map { Self.seconds($0.total) } ?? "—")
            row("frames dropped", "\(stats.dropped)")
            row("passes completed", "\(stats.completed)")
            row("input frame", stats.latest.map { "\($0.width) × \($0.height)" } ?? "—")
            row("compute units", Self.units(computeUnits))
            if let model = ModelIdentity.bundled {
                row("stage 1 model", "\(model.bboxModel) @ \(model.bboxInputSize)²")
                row("stage 2+3 model", "\(model.poseModel) @ \(model.keypointInputSize)², "
                    + "\(model.numKeypoints) kp, \(model.numClasses) cls")
                row("weights", model.provenance)
            } else {
                row("model", "TPDModelSpec.json unreadable")
            }
        }
        .font(.caption2)
        .padding(10)
        .frame(maxWidth: 290, alignment: .leading)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous)
            .strokeBorder(.white.opacity(0.18)))
        .transition(.opacity.combined(with: .move(edge: .top)))
        .accessibilityElement(children: .combine)
    }

    private func row(_ name: String, _ value: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 10) {
            Text(name).foregroundStyle(.white.opacity(0.55))
            Spacer(minLength: 0)
            Text(value).monospacedDigit().multilineTextAlignment(.trailing)
        }
    }

    // MARK: - Formatting

    /// Sub-1 rates are the normal case in the simulator, where two decimals are
    /// the difference between "0.10" and "0.00"; above 10 the decimals are noise.
    static func rate(_ value: Double?) -> String {
        guard let value else { return "—" }
        return String(format: value < 1 ? "%.2f" : value < 10 ? "%.1f" : "%.0f", value)
    }

    /// Milliseconds on a device, seconds in the simulator — one formatter that is
    /// readable at both, because "9710 ms" is not.
    static func seconds(_ value: Double) -> String {
        value >= 1 ? String(format: "%.2f s", value) : String(format: "%.0f ms", value * 1000)
    }

    static func units(_ units: MLComputeUnits?) -> String {
        guard let units else { return "—" }
        switch units {
        case .cpuOnly: return "CPU only"
        case .cpuAndGPU: return "CPU + GPU"
        case .cpuAndNeuralEngine: return "CPU + Neural Engine"
        case .all: return "CPU + GPU + Neural Engine"
        @unknown default: return "unrecognized"
        }
    }
}

/// The build metadata half of `TPDModelSpec.json`.
///
/// A second decode of that file rather than a field added to `TPDModelSpec`, on
/// purpose: the spec is the *runtime contract*, it fails the whole engine when
/// anything in it is missing, and provenance strings must never gain that power.
/// Here a missing or malformed file only costs a row in a debug panel.
struct ModelIdentity: Decodable, Sendable, Equatable {
    let bboxModel: String, poseModel: String
    let bboxInputSize: Int, keypointInputSize: Int
    let numKeypoints: Int, numClasses: Int
    let precision: String?
    let sourceCheckpoint: String?
    let sourceCheckpointSha256: String?

    /// Checkpoint plus a short digest — enough to tell two exports apart in a
    /// screenshot, which is the whole reason a debug panel names the weights.
    var provenance: String {
        let name = (sourceCheckpoint as NSString?)?.lastPathComponent ?? "unknown checkpoint"
        let digest = sourceCheckpointSha256.map { " · \($0.prefix(8))" } ?? ""
        return name + digest + (precision.map { " · \($0)" } ?? "")
    }

    /// Read once, lazily, off the inference path — the panel is the only reader
    /// and it asks after the models are already loaded.
    static let bundled: ModelIdentity? = load(from: .main)

    static func load(from bundle: Bundle) -> ModelIdentity? {
        guard let url = bundle.url(forResource: "TPDModelSpec", withExtension: "json"),
              let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(ModelIdentity.self, from: data)
    }
}
