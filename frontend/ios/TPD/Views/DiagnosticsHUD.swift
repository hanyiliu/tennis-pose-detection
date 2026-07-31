//  DiagnosticsHUD.swift
//  The always-on rate readout over the live feed, and the panel it opens.

import CoreML
import SwiftUI

/// Top-left companion to `ToggleBar`: same symbols, same yellow accent, but
/// material-backed so it stays legible over a bright court.
///
/// **Which fps.** Capture, display and inference run at wildly different rates,
/// so an unlabelled number would be a guess. This one is inference, measured
/// start of pass to start of pass: the wall-clock rate overlays appear at, idle
/// included. Not `1 / cost per frame`, which is larger and is printed as a cost.
struct DiagnosticsHUD: View {
    let stats: PerformanceMeter.Snapshot
    /// From the configuration the engine was really built with: a *request*, since
    /// Core ML has no read-back of where it dispatched. `nil` until the models load.
    let computeUnits: MLComputeUnits?

    /// Survives relaunch — which is why the pill sits above the panel, outside its
    /// scroll view: at any text size, the off switch is the first thing drawn.
    @AppStorage("diagnostics.advanced") private var advanced = false

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            readout
            if advanced { panel }
        }
        .foregroundStyle(.white)
        // Unbounded, the pill alone runs the preview's width at AX sizes.
        .frame(maxWidth: 290, alignment: .leading)
    }

    private var readout: some View {
        Button { withAnimation(.snappy(duration: 0.2)) { advanced.toggle() } } label: {
            HStack(alignment: .firstTextBaseline, spacing: 5) {
                Image(systemName: "speedometer").foregroundStyle(advanced ? .yellow : .white)
                // One `Text`: a large size wraps the unit rather than truncating.
                Text(Self.rate(stats.fps)).fontWeight(.semibold).monospacedDigit()
                    + Text(" inference fps").foregroundStyle(.white.opacity(0.7))
                Image(systemName: advanced ? "chevron.up" : "chevron.down")
                    .imageScale(.small).foregroundStyle(.white.opacity(0.7))
            }
            .font(.caption)
            .padding(.horizontal, 10)
            .padding(.vertical, 7)
            .background(.ultraThinMaterial, in: Capsule())
            .overlay(Capsule().strokeBorder(.white.opacity(0.18)))
            // The drawn capsule is the tap target, gaps between glyphs included.
            .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .accessibilityLabel("Inference rate")
        .accessibilityValue(stats.fps == nil ? "measuring" : "\(Self.rate(stats.fps)) per second")
        .accessibilityHint(advanced ? "Hides the diagnostics panel" : "Shows the diagnostics panel")
    }

    /// Only real measurements and real build metadata, no invented health score.
    /// Scrolls once the rows stop fitting — at accessibility sizes, immediately —
    /// and the height cap the caller gives it keeps it clear of the controls.
    private var panel: some View {
        ViewThatFits(in: .vertical) {
            rows
            ScrollView(.vertical) { rows }
        }
        .padding(10)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous)
            .strokeBorder(.white.opacity(0.18)))
        .transition(.opacity.combined(with: .move(edge: .top)))
        .accessibilityElement(children: .combine)
    }

    private var rows: some View {
        VStack(alignment: .leading, spacing: 5) {
            row("inference rate", stats.fps.map { "\(Self.rate($0)) fps" } ?? "measuring…")
            // The literal division above it, checkable by hand. Periods, not passes:
            // N samples span N-1 gaps. Window rows; the counters below are session.
            row("periods ÷ elapsed", Self.window(stats))
            row("stages 1–3", stats.latest.map { Self.seconds($0.model) } ?? "—")
            row("display raster", stats.latest.map { Self.seconds($0.raster) } ?? "—")
            row("cost per frame", stats.latest.map { Self.seconds($0.cost) } ?? "—")
            row("cost range", Self.range(stats.cheapest, stats.dearest))
            row("frames dropped, session", "\(stats.dropped)")
            row("passes completed, session", "\(stats.completed)")
            row("input frame", stats.latest.map { "\($0.width) × \($0.height)" } ?? "—")
            // A request, never a measurement; below it, the one dispatch fact that
            // *is* knowable, and the answer to "why seconds a pass".
            row("compute units asked for", Self.units(computeUnits))
            #if targetEnvironment(simulator)
            row("hardware", "Simulator — no Neural Engine")
            #endif
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
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    /// Name beside value while both fit, above it when they do not: an `HStack`
    /// alone answers an accessibility size by truncating both halves to a "…".
    private func row(_ name: String, _ value: String) -> some View {
        let label = Text(name).foregroundStyle(.white.opacity(0.55))
        let reading = Text(value).monospacedDigit()
        return ViewThatFits(in: .horizontal) {
            HStack(alignment: .firstTextBaseline, spacing: 10) {
                label; Spacer(minLength: 8); reading
            }
            VStack(alignment: .leading, spacing: 1) { label; reading }
        }
    }

    // MARK: - Formatting

    /// Two decimals below 1, where the simulator lives; none above 10.
    static func rate(_ value: Double?) -> String {
        guard let value else { return "—" }
        return String(format: value < 1 ? "%.2f" : value < 10 ? "%.1f" : "%.0f", value)
    }

    /// Milliseconds on a device, seconds in the simulator; "9710 ms" is neither.
    static func seconds(_ value: Double) -> String {
        value >= 1 ? String(format: "%.2f s", value) : String(format: "%.0f ms", value * 1000)
    }

    static func range(_ low: Double?, _ high: Double?) -> String {
        guard let low, let high else { return "—" }
        return "\(seconds(low)) – \(seconds(high))"
    }

    static func window(_ stats: PerformanceMeter.Snapshot) -> String {
        guard let elapsed = stats.windowElapsed else { return "—" }
        return "\(stats.windowPeriods) ÷ \(seconds(elapsed))"
    }

    /// Spelled as the API spells it: "CPU + GPU + Neural Engine" was prose for a
    /// compile-time constant, and among timings it read as a measurement.
    static func units(_ units: MLComputeUnits?) -> String {
        guard let units else { return "—" }
        switch units {
        case .cpuOnly: return "cpuOnly"
        case .cpuAndGPU: return "cpuAndGPU"
        case .cpuAndNeuralEngine: return "cpuAndNeuralEngine"
        case .all: return "all"
        @unknown default: return "unrecognized"
        }
    }
}

/// The build metadata half of `TPDModelSpec.json`, decoded apart from the spec on
/// purpose: the spec is the runtime contract and fails the whole engine when a
/// field is missing, whereas a malformed file here costs one row in a panel.
struct ModelIdentity: Decodable, Sendable, Equatable {
    let bboxModel: String, poseModel: String
    let bboxInputSize: Int, keypointInputSize: Int
    let numKeypoints: Int, numClasses: Int
    let precision: String?
    let sourceCheckpoint: String?
    let sourceCheckpointSha256: String?

    /// Checkpoint plus a digest — enough to tell two exports apart in a shot.
    var provenance: String {
        let name = (sourceCheckpoint as NSString?)?.lastPathComponent ?? "unknown checkpoint"
        let digest = sourceCheckpointSha256.map { " · \($0.prefix(8))" } ?? ""
        return name + digest + (precision.map { " · \($0)" } ?? "")
    }

    /// Read once, lazily, off the inference path; the panel is the only reader.
    static let bundled: ModelIdentity? = load(from: .main)

    static func load(from bundle: Bundle) -> ModelIdentity? {
        guard let url = bundle.url(forResource: "TPDModelSpec", withExtension: "json"),
              let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(ModelIdentity.self, from: data)
    }
}
