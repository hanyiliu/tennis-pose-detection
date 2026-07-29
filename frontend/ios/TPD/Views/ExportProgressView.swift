//  ExportProgressView.swift
//  Share: burn the overlay the user is looking at into the clip, then put the result in the
//  camera roll. Inference dominates — 7-10 s per analysed frame here — so this screen exists to
//  make that wait honest and to let the user out of it.

import Foundation
import Observation
import SwiftUI

/// What this presentation has already analysed, as a value an export can carry off the main
/// actor. `ResultCache` is `@MainActor` and `VideoExporter`'s `analyze` closure is neither async
/// nor isolated, so it is snapshotted once at Share: a frame analysed *during* an export is
/// missed, which beats hopping actors inside the exporter's loop.
struct ResultSnapshot: Sendable {
    var frameRate: Double = 30
    var entries: [Int: TPDResult] = [:]

    /// Keyed exactly as the preview keyed them — `ResultCache.index` is the one place that
    /// arithmetic lives, and a second copy here would silently miss every entry.
    func result(at seconds: Double) -> TPDResult? {
        entries[ResultCache.index(for: seconds, frameRate: frameRate)]
    }
}

/// How many frames the export got for free. `@unchecked Sendable` over a lock: the analyze
/// closure runs on the export's task and the screen reads the counts on main.
final class ExportTally: @unchecked Sendable {
    private let lock = NSLock()
    private var counts = (reused: 0, inferred: 0)
    var value: (reused: Int, inferred: Int) { lock.withLock { counts } }
    func record(reused hit: Bool) {
        lock.withLock { if hit { counts.reused += 1 } else { counts.inferred += 1 } }
    }
}

/// Owns the export task, the progress it publishes, and the save that follows it.
@MainActor
@Observable
final class ExportViewModel {
    /// `cancelling` is a state of its own because a cancel lands only at the next frame boundary
    /// and one frame is ~7-10 s: a card that dismissed instantly would be claiming a stop that
    /// has not happened yet.
    enum Phase: Equatable { case exporting, cancelling, cancelled, saving, saved, failed(String) }

    private(set) var phase: Phase = .exporting
    private(set) var fraction: Double = 0
    private(set) var startedAt = Date()
    private(set) var tally = ExportTally()
    /// The finished file, **held until the save actually succeeds**, which is what makes the
    /// denial message honest: granting access in Settings and tapping again costs one
    /// `performChanges`, not the export the user just waited minutes for.
    private(set) var exported: URL?
    @ObservationIgnored private var task: Task<Void, Never>?
    @ObservationIgnored private let saver: PhotoLibrarySaver

    init(saver: PhotoLibrarySaver = .live) { self.saver = saver }

    func start(source: URL, overlay: OverlayOptions, snapshot: ResultSnapshot) {
        guard task == nil, phase == .exporting, exported == nil else { return }
        startedAt = Date()
        tally = ExportTally()
        task = Task { [weak self] in await self?.run(source, overlay, snapshot) }
    }

    /// Cooperative, and the only exit that leaves nothing behind: `VideoExporter` checks
    /// cancellation each frame and tears reader, writer and half-written file down before the
    /// error escapes. Also the teardown `onDisappear` calls, so leaving mid-export stops the
    /// work and bins an export the user never saved.
    func stop() {
        task?.cancel()
        if phase == .exporting { phase = .cancelling }
        if let url = exported { try? FileManager.default.removeItem(at: url); exported = nil }
    }

    /// Retries the save alone, which is seconds against the minutes a re-export would be.
    func retrySave() {
        guard task == nil, let url = exported else { return }
        task = Task { [weak self] in await self?.store(url) }
    }

    /// Add, then delete the temp copy — and only then, so a failure anywhere leaves the file for
    /// `retrySave`. Internal, not private, so a test can drive the save half alone.
    func store(_ url: URL) async {
        exported = url
        phase = .saving
        do {
            try await saver.save(url)
            try? FileManager.default.removeItem(at: url)
            exported = nil
            phase = .saved
        } catch {
            phase = .failed(Self.message(error))
        }
        task = nil
    }

    private func run(_ source: URL, _ overlay: OverlayOptions, _ snapshot: ResultSnapshot) async {
        let exporter = VideoExporter(configuration: .init(overlay: overlay))
        // Read *before* `export`: a stream vended after the run has finished never yields.
        let stream = exporter.progress
        let watcher = Task { [weak self] in for await v in stream { self?.fraction = v } }
        defer { watcher.cancel(); task = nil }
        let tally = self.tally
        do {
            // Permission first, inference second: the other way round a refusal lands after
            // minutes of burn-in that no retry can hand back. See `requestAccess`.
            try await saver.requestAccess()
            // The app's one engine, not a second pair of Core ML models: `lend` hands over the
            // instance `StillInferenceWorker` already loaded, behind the lock every caller
            // now shares.
            let engine = try await StillInferenceWorker.shared.lend()
            let url = try await exporter.export(from: source) { frame in
                guard let hit = snapshot.result(at: frame.time.seconds) else {
                    tally.record(reused: false)
                    return try engine.predict(frame)
                }
                tally.record(reused: true)
                return hit
            }
            fraction = 1
            NSLog("TPD export: %d frames analysed, %d reused from the preview cache, %.0f s",
                  tally.value.inferred, tally.value.reused, -startedAt.timeIntervalSinceNow)
            await store(url)
        } catch is CancellationError {
            phase = .cancelled
        } catch {
            phase = .failed(Self.message(error))
        }
    }

    private static func message(_ error: Error) -> String {
        (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
    }
}

/// The card over the preview. Nothing here is a spinner: this runs at roughly 100x real time,
/// so the user gets a determinate bar, a percentage and a remaining estimate.
struct ExportProgressView: View {
    let model: ExportViewModel
    let onClose: () -> Void

    var body: some View {
        let (title, detail) = wording
        // The timeline ticks once a second so the estimate stays live across a 9.7 s inference,
        // without the view model holding a timer whose only job is to invalidate a view.
        return ZStack {
            Color.black.opacity(0.82).ignoresSafeArea()
            TimelineView(.periodic(from: model.startedAt, by: 1)) { tick in
                VStack(spacing: 14) {
                    Text(title).font(.headline)
                    if model.phase == .exporting || model.phase == .cancelling {
                        ProgressView(value: model.fraction).tint(.yellow)
                        Text(timings(tick.date)).font(.system(size: 12).monospacedDigit())
                            .foregroundStyle(.white.opacity(0.7))
                    }
                    Text(detail).font(.footnote).multilineTextAlignment(.center)
                        .foregroundStyle(.white.opacity(0.75))
                    HStack(spacing: 10) { buttons }.padding(.top, 2)
                }
                .padding(24).frame(maxWidth: 340)
                .background(Color(white: 0.11),
                            in: RoundedRectangle(cornerRadius: 18, style: .continuous))
                .foregroundStyle(.white).padding(24)
            }
        }
    }

    @ViewBuilder
    private var buttons: some View {
        switch model.phase {
        case .exporting: button("Cancel", filled: false) { model.stop() }
        case .cancelling, .saving: EmptyView()
        case .failed:
            if model.exported != nil { button("Save again", filled: true) { model.retrySave() } }
            button("Close", filled: model.exported == nil, action: onClose)
        case .saved, .cancelled: button("Done", filled: true, action: onClose)
        }
    }

    private func button(_ title: String, filled: Bool, action: @escaping () -> Void) -> some View {
        Button(title, action: action)
            .font(.system(size: 15, weight: .semibold))
            .padding(.horizontal, 18).padding(.vertical, 9)
            .background(filled ? Color.yellow : Color.white.opacity(0.16), in: Capsule())
            .foregroundStyle(filled ? Color.black : Color.white).buttonStyle(.plain)
    }

    /// The cadence and the reuse are stated rather than hidden: they are where the minutes go,
    /// and the second is the whole reason scrubbing before sharing pays for itself.
    private var wording: (String, String) {
        let (reused, inferred) = model.tally.value
        switch model.phase {
        case .exporting: return ("Burning in the overlay…", "Every third frame is analysed and "
                                     + "held for the two after it."
                                     + (reused > 0 ? " \(reused) came free from the preview." : ""))
        case .cancelling: return ("Stopping…", "One frame's inference cannot be interrupted, so "
                                      + "this takes a few seconds. No file is left behind.")
        case .cancelled: return ("Export cancelled", "Nothing was written to your library.")
        case .saving: return ("Saving to your camera roll…", "Adding it as a new video.")
        case .saved: return ("Saved to your camera roll", "\(inferred) frames analysed, \(reused) "
                                 + "reused from the preview. The original is untouched.")
        case .failed(let why): return ("Could not finish", why)
        }
    }

    /// Percent and a remaining estimate extrapolated from the fraction done — crude, but
    /// measured from this run rather than guessed, and it is what the user needs in order to
    /// decide whether to wait. Withheld below 5%, where one sample estimates nothing.
    private func timings(_ now: Date) -> String {
        let elapsed = max(now.timeIntervalSince(model.startedAt), 0)
        guard model.fraction >= 0.05 else { return "\(Int(elapsed)) s elapsed · estimating…" }
        let left = Int(elapsed / model.fraction - elapsed)
        return String(format: "%d%% · about %d:%02d left", Int(model.fraction * 100),
                      left / 60, left % 60)
    }
}
