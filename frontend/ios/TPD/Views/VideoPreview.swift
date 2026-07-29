//  VideoPreview.swift
//  An AVPlayer whose frames we pull ourselves, the overlay stack the live view
//  uses, and a badge saying whether the two describe the same frame.

import AVFoundation
import CoreImage
import QuartzCore
import SwiftUI

/// Playback, exact seeking and opportunistic analysis. `NSObject` only because
/// `CADisplayLink` needs an `@objc` target; no engine of its own, so every pick
/// reuses the models `StillInferenceWorker.shared` already loaded.
@MainActor
@Observable
final class VideoPreviewModel: NSObject {
    /// `.stale` carries the time the overlay *does* describe. A wrong-but-pretty
    /// overlay pinned silently over a moving picture is worse than none.
    enum Overlay: Equatable { case none, current, stale(Double) }

    private(set) var image: CGImage?
    private(set) var result: TPDResult?
    private(set) var overlay: Overlay = .none
    private(set) var isPlaying = false
    private(set) var time: Double = 0
    private(set) var duration: Double = 0
    private(set) var failure: String?

    @ObservationIgnored private var player: AVPlayer?
    @ObservationIgnored private var output: AVPlayerItemVideoOutput?
    @ObservationIgnored private var link: CADisplayLink?
    /// Built on first use: `@State` constructs this model on main.
    @ObservationIgnored private var display: CIContext?
    @ObservationIgnored private var cache: ResultCache?
    /// The buffer on screen and when it appeared; `sync()` re-targets from it.
    @ObservationIgnored private var shown: VideoFrame?
    @ObservationIgnored private var shownSince: CFTimeInterval = 0
    @ObservationIgnored private var analysis: Task<Void, Never>?
    @ObservationIgnored private var resultTime: Double?
    @ObservationIgnored private var source: URL?
    /// `scrubbing` keeps the bar following the finger; `stopped` keeps a cancelled
    /// pass from re-arming `sync()` after teardown.
    @ObservationIgnored private var scrubbing = false
    @ObservationIgnored private var stopped = false

    /// Idempotent — the view's `.task` may re-run without the screen going away.
    func start(url: URL) async {
        guard player == nil, !stopped else { return }
        source = url
        let asset = AVURLAsset(url: url)
        do {
            guard let track = try await asset.loadTracks(withMediaType: .video).first else {
                throw FrameSourceError.videoDecodeUnavailable("clip contains no video track")
            }
            // As in `VideoFileFrameSource`: a bare output hands back the *encoded*
            // buffer and ignores `preferredTransform`, so a portrait clip would
            // arrive sideways and take every overlay coordinate with it.
            let composition = try await AVVideoComposition.videoComposition(withPropertiesOf: asset)
            duration = max(try await asset.load(.duration).seconds, 0)
            cache = ResultCache(frameRate: Double(try await track.load(.nominalFrameRate)))
            let item = AVPlayerItem(asset: asset)
            item.videoComposition = composition
            let output = AVPlayerItemVideoOutput(pixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
            ])
            item.add(output); self.output = output
            let player = AVPlayer(playerItem: item)
            player.isMuted = true; player.actionAtItemEnd = .pause
            self.player = player
            // Deliberately **not** `play()`: one pass costs ~9.7 s here, so an
            // auto-playing clip ends before a single overlay exists.
            link = CADisplayLink(target: self, selector: #selector(tick))
            link?.add(to: .main, forMode: .common)
        } catch {
            failure = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
        }
    }

    /// Cancelling `analysis` is the cooperative half: `analyse` re-checks at its
    /// stage boundaries, so a pass that has not reached `predict` stops there.
    func stop() {
        stopped = true
        analysis?.cancel(); analysis = nil
        link?.invalidate(); link = nil
        player?.pause(); shown = nil
        // The picker handed us a copy in the temp directory; it is ours to bin.
        if let source { try? FileManager.default.removeItem(at: source) }
    }

    func togglePlay() {
        guard let player else { return }
        // Parked at the very end, `play()` is a no-op until a seek back actually
        // lands — a dead-looking button — so the resume rides the seek's
        // completion. `isPlaying` is published by `tick`, every vsync.
        if player.rate > 0 { player.pause() } else if duration > 0, time >= duration - 0.05 {
            seek(to: 0, resuming: true)
        } else {
            player.play()
        }
    }

    /// Every drag update seeks **exactly**, which the cache needs as much as the
    /// spec does: a tolerant seek lands on the nearest sync sample, so one bar
    /// position would key different frames from drag to drag. Overlapping seeks
    /// need no queue of ours — `AVPlayer` drops the one in flight for the newest.
    func scrub(to fraction: Double, hasEnded: Bool) {
        guard duration > 0 else { return }
        if !scrubbing { player?.pause() }
        scrubbing = !hasEnded
        time = min(max(fraction, 0), 1) * duration
        seek(to: time)
    }

    private func seek(to seconds: Double, resuming: Bool = false) {
        player?.seek(to: CMTime(seconds: seconds, preferredTimescale: 600),
                     toleranceBefore: .zero, toleranceAfter: .zero) { [weak self] _ in
            if resuming { Task { @MainActor in self?.player?.play() } }
        }
    }

    @objc private func tick() {
        guard let output, let player else { return }
        // Paused, the timebase is not advancing, and "the time this vsync will
        // show" lags a completed seek by a beat; `currentTime()` is right there.
        var itemTime = output.itemTime(forHostTime: CACurrentMediaTime())
        if player.rate == 0 { itemTime = player.currentTime() }
        if (player.rate > 0) != isPlaying { isPlaying = player.rate > 0 }
        guard output.hasNewPixelBuffer(forItemTime: itemTime),
              let buffer = output.copyPixelBuffer(forItemTime: itemTime,
                                                  itemTimeForDisplay: nil) else { return }
        shown = VideoFrame(pixelBuffer: buffer, time: itemTime)
        shownSince = CACurrentMediaTime()
        let context = display ?? CIContext(); display = context
        let ciImage = CIImage(cvPixelBuffer: buffer)
        image = context.createCGImage(ciImage, from: ciImage.extent)
        if !scrubbing { time = itemTime.seconds }
        sync()
    }

    /// One place decides both what the overlay shows and what is analysed next. A
    /// cached result for the frame on screen is `.current`; anything else keeps the
    /// last finished result, flags it `.stale` with its own timestamp, and starts a
    /// pass — one at a time, so playback and scrubbing never wait on inference.
    /// Re-entered when a pass lands: the frame on screen has usually moved on, and
    /// paused on one frame there may never be another tick carrying a new buffer.
    private func sync() {
        guard let cache, let frame = shown else { return }
        let index = cache.index(for: frame.time.seconds)
        if let hit = cache.value(at: index) {
            // The number the cache exists for: picture on screen -> overlay right.
            if overlay != .current {
                NSLog("TPD video: frame at %.2f s got its overlay %.0f ms after appearing",
                      frame.time.seconds, (CACurrentMediaTime() - shownSince) * 1000)
            }
            result = hit
            overlay = .current
            return
        }
        overlay = resultTime.map(Overlay.stale) ?? .none
        guard !stopped, analysis == nil else { return }
        let seconds = frame.time.seconds
        analysis = Task { [weak self] in
            let fresh = try? await cache.result(at: index) {
                try await StillInferenceWorker.shared.analyse(frame)
            }
            guard let self else { return }
            analysis = nil
            if let fresh { result = fresh; resultTime = seconds }
            sync()
        }
    }
}

/// The picture and its overlay; chrome belongs to `PhotoPreviewView`.
struct VideoPreview: View {
    let model: VideoPreviewModel
    let url: URL
    let options: OverlayOptions

    var body: some View {
        ZStack {
            FramePreview(image: model.image)
            OverlayCanvas(result: model.result, options: options)
                // Unmistakable, and it degrades gracefully: a stale overlay is
                // still roughly informative, it stops looking authoritative.
                .opacity(model.overlay == .current ? 1 : 0.3)
            VStack(spacing: 0) { badge.padding(.top, 54); Spacer(minLength: 0) }
        }
        .task { await model.start(url: url) }
        .onDisappear { model.stop() }
    }

    private var badge: some View {
        let here = String(format: "%.2f s", model.time)
        var text = "\(here)  analysing this frame…", tint = Color.yellow
        if case .current = model.overlay { text = "\(here)  overlay matches"; tint = .green }
        if case .stale(let at) = model.overlay {
            text = String(format: "%@  overlay is from %.2f s — analysing", here, at)
            tint = .orange
        }
        if let failure = model.failure { text = failure; tint = .red }
        return Text(text)
            .font(.system(size: 11, weight: .medium).monospacedDigit())
            .foregroundStyle(tint)
            .padding(.horizontal, 11).padding(.vertical, 6)
            .background(.black.opacity(0.65), in: Capsule())
    }
}
