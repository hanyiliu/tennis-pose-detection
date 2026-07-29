//  VideoPreview.swift
//  An AVPlayer we pull frames from, the live view's overlay stack, and a badge.

import AVFoundation
import CoreImage
import QuartzCore
import SwiftUI

/// Rasterization, off main and *not* on the engine's parity `CIContext`: one
/// 608x1080 `createCGImage` measures **400–1375 ms** here, and in the tick it held
/// main 400 of every 430 ms — SwiftUI never redrew, so the transport looked dead.
actor FrameRasterizer {
    private var context: CIContext?
    func image(from frame: VideoFrame) -> CGImage? {
        let context = context ?? CIContext(); self.context = context
        let image = CIImage(cvPixelBuffer: frame.pixelBuffer)
        return context.createCGImage(image, from: image.extent)
    }
}

/// Playback, exact seeking and opportunistic analysis; `NSObject` for `@objc`.
@MainActor
@Observable
final class VideoPreviewModel: NSObject {
    /// `.stale` carries the time the overlay *does* describe — silently pinning a
    /// wrong one over a moving picture is worse than showing none.
    enum Overlay: Equatable { case none, current, stale(Double) }
    /// A result and the frame time it came from, one value for the reason
    /// `RenderedFrame` is: apart they drift, and the badge names a stranger.
    struct Held: Equatable { let result: TPDResult, time: Double }

    private(set) var image: CGImage?
    private(set) var held: Held?
    private(set) var overlay = Overlay.none
    private(set) var isPlaying = false
    private(set) var time: Double = 0
    private(set) var duration: Double = 0
    private(set) var failure: String?
    var result: TPDResult? { held?.result }   // never set without its timestamp
    @ObservationIgnored private var player: AVPlayer?
    @ObservationIgnored private var output: AVPlayerItemVideoOutput?
    @ObservationIgnored private var link: CADisplayLink?
    @ObservationIgnored private var cache: ResultCache?
    @ObservationIgnored private let raster = FrameRasterizer()
    @ObservationIgnored private var shown: VideoFrame?   // on screen; sync() re-targets
    @ObservationIgnored private var shownSince: CFTimeInterval = 0
    @ObservationIgnored private var analysis: Task<Void, Never>?
    @ObservationIgnored private var source: URL?
    /// `resuming` awaits its rewind; `stopped` blocks re-arming after teardown.
    @ObservationIgnored private var scrubbing = false
    @ObservationIgnored private var resuming = false
    @ObservationIgnored private var stopped = false
    @ObservationIgnored private var rastering = false

    /// What this presentation has already paid ~9.7 s a frame for, handed to an export so it
    /// re-infers only the frames nobody has looked at yet.
    var analysed: ResultSnapshot {
        ResultSnapshot(frameRate: cache?.frameRate ?? 30, entries: cache?.snapshot ?? [:])
    }

    /// Idempotent — the view's `.task` may re-run without the screen going away. All four
    /// awaits can be dismissed across; anything installed after `stop()` leaks, because the
    /// run loop retains the display link.
    func start(url: URL) async {
        guard player == nil, !stopped else { return }
        source = url
        let asset = AVURLAsset(url: url)
        do {
            guard let track = try await asset.loadTracks(withMediaType: .video).first else {
                throw FrameSourceError.videoDecodeUnavailable("clip contains no video track")
            }
            guard !stopped else { return }
            // As in `VideoFileFrameSource`: a bare output hands back the *encoded*
            // buffer, so a portrait clip arrives sideways, overlays and all.
            let composition = try await AVVideoComposition.videoComposition(withPropertiesOf: asset)
            guard !stopped else { return }
            duration = max(try await asset.load(.duration).seconds, 0)
            guard !stopped else { return }
            cache = ResultCache(frameRate: Double(try await track.load(.nominalFrameRate)))
            guard !stopped else { return }
            let item = AVPlayerItem(asset: asset); item.videoComposition = composition
            let output = AVPlayerItemVideoOutput(pixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
            item.add(output); self.output = output
            let player = AVPlayer(playerItem: item); self.player = player
            player.isMuted = true; player.actionAtItemEnd = .pause
            // Deliberately **not** `play()`: a pass costs ~9.7 s, so an
            // auto-playing clip would end before one overlay existed.
            link = CADisplayLink(target: self, selector: #selector(tick))
            link?.add(to: .main, forMode: .common)
        } catch {
            failure = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
        }
    }

    /// `analyse` re-checks at its boundaries; the temp copy is ours to bin.
    func stop() {
        stopped = true
        analysis?.cancel(); analysis = nil
        link?.invalidate(); link = nil
        player?.pause(); shown = nil; resuming = false
        if let source { try? FileManager.default.removeItem(at: source) }
    }

    /// Playback is an *intent*: at the end `play()` is a no-op, so a restart rewinds
    /// and `tick` starts it once the playhead moves — a `play()` beside the rewind races it.
    func togglePlay() {
        guard let player, !stopped else { return }
        if player.rate > 0 || resuming {
            resuming = false; player.pause()
        } else if duration > 0, player.currentTime().seconds >= duration - 0.05 {
            // Ask the PLAYER, not the published `time`: that is the last *rasterized*
            // frame's stamp and trails by seconds once the clip stops, so a tap in that
            // window read as "not at the end" and fell to a no-op `play()`.
            // `pause()` is not redundant at rate 0: it clears the played-to-end
            // disposition, without which the rewind is sometimes dropped.
            resuming = true; player.pause(); seek(to: 0)
        } else {
            player.play()
        }
    }

    /// Every drag seeks **exactly**: a tolerant seek lands on the nearest sync
    /// sample, so one bar position would key different frames from drag to drag and
    /// the cache would stop hitting. A drag also calls off a pending restart.
    func scrub(to fraction: Double, hasEnded: Bool) {
        guard duration > 0 else { return }
        if !scrubbing { resuming = false; player?.pause() }
        scrubbing = !hasEnded
        time = min(max(fraction, 0), 1) * duration
        seek(to: time)
    }

    private func seek(to seconds: Double) {
        player?.seek(to: CMTime(seconds: seconds, preferredTimescale: 600),
                     toleranceBefore: .zero, toleranceAfter: .zero)
    }
    @objc private func tick() {
        guard let output, let player else { return }
        // Paused, "the time this vsync shows" lags a seek; `currentTime()` does not.
        var itemTime = output.itemTime(forHostTime: CACurrentMediaTime())
        if player.rate == 0 { itemTime = player.currentTime() }
        // The restart's second half: the rewind has landed once the playhead is off
        // the end; re-asserted until the rate agrees so no dropped resume strands it.
        if resuming, player.currentTime().seconds < duration - 0.05 {
            player.play()
            resuming = player.rate == 0
        }
        let playing = player.rate > 0 || resuming
        if playing != isPlaying { isPlaying = playing }
        // One raster at a time, newest frame wins — the live view's drop gate —
        // and picture, playhead and overlay publish together when it lands, so the
        // badge still describes the frame on screen.
        guard !rastering, output.hasNewPixelBuffer(forItemTime: itemTime),
              let buffer = output.copyPixelBuffer(forItemTime: itemTime,
                                                  itemTimeForDisplay: nil) else { return }
        let frame = VideoFrame(pixelBuffer: buffer, time: itemTime); rastering = true
        Task { [weak self, raster] in
            let drawn = await raster.image(from: frame)
            guard let self else { return }
            rastering = false
            guard let drawn, !stopped else { return }
            image = drawn; shown = frame; shownSince = CACurrentMediaTime()
            if !scrubbing { time = frame.time.seconds }
            sync()
        }
    }
    /// What the badge may claim: a cached result for the frame on screen *is* that
    /// frame's and reads `.current`; anything else keeps the last pair and names its
    /// frame. Pure, so miss -> hit -> miss is a test rather than a scrub and a hope.
    static func resolve(frameAt seconds: Double, cached: TPDResult?,
                        held: Held?) -> (held: Held?, overlay: Overlay) {
        if let cached { return (Held(result: cached, time: seconds), .current) }
        return (held, held.map { Overlay.stale($0.time) } ?? .none)
    }
    /// One place decides the overlay and the next pass, one at a time; re-entered
    /// when a pass lands, because the frame on screen has usually moved on.
    private func sync() {
        guard let cache, let frame = shown else { return }
        let seconds = frame.time.seconds, index = cache.index(for: seconds)
        let hit = cache.value(at: index)
        if hit != nil, overlay != .current {   // the number the cache exists for
            NSLog("TPD video: frame at %.2f s got its overlay %.0f ms after appearing",
                  seconds, (CACurrentMediaTime() - shownSince) * 1000)
        }
        (held, overlay) = Self.resolve(frameAt: seconds, cached: hit, held: held)
        guard hit == nil, !stopped, analysis == nil else { return }
        analysis = Task { [weak self] in
            let fresh = try? await cache.result(at: index) {
                try await StillInferenceWorker.shared.analyse(frame) }
            guard let self else { return }
            analysis = nil
            if let fresh { held = Held(result: fresh, time: seconds) }
            sync()
        }
    }
}

/// The picture and its overlay; chrome belongs to `PhotoPreviewView`.
struct VideoPreview: View {
    let model: VideoPreviewModel, url: URL, options: OverlayOptions

    var body: some View {
        ZStack {
            FramePreview(image: model.image)
            OverlayCanvas(result: model.result, options: options)   // stale: still
                .opacity(model.overlay == .current ? 1 : 0.3)          // informative
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
            text = String(format: "%@  overlay is from %.2f s — analysing", here, at); tint = .orange
        }
        if let failure = model.failure { text = failure; tint = .red }
        return Text(text)
            .font(.system(size: 11, weight: .medium).monospacedDigit()).foregroundStyle(tint)
            .padding(.horizontal, 11).padding(.vertical, 6)
            .background(.black.opacity(0.65), in: Capsule())
    }
}
