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
    /// A pass for the frame on screen is in flight; published because the badge claims it.
    private(set) var analysing = false
    var result: TPDResult? { held?.result }   // never set without its timestamp
    @ObservationIgnored private var player: AVPlayer?
    @ObservationIgnored private var output: AVPlayerItemVideoOutput?
    @ObservationIgnored private var link: CADisplayLink?
    @ObservationIgnored private var cache: ResultCache?
    /// Itself `@Observable`, so the badge reading `model.preload.plan` tracks the sweep.
    let preload = PreloadCoordinator()
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

    /// What this presentation already paid ~9.7 s a frame for, so an export re-infers
    /// only the frames nobody looked at.
    var analysed: ResultSnapshot {
        ResultSnapshot(frameRate: cache?.frameRate ?? 30, entries: cache?.snapshot ?? [:])
    }

    /// Idempotent — the `.task` may re-run without the screen going away. All four awaits
    /// can be dismissed across; anything installed after `stop()` leaks the display link.
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
            let cache = ResultCache(frameRate: Double(try await track.load(.nominalFrameRate)))
            self.cache = cache
            guard !stopped else { return }
            // The whole clip, from frame 0 until the first tick moves the playhead. Nothing waits
            // on it: the sweep is a task whose every step is an await off this actor.
            preload.onResult = { [weak self] in self?.sync() }
            preload.begin(url: url, frameCount: Int((duration * cache.frameRate).rounded()),
                          frameRate: cache.frameRate, cache: cache)
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
        preload.cancel()
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
    /// The neighbour-tolerant form the sweep needs: a result from within the tolerance is a real
    /// answer, just not this frame's, so it is held with **its own** time and badged `.stale`.
    static func resolve(frameAt seconds: Double, near: (index: Int, result: TPDResult)?,
                        index: Int, frameRate: Double,
                        held: Held?) -> (held: Held?, overlay: Overlay) {
        guard let near, near.index != index else {
            return resolve(frameAt: seconds, cached: near?.result, held: held)
        }
        let at = Double(near.index) / (frameRate > 0 ? frameRate : 30)
        return (Held(result: near.result, time: at), .stale(at))
    }
    /// One place decides the overlay and the next pass, one at a time; re-entered
    /// when a pass lands, because the frame on screen has usually moved on.
    private func sync() {
        guard let cache, let frame = shown else { return }
        let seconds = frame.time.seconds, index = cache.index(for: seconds)
        // Re-aim the sweep first: it picks its next frame from here, so this is what makes the
        // frame the user just scrubbed to the next one analysed.
        preload.look(at: index)
        let near = cache.nearest(to: index, within: preload.tolerance)
        let next = Self.resolve(frameAt: seconds, near: near, index: index,
                                frameRate: cache.frameRate, held: held)
        // The number the cache exists for, against what is already published rather than just
        // "there was a hit": a neighbour answers every frame of a played-through clip.
        if near != nil, next.overlay != overlay {
            NSLog("TPD video: frame at %.2f s got its overlay %.0f ms after appearing",
                  seconds, (CACurrentMediaTime() - shownSince) * 1000)
        }
        (held, overlay) = next
        // Only an *exact* hit settles the frame on screen: a neighbour's answers a different
        // question, so the frame the user is parked on still owes a pass of its own or four in
        // five would show a stranger's pose forever. Deferred while the sweep is live — it owns
        // the engine, `look` has just aimed it here — so this fires once the sweep is done.
        guard near?.index != index, !preload.isSweeping, !stopped, analysis == nil else { return }
        analysing = true
        analysis = Task { [weak self] in
            let fresh = try? await cache.result(at: index) {
                try await StillInferenceWorker.shared.analyse(frame) }
            guard let self else { return }
            analysis = nil; analysing = false
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
            VStack(spacing: 6) { badge.padding(.top, 54); sweep; Spacer(minLength: 0) }
        }
        .task { await model.start(url: url) }
        .onDisappear { model.stop() }
    }

    private var badge: some View {
        let here = String(format: "%.2f s", model.time)
        // "analysing this frame" only when this frame is what is running: while the sweep holds
        // the engine the work in flight is a neighbour's, and the capsule below says so.
        var text = "\(here)  \(model.analysing ? "analysing this frame…" : "no overlay yet")"
        var tint = Color.yellow
        if case .current = model.overlay { text = "\(here)  overlay matches"; tint = .green }
        if case .stale(let at) = model.overlay {
            text = String(format: "%@  overlay is from %.2f s%@", here, at,
                          model.analysing ? " — analysing" : ""); tint = .orange
        }
        if let failure = model.failure { text = failure; tint = .red }
        return Text(text)
            .font(.system(size: 11, weight: .medium).monospacedDigit()).foregroundStyle(tint)
            .padding(.horizontal, 11).padding(.vertical, 6)
            .background(.black.opacity(0.65), in: Capsule())
    }

    /// The sweep, stated rather than hidden: at ~9.7 s a frame the user has to see the app work
    /// through the whole clip before deciding to wait. The cadence is named so "30 frames" over a
    /// 150-frame clip reads as the subsample it is.
    @ViewBuilder
    private var sweep: some View {
        if let plan = model.preload.plan, plan.total > 0 {
            let scope = plan.cadence == 1 ? "every frame" : "every \(plan.cadence)th frame"
            let tint = plan.isComplete ? Color.green : Color.yellow
            VStack(spacing: 5) {
                Text(plan.isComplete ? "clip analysed · \(plan.total) frames"
                                     : "analysing \(scope) · \(plan.analysed) of \(plan.total)")
                    .font(.system(size: 11, weight: .medium).monospacedDigit())
                ProgressView(value: Double(plan.analysed), total: Double(max(plan.total, 1)))
                    .frame(width: 170)
            }
            .tint(tint).foregroundStyle(tint).padding(.horizontal, 12).padding(.vertical, 7)
            .background(.black.opacity(0.65),
                        in: RoundedRectangle(cornerRadius: 13, style: .continuous))
            .accessibilityElement(children: .combine)
        }
    }
}
