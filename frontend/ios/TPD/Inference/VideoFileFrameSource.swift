//  VideoFileFrameSource.swift
//  Simulator (and desk-testing) frame source: a looped bundled clip decoded
//  into the same AsyncStream<VideoFrame> the camera feeds.
//
//  THE CLIP IS NOT IN THE REPOSITORY. `sample_rally.mp4` is a video binary and
//  this repo has never used Git LFS, so it is supplied separately: drop it at
//  `frontend/ios/TPD/Media/sample_rally.mp4` and re-run `make generate` — the
//  `sources: [TPD]` glob bundles it with no project.yml edit. Do not commit it.
//  When absent this source throws `.missingBundledVideo`, which the UI can
//  show; it must never crash, since a fresh clone has no clip.

import AVFoundation
import QuartzCore

final class VideoFileFrameSource: NSObject, FrameSource, @unchecked Sendable {
    private let resource: String
    private let fileExtension: String
    private let bundle: Bundle

    /// `@unchecked Sendable`: the three below are created and touched only on
    /// main (`@MainActor` and `dispatchPrecondition` enforce it), and the
    /// display-link callback touches nothing but `lifecycle`, which locks.
    private let lifecycle = FrameSourceLifecycle()
    private var player: AVPlayer?
    private var videoOutput: AVPlayerItemVideoOutput?
    private var displayLink: CADisplayLink?

    init(resource: String = "sample_rally",
         fileExtension: String = "mp4",
         bundle: Bundle = .main) {
        self.resource = resource
        self.fileExtension = fileExtension
        self.bundle = bundle
        super.init()
    }

    deinit {
        NotificationCenter.default.removeObserver(self)
        lifecycle.stop()
    }

    // MARK: - Lifecycle

    /// Returns the stream for this run; a stop/start cycle vends a new one.
    func start() async throws -> FrameStream {
        let (stream, token) = lifecycle.begin()
        guard let url = bundle.url(forResource: resource, withExtension: fileExtension) else {
            throw FrameSourceError.missingBundledVideo(resource: resource,
                                                       fileExtension: fileExtension)
        }
        try await startOnMain(url: url, token: token)
        return stream
    }

    /// Idempotent, callable from any thread. Records the intent to stop
    /// synchronously — that is what a `start()` parked on an await re-reads —
    /// then tears down on main. Invalidating the link is mandatory: it retains
    /// its target and the run loop retains it, so a live link would keep `self`
    /// alive forever and `deinit` would never run. `start()` builds a fresh one.
    func stop() {
        lifecycle.stop()
        let stopped = lifecycle.latest
        DispatchQueue.main.async { [self] in
            // Unlike the camera's serial session queue, this teardown and
            // `startOnMain` reach main by different routes, so nothing orders
            // them. Stand down if a start() has already superseded this stop.
            guard lifecycle.latest == stopped else { return }
            displayLink?.invalidate()
            displayLink = nil
            player?.pause()
        }
    }

    // MARK: - Main-actor internals

    /// The player, its output and the display link are all non-`Sendable` and
    /// live on the main actor for their whole lifetime — asset loads included,
    /// which is why nothing has to cross an isolation boundary.
    @MainActor
    private func startOnMain(url: URL, token: UInt64) async throws {
        if player == nil {
            let asset = AVURLAsset(url: url)
            let composition: AVVideoComposition
            do {
                // Validate before touching the player: a clip with no video
                // track would "play" forever and yield nothing — a hang, from
                // the outside.
                let tracks = try await asset.loadTracks(withMediaType: .video)
                guard !tracks.isEmpty else {
                    throw FrameSourceError.videoDecodeUnavailable("clip contains no video track")
                }
                // This is what keeps VideoFrame's "upright" invariant true here.
                // A bare AVPlayerItemVideoOutput hands back the *encoded* buffer
                // and ignores the track's `preferredTransform`, so a clip shot in
                // portrait — encoded landscape plus a 90° transform, which is how
                // every iPhone records — would arrive sideways while the camera
                // arrives portrait, and overlay geometry from one would be wrong
                // against the other. This bakes the transform into the output.
                composition = try await AVVideoComposition.videoComposition(withPropertiesOf: asset)
            } catch let error as FrameSourceError {
                throw error
            } catch {
                throw FrameSourceError.videoDecodeUnavailable(error.localizedDescription)
            }

            let item = AVPlayerItem(asset: asset)
            item.videoComposition = composition
            // 32BGRA to match what CameraFrameSource delivers, so the engine
            // sees one pixel format regardless of which source is running.
            let output = AVPlayerItemVideoOutput(pixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
            ])
            item.add(output)
            videoOutput = output

            let player = AVPlayer(playerItem: item)
            player.isMuted = true
            // Manual loop rather than AVPlayerLooper: the looper plays *copies*
            // of the template item and an AVPlayerItemVideoOutput attached to
            // the template does not follow them — frames would stop at the first
            // wrap. One item plus seek-to-zero keeps our output attached.
            player.actionAtItemEnd = .none
            NotificationCenter.default.addObserver(
                self,
                selector: #selector(itemDidPlayToEnd(_:)),
                name: AVPlayerItem.didPlayToEndTimeNotification,
                object: item
            )
            self.player = player
        }

        // Both loads above suspend, and a stop() during either must win; its
        // teardown can land on either side of this line, so intent decides.
        guard lifecycle.isCurrent(token) else { return }
        if displayLink == nil {
            let link = CADisplayLink(target: self, selector: #selector(tick(_:)))
            link.add(to: .main, forMode: .common)
            displayLink = link
        }
        player?.play()
    }

    @objc private func itemDidPlayToEnd(_ notification: Notification) {
        // Posted on the notification's own thread; hop first. Seek + play is
        // the whole loop.
        DispatchQueue.main.async { [self] in
            player?.seek(to: .zero)
            player?.play()
        }
    }

    @objc private func tick(_ link: CADisplayLink) {
        dispatchPrecondition(condition: .onQueue(.main))
        guard let videoOutput else { return }
        // `targetTimestamp` is the host time this frame will actually be shown
        // at, so we ask for the frame the display is about to need.
        let itemTime = videoOutput.itemTime(forHostTime: link.targetTimestamp)
        guard videoOutput.hasNewPixelBuffer(forItemTime: itemTime),
              let pixelBuffer = videoOutput.copyPixelBuffer(forItemTime: itemTime,
                                                            itemTimeForDisplay: nil) else {
            // No new frame this vsync (clip fps < display fps, or mid-seek).
            // Re-yielding the last buffer would just make the consumer re-run
            // inference on a frame it already processed.
            return
        }
        lifecycle.yield(VideoFrame(pixelBuffer: pixelBuffer, time: itemTime))
    }
}
