//  VideoFileFrameSource.swift
//  Simulator (and desk-testing) frame source: a looped bundled clip decoded
//  into the same AsyncStream<VideoFrame> the camera feeds.
//
//  THE CLIP IS NOT IN THE REPOSITORY. `sample_rally.mp4` is a video binary and
//  this repo has never used Git LFS, so the clip is supplied separately: drop
//  it at `frontend/ios/TPD/Media/sample_rally.mp4` and re-run `make generate`
//  — the app target's `sources: [TPD]` directory glob bundles it with no
//  project.yml edit. Do not commit it. When it is absent this source fails with
//  `FrameSourceError.missingBundledVideo`, which the UI can show; it must never
//  crash, because a fresh clone genuinely has no clip.

import AVFoundation
import QuartzCore

final class VideoFileFrameSource: NSObject, FrameSource, @unchecked Sendable {
    let frames: FrameStream
    private let continuation: FrameStream.Continuation

    private let resource: String
    private let fileExtension: String
    private let bundle: Bundle

    /// `@unchecked Sendable`: everything below is created and touched only on
    /// the main queue (`dispatchPrecondition` enforces it), and the display-link
    /// callback — also main — touches nothing else but `continuation`, which is
    /// `Sendable` on its own.
    private var player: AVPlayer?
    private var videoOutput: AVPlayerItemVideoOutput?
    private var displayLink: CADisplayLink?

    init(resource: String = "sample_rally",
         fileExtension: String = "mp4",
         bundle: Bundle = .main) {
        self.resource = resource
        self.fileExtension = fileExtension
        self.bundle = bundle
        (frames, continuation) = FrameStream.makeLatestWins()
        super.init()
    }

    deinit {
        NotificationCenter.default.removeObserver(self)
        continuation.finish()
    }

    // MARK: - Lifecycle

    func start() async throws {
        guard let url = bundle.url(forResource: resource, withExtension: fileExtension) else {
            throw FrameSourceError.missingBundledVideo(resource: resource,
                                                       fileExtension: fileExtension)
        }
        // Validate before touching the player: a clip with no video track would
        // otherwise "play" forever and yield nothing, which looks like a hang.
        // This probe asset stays here rather than being handed to the main queue
        // — AVURLAsset is not Sendable, and re-creating one from the URL costs
        // nothing (no I/O happens until a property is loaded).
        do {
            let tracks = try await AVURLAsset(url: url).loadTracks(withMediaType: .video)
            guard !tracks.isEmpty else {
                throw FrameSourceError.videoDecodeUnavailable("clip contains no video track")
            }
        } catch let error as FrameSourceError {
            throw error
        } catch {
            throw FrameSourceError.videoDecodeUnavailable(error.localizedDescription)
        }

        await withCheckedContinuation { (resume: CheckedContinuation<Void, Never>) in
            DispatchQueue.main.async { [self] in
                startOnMain(url: url)
                resume.resume()
            }
        }
    }

    /// Idempotent, callable from any thread.
    ///
    /// Invalidating the display link is mandatory, not tidiness: `CADisplayLink`
    /// holds a **strong** reference to its target, and the run loop holds the
    /// link, so a live link keeps `self` alive forever and `deinit` never runs.
    /// `start()` builds a fresh link when it needs one.
    func stop() {
        DispatchQueue.main.async { [self] in
            displayLink?.invalidate()
            displayLink = nil
            player?.pause()
        }
    }

    // MARK: - Main-queue internals

    private func startOnMain(url: URL) {
        dispatchPrecondition(condition: .onQueue(.main))

        if player == nil {
            let item = AVPlayerItem(asset: AVURLAsset(url: url))
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
            // of the template item, and an AVPlayerItemVideoOutput attached to
            // the template does not follow those copies — frames would stop
            // arriving at the first wrap. One item plus a seek-to-zero on the
            // end notification keeps our output attached for the whole run.
            player.actionAtItemEnd = .none
            NotificationCenter.default.addObserver(
                self,
                selector: #selector(itemDidPlayToEnd(_:)),
                name: AVPlayerItem.didPlayToEndTimeNotification,
                object: item
            )
            self.player = player
        }

        if displayLink == nil {
            let link = CADisplayLink(target: self, selector: #selector(tick(_:)))
            link.add(to: .main, forMode: .common)
            displayLink = link
        }
        player?.play()
    }

    @objc private func itemDidPlayToEnd(_ notification: Notification) {
        // Posted on the notification's own thread, so hop before touching the
        // player. `.zero` + play is the whole loop.
        DispatchQueue.main.async { [self] in
            player?.seek(to: .zero)
            player?.play()
        }
    }

    @objc private func tick(_ link: CADisplayLink) {
        dispatchPrecondition(condition: .onQueue(.main))
        guard let videoOutput else { return }
        // `targetTimestamp` is the host time this frame will actually be shown
        // at, so it asks the output for the frame the display is about to need
        // rather than the one it has already missed.
        let itemTime = videoOutput.itemTime(forHostTime: link.targetTimestamp)
        guard videoOutput.hasNewPixelBuffer(forItemTime: itemTime),
              let pixelBuffer = videoOutput.copyPixelBuffer(forItemTime: itemTime,
                                                            itemTimeForDisplay: nil) else {
            // No new frame this vsync (clip fps < display fps, or mid-seek).
            // Yielding the previous buffer again would just make the consumer
            // re-run inference on a frame it already processed.
            return
        }
        continuation.yield(VideoFrame(pixelBuffer: pixelBuffer, time: itemTime))
    }
}
