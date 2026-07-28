//  CameraFrameSource.swift
//  Device frame source: AVCaptureSession -> AsyncStream<VideoFrame>. Hardware
//  only — in the Simulator `AVCaptureDevice.default` returns nil, which is why
//  nothing here force-unwraps a device and why `FrameSourceFactory` picks
//  `VideoFileFrameSource` there.

import AVFoundation

final class CameraFrameSource: NSObject, FrameSource, @unchecked Sendable {
    /// iOS 17 replaced the deprecated `AVCaptureConnection.videoOrientation`
    /// with a rotation angle in degrees, counter-clockwise from the sensor's
    /// native landscape-right. 90° is portrait — the only orientation this app
    /// supports (`UISupportedInterfaceOrientations`).
    private static let portraitRotationAngle: CGFloat = 90

    /// `@unchecked Sendable` rests on two invariants, both enforced by
    /// `dispatchPrecondition` below: `session`, `output` and `isConfigured` are
    /// touched only on `sessionQueue`; sample buffers arrive only on
    /// `sampleQueue` and touch nothing but `lifecycle`, which locks internally.
    private let lifecycle = FrameSourceLifecycle()
    private let session = AVCaptureSession()
    private let output = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(label: "com.hanyi.TPD.camera.session")
    private let sampleQueue = DispatchQueue(label: "com.hanyi.TPD.camera.samples")
    private let position: AVCaptureDevice.Position
    private var isConfigured = false

    init(position: AVCaptureDevice.Position = .back) {
        self.position = position
        super.init()
    }

    deinit { lifecycle.stop() }

    // MARK: - Lifecycle

    /// Returns the stream for this run; a stop/start cycle vends a new one.
    func start() async throws -> FrameStream {
        let (stream, token) = lifecycle.begin()
        try await requestAccess()
        // The permission prompt can sit on screen indefinitely; a stop() during
        // it must win rather than be overwritten by the work queued behind it.
        guard lifecycle.isCurrent(token) else { return stream }
        try await withCheckedThrowingContinuation { (resume: CheckedContinuation<Void, Error>) in
            sessionQueue.async { [self] in
                // Re-checked here too: stop()'s teardown uses this same queue
                // and may land either side of this block, so intent decides.
                guard lifecycle.isCurrent(token) else {
                    resume.resume()
                    return
                }
                do {
                    try configureIfNeeded()
                    // Re-attached on every start because `stop()` detaches it to
                    // break the session -> output -> self reference cycle.
                    output.setSampleBufferDelegate(self, queue: sampleQueue)
                    if !session.isRunning { session.startRunning() }
                    resume.resume()
                } catch {
                    resume.resume(throwing: error)
                }
            }
        }
        return stream
    }

    /// Idempotent and safe from any thread: records the stop intent
    /// synchronously (what a parked `start()` re-reads), then parks the session.
    func stop() {
        lifecycle.stop()
        sessionQueue.async { [self] in
            if session.isRunning { session.stopRunning() }
            output.setSampleBufferDelegate(nil, queue: nil)
        }
    }

    // MARK: - Authorization

    private func requestAccess() async throws {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            return
        case .notDetermined:
            // Presents the system prompt (Info.plist NSCameraUsageDescription).
            guard await AVCaptureDevice.requestAccess(for: .video) else {
                throw FrameSourceError.cameraAccessDenied
            }
        case .denied, .restricted:
            throw FrameSourceError.cameraAccessDenied
        @unknown default:
            throw FrameSourceError.cameraAccessDenied
        }
    }

    // MARK: - Session configuration

    private func configureIfNeeded() throws {
        dispatchPrecondition(condition: .onQueue(sessionQueue))
        guard !isConfigured else { return }

        // Not a force-unwrap: this is exactly the nil the Simulator returns.
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera,
                                                   for: .video,
                                                   position: position) else {
            throw FrameSourceError.cameraUnavailable
        }
        let input: AVCaptureDeviceInput
        do {
            input = try AVCaptureDeviceInput(device: device)
        } catch {
            throw FrameSourceError.captureConfigurationFailed(error.localizedDescription)
        }

        session.beginConfiguration()
        defer { session.commitConfiguration() }

        session.sessionPreset = .high
        guard session.canAddInput(input) else {
            throw FrameSourceError.captureConfigurationFailed("session rejected the camera input")
        }
        session.addInput(input)

        // Latest-frame-wins, first line of defence: never hand the delegate a
        // frame already stale because inference ran long. The stream's
        // `.bufferingNewest(1)` policy is the second.
        output.alwaysDiscardsLateVideoFrames = true
        // 32BGRA is what CoreImage and Core ML both consume without a second
        // conversion; capture does the YUV->BGRA step in hardware.
        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        guard session.canAddOutput(output) else {
            throw FrameSourceError.captureConfigurationFailed("session rejected the video output")
        }
        session.addOutput(output)

        if let connection = output.connection(with: .video) {
            if connection.isVideoRotationAngleSupported(Self.portraitRotationAngle) {
                connection.videoRotationAngle = Self.portraitRotationAngle
            }
            // Mirror the selfie camera to match what the user expects; the
            // engine sees the same pixels the user sees.
            if position == .front, connection.isVideoMirroringSupported {
                connection.automaticallyAdjustsVideoMirroring = false
                connection.isVideoMirrored = true
            }
        }

        isConfigured = true
    }
}

// MARK: - Sample buffer delegate

extension CameraFrameSource: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        dispatchPrecondition(condition: .onQueue(sampleQueue))
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        // Yielding retains the buffer into the stream's one-slot store, so it
        // outlives this callback and cannot be recycled by the capture pool
        // under the consumer. Only one frame is held back, so the pool cannot
        // starve. A frame arriving after stop() is dropped on the floor.
        lifecycle.yield(VideoFrame(pixelBuffer: pixelBuffer,
                                   time: CMSampleBufferGetPresentationTimeStamp(sampleBuffer)))
    }
}
