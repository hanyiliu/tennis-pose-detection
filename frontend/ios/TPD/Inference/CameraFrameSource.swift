//  CameraFrameSource.swift
//  Device frame source: AVCaptureSession -> AsyncStream<VideoFrame>.
//
//  Runs on hardware only. In the Simulator `AVCaptureDevice.default` returns
//  nil, which is why nothing here force-unwraps a device and why
//  `FrameSourceFactory` picks `VideoFileFrameSource` there instead.

import AVFoundation

final class CameraFrameSource: NSObject, FrameSource, @unchecked Sendable {
    /// Portrait. iOS 17 replaced `AVCaptureConnection.videoOrientation` (now
    /// deprecated) with a rotation angle in degrees, counter-clockwise from the
    /// sensor's native landscape-right. 90° is portrait, which is the only
    /// orientation this app supports (`UISupportedInterfaceOrientations`).
    private static let portraitRotationAngle: CGFloat = 90

    let frames: FrameStream
    private let continuation: FrameStream.Continuation

    /// `@unchecked Sendable` rests on these two invariants, both enforced by
    /// `dispatchPrecondition` below:
    ///   * `session`, `output` and `isConfigured` are touched only on
    ///     `sessionQueue`;
    ///   * sample buffers arrive only on `sampleQueue`, and the only thing that
    ///     path touches is `continuation`, which is `Sendable` by itself.
    private let session = AVCaptureSession()
    private let output = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(label: "com.hanyi.TPD.camera.session")
    private let sampleQueue = DispatchQueue(label: "com.hanyi.TPD.camera.samples")
    private let position: AVCaptureDevice.Position
    private var isConfigured = false

    init(position: AVCaptureDevice.Position = .back) {
        self.position = position
        (frames, continuation) = FrameStream.makeLatestWins()
        super.init()
    }

    deinit {
        continuation.finish()
    }

    // MARK: - Lifecycle

    func start() async throws {
        try await requestAccess()
        try await withCheckedThrowingContinuation { (resume: CheckedContinuation<Void, Error>) in
            sessionQueue.async { [self] in
                do {
                    try configureIfNeeded()
                    // Re-attached on every start because `stop()` detaches it to
                    // break the session -> output -> self reference cycle.
                    output.setSampleBufferDelegate(self, queue: sampleQueue)
                    if !session.isRunning {
                        session.startRunning()
                    }
                    resume.resume()
                } catch {
                    resume.resume(throwing: error)
                }
            }
        }
    }

    /// Idempotent, and safe to call from any thread. Parks the session without
    /// finishing the stream, so a later `start()` resumes into the same stream.
    func stop() {
        sessionQueue.async { [self] in
            if session.isRunning {
                session.stopRunning()
            }
            output.setSampleBufferDelegate(nil, queue: nil)
        }
    }

    // MARK: - Authorization

    private func requestAccess() async throws {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            return
        case .notDetermined:
            // Presents the system prompt; the Info.plist string is
            // NSCameraUsageDescription.
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
        // frame that is already stale because inference ran long. The stream's
        // `.bufferingNewest(1)` policy is the second.
        output.alwaysDiscardsLateVideoFrames = true
        // 32BGRA is what CoreImage and Core ML both consume without a second
        // conversion; the capture pipeline does the YUV->BGRA step in hardware.
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
            // Mirror the selfie camera so the preview matches what the user
            // expects; the engine sees the same pixels the user sees.
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
        // underneath the consumer. Exactly one frame is ever held back, which is
        // why the pool cannot starve.
        continuation.yield(VideoFrame(pixelBuffer: pixelBuffer,
                                      time: CMSampleBufferGetPresentationTimeStamp(sampleBuffer)))
    }
}
