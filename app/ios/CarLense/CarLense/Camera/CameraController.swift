// CameraController.swift
//
// Thin AVCaptureSession wrapper. Owns the session, the back-camera input,
// and a video data output that drains 30fps BGRA frames into a serial
// processing queue. A delegate (the ViewModel) is notified for every frame.
//
// Why BGRA and not 420f? Vision's VNRecognizeObjectsRequest accepts both,
// but we also need to crop+preprocess for Core ML. BGRA is the simplest
// path: one plane, easy to feed into vImage / Core Image.
//
// Frame throttling is the ViewModel's job — this controller emits every
// frame and the downstream pipeline decides what to keep.

import AVFoundation
import CoreVideo
import Foundation

protocol CameraControllerDelegate: AnyObject {
    /// Called on `processingQueue` (NOT the main thread) for each frame.
    /// The pixel buffer is only valid for the duration of this call;
    /// callers MUST copy or process synchronously before returning.
    func camera(_ controller: CameraController,
                didOutput pixelBuffer: CVPixelBuffer,
                orientation: CGImagePropertyOrientation,
                timestamp: CMTime)
}

final class CameraController: NSObject {
    /// Public so CameraView can wire it into AVCaptureVideoPreviewLayer.
    let session = AVCaptureSession()

    weak var delegate: CameraControllerDelegate?

    /// All frame callbacks run on this queue. Mark `@Sendable` work as needed.
    let processingQueue = DispatchQueue(label: "com.ashzelazny.carlense.camera",
                                        qos: .userInitiated)

    private let videoOutput = AVCaptureVideoDataOutput()
    private var isConfigured = false

    enum CameraError: Error {
        case permissionDenied
        case noBackCamera
        case cannotAddInput
        case cannotAddOutput
    }

    /// Asks the user for camera permission if not already granted.
    /// Returns once a decision is recorded; throws on denial.
    func requestPermission() async throws {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            return
        case .notDetermined:
            let granted = await AVCaptureDevice.requestAccess(for: .video)
            if !granted { throw CameraError.permissionDenied }
        case .denied, .restricted:
            throw CameraError.permissionDenied
        @unknown default:
            throw CameraError.permissionDenied
        }
    }

    /// Configures the session (input + output) exactly once. Safe to call repeatedly.
    func configure() throws {
        guard !isConfigured else { return }
        session.beginConfiguration()
        defer { session.commitConfiguration() }

        // 1280x720 is the sweet spot: Vision detects cars well at this resolution
        // and the per-frame work fits inside the ~33ms 30fps budget on iPhone 13+.
        session.sessionPreset = .hd1280x720

        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera,
                                                   for: .video,
                                                   position: .back) else {
            throw CameraError.noBackCamera
        }

        let input = try AVCaptureDeviceInput(device: device)
        guard session.canAddInput(input) else { throw CameraError.cannotAddInput }
        session.addInput(input)

        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: processingQueue)

        guard session.canAddOutput(videoOutput) else { throw CameraError.cannotAddOutput }
        session.addOutput(videoOutput)

        // Lock orientation: app is portrait-only (see Info.plist).
        // iOS 17 prefers `videoRotationAngle` over the deprecated `videoOrientation` API.
        if let connection = videoOutput.connection(with: .video) {
            if connection.isVideoRotationAngleSupported(90) {
                connection.videoRotationAngle = 90 // portrait
            }
        }

        isConfigured = true
    }

    /// Starts the session on a background thread. Calling on the main thread
    /// triggers an Apple-runtime warning.
    func start() {
        processingQueue.async { [session] in
            if !session.isRunning {
                session.startRunning()
            }
        }
    }

    func stop() {
        processingQueue.async { [session] in
            if session.isRunning {
                session.stopRunning()
            }
        }
    }
}

extension CameraController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let timestamp = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        // The video pipeline is rotated to portrait via videoRotationAngle=90,
        // so the buffer is already upright. Vision still wants an orientation hint:
        // .up is correct for an already-rotated buffer.
        delegate?.camera(self,
                         didOutput: pixelBuffer,
                         orientation: .up,
                         timestamp: timestamp)
    }
}
