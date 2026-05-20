// ViewModel.swift
//
// Owns the camera + ML pipeline and publishes detections + predictions to SwiftUI.
//
// Threading model:
//   - The CameraController invokes our delegate on its serial `processingQueue`.
//   - We run vehicle detection synchronously on that queue (Vision is fast,
//     ~10-30ms, and the queue is dedicated, so it doesn't block UI).
//   - Classification is dispatched off that queue onto a detached Task because
//     Core ML inference is ~30ms and we don't want to back-pressure detect.
//   - All `@Published` mutations hop to @MainActor.
//
// Pipeline:
//   1. detect()    : VNRecognizeObjectsRequest -> [Detection]
//      Throttle: every 2nd frame (15Hz at 30fps source).
//   2. top-3       : sort by score (bbox area × confidence), take 3.
//   3. classify()  : Core ML embed -> SGEMV vs prototypes -> top-K
//      Throttle: every 5th detection-pass per car slot (round-robin) so each
//      visible car is re-classified ~1 Hz. Plenty for stable labels.
//   4. publish     : hop to @MainActor and overwrite `detections`.
//
// MVP behavior: no tracker, no smoothing. Labels flicker between similar
// classes when the camera moves quickly — fine for v1.0.

import AVFoundation
import Combine
import CoreVideo
import Foundation
import QuartzCore
import SwiftUI
import Vision

@MainActor
final class ViewModel: ObservableObject {
    // MARK: - Published UI state
    @Published private(set) var detections: [(detection: Detection, prediction: Prediction?)] = []
    @Published private(set) var fps: Double = 0
    @Published private(set) var lastClassifyMs: Double = 0
    @Published private(set) var errorMessage: String?

    // MARK: - Camera plumbing
    private let camera = CameraController()
    /// Exposed for CameraView so it can attach AVCaptureVideoPreviewLayer.
    var cameraSession: AVCaptureSession { camera.session }

    // MARK: - ML (touched from camera queue + main actor; all internals are
    // reference types whose mutable state is contained within them).
    private let pipeline = PipelineState()

    init() {
        camera.delegate = self
    }

    /// Called once from ContentView.onAppear. Idempotent.
    func start() {
        Task { @MainActor in
            do {
                try await camera.requestPermission()
            } catch {
                errorMessage = "Camera permission denied."
                return
            }

            // Load models on a background task — this is ~200ms cold.
            Task.detached(priority: .userInitiated) { [pipeline] in
                do {
                    print("[classifier] loading from bundle…")
                    let cls = try Classifier.loadFromBundle()
                    print("[classifier] loaded ok")
                    pipeline.installClassifier(cls)
                } catch {
                    print("[classifier] LOAD FAILED: \(error)")
                    await MainActor.run {
                        self.errorMessage = "Model load failed: \(error)"
                    }
                }
            }

            do {
                try camera.configure()
                camera.start()
            } catch {
                errorMessage = "Camera setup failed: \(error)"
            }
        }
    }

    func stop() {
        camera.stop()
    }

    // MARK: - Publish (called from camera queue, hops to MainActor)

    fileprivate nonisolated func publish(detections snapshot: [(Detection, Prediction?)]?,
                                         fps: Double?,
                                         lastClassifyMs: Double?) {
        Task { @MainActor in
            if let snapshot { self.detections = snapshot }
            if let f = fps { self.fps = f }
            if let ms = lastClassifyMs { self.lastClassifyMs = ms }
        }
    }
}

// MARK: - PipelineState (camera-queue resident)

/// Holds all pipeline state that lives on the camera queue. Splitting it out
/// of the @MainActor ViewModel lets us run detection and classification
/// dispatch without bouncing to main per-frame.
private final class PipelineState {
    private let detector = VehicleDetector()
    private var classifier: Classifier?

    private let detectionEveryNFrames: Int = 2
    private let classifyEveryNDetections: Int = 5

    private var frameCounter: Int = 0
    private var detectionCounter: Int = 0
    private var roundRobinSlot: Int = 0
    private var lastPredictions: [Int: Prediction] = [:]
    private var classifyInFlight: Bool = false

    // Temporal smoothing: per-slot history of raw classifier outputs.
    // Smoothed prediction = mode-vote on top-1 displayName, confidence =
    // mean confidence of winning-class frames × (winning-class vote share).
    // This reduces flicker AND raises perceived confidence when the classifier
    // is consistent across frames — exactly the v1.1 behavior the README promised.
    private var predictionHistory: [Int: [Prediction]] = [:]
    private let smoothingWindow: Int = 8

    private var lastFpsTick: CFTimeInterval = CACurrentMediaTime()
    private var framesSinceFpsTick: Int = 0

    /// Owner-installed once model load finishes. Reads on camera queue are racey
    /// in the strict sense but it's a single-pointer write to a reference type;
    /// worst case is one frame skipped until the publish reaches the queue.
    func installClassifier(_ cls: Classifier) {
        // Marshal onto the camera queue is overkill; this is invoked rarely.
        self.classifier = cls
    }

    /// Process one frame on the camera queue. Returns the snapshot to publish
    /// (the caller hops to MainActor).
    func processFrame(pixelBuffer: CVPixelBuffer,
                      orientation: CGImagePropertyOrientation,
                      publish: @escaping ([(Detection, Prediction?)]?, Double?, Double?) -> Void) {
        frameCounter &+= 1
        framesSinceFpsTick += 1
        let now = CACurrentMediaTime()
        var fpsUpdate: Double? = nil
        if now - lastFpsTick > 1.0 {
            fpsUpdate = Double(framesSinceFpsTick) / (now - lastFpsTick)
            framesSinceFpsTick = 0
            lastFpsTick = now
        }

        // Throttle detection: skip this frame's detect, but still publish
        // the FPS update if we ticked over a 1s window.
        guard frameCounter % detectionEveryNFrames == 0 else {
            if let f = fpsUpdate { publish(nil, f, nil) }
            return
        }
        detectionCounter &+= 1

        // Biggest-car-only: take just the single most prominent detection
        // (largest bbox × confidence). Multi-car classification was producing
        // jumpy labels and the smoother needs one steady target.
        let rawDets = detector.detect(in: pixelBuffer, orientation: orientation)
            .sorted { $0.score > $1.score }
            .prefix(1)

        // Re-index by sorted position so detection.id is stable for round-robin slots.
        var dets: [Detection] = []
        dets.reserveCapacity(rawDets.count)
        for (offset, det) in rawDets.enumerated() {
            dets.append(Detection(id: offset,
                                  bbox: det.bbox,
                                  confidence: det.confidence,
                                  coarseLabel: det.coarseLabel))
        }

        let snapshot: [(Detection, Prediction?)] = dets.map { ($0, lastPredictions[$0.id]) }
        publish(snapshot, fpsUpdate, nil)

        // Kick off one classify per detection-tick on a detached task.
        guard let classifier = self.classifier else { return }
        guard !classifyInFlight else { return }
        guard detectionCounter % classifyEveryNDetections == 0 else { return }
        guard !dets.isEmpty else { return }

        let slot = roundRobinSlot % dets.count
        roundRobinSlot &+= 1
        let target = dets[slot]
        classifyInFlight = true

        // CVPixelBuffer is a CF type; capturing it in the closure retains it
        // for the duration of the Task — safe to use off the camera queue.
        let bufCopy = pixelBuffer
        let bbox = target.bbox
        let targetId = target.id

        Task.detached(priority: .userInitiated) { [weak self] in
            let t0 = CACurrentMediaTime()
            let preds: [Prediction]
            do {
                preds = try classifier.classify(pixelBuffer: bufCopy, bbox: bbox)
                let summary = preds.map {
                    "\($0.displayName)@\(String(format: "%.2f", $0.confidence))"
                }.joined(separator: " | ")
                print("[classify] top-\(preds.count): \(summary)")
            } catch {
                print("[classify] THREW: \(error)")
                preds = []
            }
            let dt = (CACurrentMediaTime() - t0) * 1000
            self?.applyClassify(targetId: targetId, preds: preds, latencyMs: dt, publish: publish)
        }
    }

    /// Apply classify result. Called from the detached classify task.
    private func applyClassify(targetId: Int,
                               preds: [Prediction],
                               latencyMs: Double,
                               publish: @escaping ([(Detection, Prediction?)]?, Double?, Double?) -> Void) {
        classifyInFlight = false
        if let top = preds.first {
            var hist = predictionHistory[targetId] ?? []
            hist.append(top)
            if hist.count > smoothingWindow {
                hist.removeFirst(hist.count - smoothingWindow)
            }
            predictionHistory[targetId] = hist
            lastPredictions[targetId] = Self.smoothed(history: hist) ?? top
        }
        publish(nil, nil, latencyMs)
    }

    /// Mode-vote smoothing over a window of raw classifier outputs.
    private static func smoothed(history: [Prediction]) -> Prediction? {
        guard !history.isEmpty else { return nil }
        var counts: [String: (count: Int, sumConf: Float, classId: String)] = [:]
        for p in history {
            var entry = counts[p.displayName] ?? (0, 0, p.classId)
            entry.count += 1
            entry.sumConf += p.confidence
            counts[p.displayName] = entry
        }
        guard let winner = counts.max(by: { $0.value.count < $1.value.count }) else { return nil }
        let avgConf = winner.value.sumConf / Float(winner.value.count)
        return Prediction(classId: winner.value.classId,
                          displayName: winner.key,
                          confidence: avgConf)
    }
}

// MARK: - CameraControllerDelegate

extension ViewModel: CameraControllerDelegate {
    nonisolated func camera(_ controller: CameraController,
                            didOutput pixelBuffer: CVPixelBuffer,
                            orientation: CGImagePropertyOrientation,
                            timestamp: CMTime) {
        pipeline.processFrame(pixelBuffer: pixelBuffer, orientation: orientation) {
            [weak self] snapshot, fpsUpdate, latencyUpdate in
            self?.publish(detections: snapshot, fps: fpsUpdate, lastClassifyMs: latencyUpdate)
        }
    }
}
