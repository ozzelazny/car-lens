// VehicleDetector.swift
//
// Vehicle detector built on Vision's built-in VNRecognizeObjectsRequest.
// Apple's stock model is good enough for the MVP — it recognizes cars, trucks,
// buses, vans, SUVs, and motorcycles at low single-digit-ms latency on the
// Neural Engine and ships with the OS, so we don't pay app-binary size for it.
//
// If we want tighter bounding boxes (Vision's are coarse) we can drop in a
// shipped YOLOv8-n trained on COCO-vehicle classes — that's a v1.1 task.
//
// Output is a list of `Detection` in Vision-space normalized coords.

import CoreVideo
import Foundation
import Vision

final class VehicleDetector {

    /// Vision's coarse vehicle labels. Anything else (people, food, etc.) is dropped.
    private static let vehicleLabels: Set<String> = [
        "Car", "Truck", "Bus", "Van", "SUV", "Motorcycle",
        // Lowercase variants — Vision label casing varies by iOS minor version.
        "car", "truck", "bus", "van", "suv", "motorcycle"
    ]

    /// Minimum detector confidence to surface a detection at all.
    /// Tuned empirically: <0.30 is mostly false positives.
    private static let minConfidence: Float = 0.30

    private let request: VNRecognizeObjectsRequest

    init() {
        // Apple's stock recognizer. Errors here would be a system-level problem;
        // we crash rather than silently ship a no-op detector.
        do {
            self.request = try VNRecognizeObjectsRequest()
        } catch {
            fatalError("Failed to initialize VNRecognizeObjectsRequest: \(error)")
        }
        // Default model revision is fine; pin it later if we see drift across OS updates.
    }

    /// Runs detection synchronously on the given pixel buffer.
    ///
    /// Vision is thread-safe per-request as long as we don't share request
    /// instances across queues. The ViewModel's processing queue is serial,
    /// so one handler call at a time is correct.
    func detect(in pixelBuffer: CVPixelBuffer,
                orientation: CGImagePropertyOrientation) -> [Detection] {
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer,
                                            orientation: orientation,
                                            options: [:])
        do {
            try handler.perform([request])
        } catch {
            // A failed handler perform means the system rejected this specific
            // buffer (rare). Drop the frame and continue.
            return []
        }

        guard let observations = request.results else { return [] }

        var detections: [Detection] = []
        detections.reserveCapacity(observations.count)
        for (index, obs) in observations.enumerated() {
            guard let top = obs.labels.first else { continue }
            guard Self.vehicleLabels.contains(top.identifier) else { continue }
            guard obs.confidence >= Self.minConfidence else { continue }
            detections.append(Detection(
                id: index,
                bbox: obs.boundingBox,
                confidence: obs.confidence,
                coarseLabel: top.identifier
            ))
        }
        return detections
    }
}
