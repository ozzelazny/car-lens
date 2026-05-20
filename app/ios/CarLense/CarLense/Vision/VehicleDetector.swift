// VehicleDetector.swift
//
// Vehicle detector built on Apple's YOLOv3-Tiny Core ML model. The bundled
// pipeline does image preprocessing, YOLO inference, and NMS in one shot, so
// Vision returns `VNRecognizedObjectObservation`s directly. COCO classes —
// we filter to the four wheeled-vehicle categories.
//
// Output is a list of `Detection` in Vision-space normalized coords.

import CoreML
import CoreVideo
import Foundation
import Vision

final class VehicleDetector {

    /// COCO vehicle class names emitted by YOLOv3-Tiny.
    private static let vehicleLabels: Set<String> = [
        "car", "truck", "bus", "motorcycle"
    ]

    /// Minimum detector confidence to surface a detection at all.
    private static let minConfidence: Float = 0.15

    private let request: VNCoreMLRequest

    init() {
        // Xcode compiles YOLOv3Tiny.mlmodel into YOLOv3Tiny.mlmodelc at build time.
        guard let modelURL = Bundle.main.url(forResource: "YOLOv3Tiny", withExtension: "mlmodelc") else {
            fatalError("YOLOv3Tiny.mlmodelc missing from app bundle")
        }
        do {
            let mlModel = try MLModel(contentsOf: modelURL)
            let visionModel = try VNCoreMLModel(for: mlModel)
            self.request = VNCoreMLRequest(model: visionModel)
            self.request.imageCropAndScaleOption = .scaleFit
        } catch {
            fatalError("Failed to load YOLOv3Tiny: \(error)")
        }
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

        guard let observations = request.results as? [VNRecognizedObjectObservation] else {
            print("[detect] no observations (results type: \(type(of: request.results)))")
            return []
        }

        if !observations.isEmpty {
            let summary = observations.prefix(5).map { o in
                "\(o.labels.first?.identifier ?? "?")@\(String(format: "%.2f", o.confidence))"
            }.joined(separator: ", ")
            print("[detect] \(observations.count) obs: \(summary)")
        } else {
            print("[detect] 0 observations from YOLO")
        }

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
