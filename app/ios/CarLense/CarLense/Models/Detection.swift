// Detection.swift
//
// One detected vehicle in the current frame. Coordinates are in Vision's
// "normalized image space" (0..1, origin in the bottom-left, y-up). We keep
// them in this space all the way to OverlayView, which does the single
// conversion to SwiftUI screen coords.

import CoreGraphics
import Foundation

struct Detection: Identifiable, Hashable {
    /// Stable identity for SwiftUI ForEach within a single frame. We re-use the
    /// detection's index in the frame's output as the id since we don't run
    /// a tracker in MVP — IDs reset every frame and that's fine.
    let id: Int

    /// Vision-space normalized bbox (origin bottom-left, y-up, 0..1).
    let bbox: CGRect

    /// Detector confidence reported by Vision, 0..1.
    let confidence: Float

    /// Bbox area × confidence. Used to pick the "most prominent" 3 cars per frame.
    var score: Float {
        Float(bbox.width * bbox.height) * confidence
    }

    /// Coarse class string from Vision ("car", "truck", "bus", "van", "SUV",
    /// "motorcycle"). Useful for debugging; not surfaced to the user.
    let coarseLabel: String
}
