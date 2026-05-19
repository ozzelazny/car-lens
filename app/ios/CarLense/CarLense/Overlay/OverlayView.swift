// OverlayView.swift
//
// Draws one labeled bounding box per detected vehicle. Pulls detection+prediction
// state from the ViewModel and lays them out in SwiftUI screen coordinates.
//
// Coordinate conversion: Vision returns normalized bboxes with origin
// bottom-left and y-up (0..1 in each axis). SwiftUI uses origin top-left,
// y-down. We flip y at draw time.
//
// Note on aspect: the camera preview is .resizeAspectFill, so the on-screen
// preview crops the camera buffer to fit. For an exact overlay we'd compute
// the visible-rect transform too. The MVP draws into the full overlay area —
// boxes may be slightly off near screen edges when the preview is cropped.
// This is a known v1.1 follow-up.

import SwiftUI

struct OverlayView: View {
    @EnvironmentObject var viewModel: ViewModel

    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .topLeading) {
                ForEach(Array(viewModel.detections.enumerated()), id: \.offset) { _, entry in
                    let det = entry.detection
                    let pred = entry.prediction
                    let rect = Self.screenRect(forNormalized: det.bbox, in: geo.size)
                    let color = Self.color(for: pred?.confidence)

                    ZStack(alignment: .topLeading) {
                        Rectangle()
                            .stroke(color, lineWidth: 2)
                            .frame(width: rect.width, height: rect.height)
                            .position(x: rect.midX, y: rect.midY)

                        labelView(pred: pred, color: color, det: det)
                            .position(x: rect.minX + 4, y: max(12, rect.minY - 12))
                            .frame(maxWidth: rect.width, alignment: .leading)
                    }
                }
            }
            .frame(width: geo.size.width, height: geo.size.height)
            .allowsHitTesting(false)
        }
    }

    private func labelView(pred: Prediction?, color: Color, det: Detection) -> some View {
        let text: String
        if let p = pred {
            text = String(format: "%@  %.0f%%", p.displayName, p.confidence * 100)
        } else {
            text = "…"  // classifier hasn't returned yet for this slot
        }
        return Text(text)
            .font(.caption2.monospacedDigit())
            .foregroundColor(.black)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(color)
            .clipShape(RoundedRectangle(cornerRadius: 4))
            .lineLimit(1)
    }

    // MARK: - geometry

    /// Vision normalized (bottom-left origin, y-up) → SwiftUI screen (top-left, y-down).
    private static func screenRect(forNormalized bbox: CGRect, in size: CGSize) -> CGRect {
        let x = bbox.minX * size.width
        let y = (1 - bbox.maxY) * size.height
        let w = bbox.width  * size.width
        let h = bbox.height * size.height
        return CGRect(x: x, y: y, width: w, height: h)
    }

    private static func color(for confidence: Float?) -> Color {
        guard let c = confidence else { return .gray.opacity(0.8) }
        if c >= 0.70 { return .yellow }
        if c >= 0.40 { return .orange }
        return .red
    }
}
