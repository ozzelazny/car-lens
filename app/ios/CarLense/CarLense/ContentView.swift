// ContentView.swift
//
// Root view. Stacks the camera preview behind an overlay that draws bounding
// boxes and classification labels. The ViewModel publishes the current frame's
// detections + predictions; both subviews observe it.

import SwiftUI

struct ContentView: View {
    @EnvironmentObject var viewModel: ViewModel

    var body: some View {
        ZStack {
            CameraView()
                .ignoresSafeArea()

            OverlayView()
                .ignoresSafeArea()

            VStack {
                Spacer()
                statusBar
                    .padding(.bottom, 24)
            }
        }
        .onAppear {
            viewModel.start()
        }
        .onDisappear {
            viewModel.stop()
        }
    }

    /// Small bottom strip showing FPS and the last classification latency.
    /// Helps validate the on-device perf target (≤30ms classifier, 15-30fps detect).
    private var statusBar: some View {
        HStack(spacing: 16) {
            Text(String(format: "%.0f fps", viewModel.fps))
            Text(String(format: "cls %.0fms", viewModel.lastClassifyMs))
            if let err = viewModel.errorMessage {
                Text(err)
                    .foregroundColor(.red)
                    .lineLimit(2)
            }
        }
        .font(.caption.monospacedDigit())
        .foregroundColor(.white)
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(.black.opacity(0.5), in: Capsule())
    }
}
