// CameraView.swift
//
// SwiftUI wrapper around AVCaptureVideoPreviewLayer. We host a plain UIView
// whose backing CALayer is a preview layer, then bridge it via
// UIViewRepresentable so it composes with the rest of the SwiftUI tree.

import AVFoundation
import SwiftUI
import UIKit

struct CameraView: UIViewRepresentable {
    @EnvironmentObject var viewModel: ViewModel

    func makeUIView(context: Context) -> PreviewUIView {
        let view = PreviewUIView()
        view.previewLayer.session = viewModel.cameraSession
        view.previewLayer.videoGravity = .resizeAspectFill
        return view
    }

    func updateUIView(_ uiView: PreviewUIView, context: Context) {
        // No-op. The session is set once and lives for the app lifetime.
    }
}

/// UIView whose backing layer is an AVCaptureVideoPreviewLayer. Lets us avoid
/// manually managing layer geometry — UIKit's autolayout pipeline keeps it sized.
final class PreviewUIView: UIView {
    override class var layerClass: AnyClass {
        AVCaptureVideoPreviewLayer.self
    }

    var previewLayer: AVCaptureVideoPreviewLayer {
        // swiftlint:disable:next force_cast
        layer as! AVCaptureVideoPreviewLayer
    }
}
