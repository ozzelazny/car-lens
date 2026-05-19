// CarLenseApp.swift
//
// SwiftUI @main entry point. Owns a single ViewModel instance for the whole
// app lifetime so the camera session, Core ML model, and prototype bank all
// load exactly once and are shared with every view that needs them.
//
// We force portrait orientation in Info.plist (UISupportedInterfaceOrientations)
// because the AVCaptureVideoPreviewLayer and Vision bounding-box coordinate
// transforms in OverlayView assume portrait. Re-supporting landscape later
// is a v1.1 task that needs explicit affine transforms in OverlayView.

import SwiftUI

@main
struct CarLenseApp: App {
    @StateObject private var viewModel = ViewModel()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(viewModel)
                .statusBarHidden(true)
                .preferredColorScheme(.dark)
        }
    }
}
