# Car Lense — iOS MVP

A live-camera car recognizer. Point the phone at any vehicle, get a bounding box
plus a predicted year/make/model label. Everything runs on-device — no network
calls, no telemetry, no accounts.

This is the **MVP**. It proves the loop works end-to-end: camera frames →
Vision vehicle detector → Core ML MobileCLIP-B embedder → 6,423-class prototype
retrieval → overlay. No tracker, no smoothing, no settings, no tap-to-promote.
Those land in v1.1 once the loop is stable.

---

## Prerequisites

- **macOS 14+** with **Xcode 15+** (free from the Mac App Store)
- **Apple ID** — a free one is fine; no paid Developer Program needed
- **iPhone running iOS 17+**
- A **USB-C / Lightning cable** to connect the phone to the Mac
- **Homebrew** for installing XcodeGen

---

## First-time setup

```bash
# One-time, on your Mac
brew install xcodegen

# From inside the iOS app folder
cd app/ios/CarLense

# Materialize the .xcodeproj. Re-run this any time you change project.yml.
xcodegen generate
```

This generates `CarLense.xcodeproj` from `project.yml`. We deliberately do NOT
commit the `.xcodeproj` — XcodeGen rebuilds it deterministically from YAML, and
that avoids the merge-conflict hell of hand-maintained pbxproj files.

---

## Bundle the model assets

The Core ML model, prototypes, class names, and preprocessing spec live in the
engine repo's `dist/ios/` and are **gitignored** (they're large and rebuildable).
Copy them into the app's `Resources/` directory before building:

```bash
# From the repo root
cp -R dist/ios/model.mlpackage      app/ios/CarLense/CarLense/Resources/
cp    dist/ios/prototypes.bin       app/ios/CarLense/CarLense/Resources/
cp    dist/ios/class_names.json     app/ios/CarLense/CarLense/Resources/
cp    dist/ios/preprocessing.json   app/ios/CarLense/CarLense/Resources/
```

You can re-run this any time the engine ships a new bundle.

If you ever regenerate the engine's `dist/ios/` and the **output tensor name
changes** (it's currently `var_685`, coremltools' auto-assigned name), update
the constant in `CarLense/Classifier/Classifier.swift`:

```swift
private static let OUTPUT_NAME = "var_685"
```

Verify the name by opening `model.mlpackage` in Xcode — it shows inputs/outputs.

---

## Signing

Open `CarLense.xcodeproj` in Xcode, then:

1. Select the **CarLense** target in the project navigator.
2. **Signing & Capabilities** tab.
3. Tick **Automatically manage signing**.
4. **Team** → pick your personal Apple ID (the dropdown calls it `<you> (Personal Team)`).
5. **Bundle Identifier**: the default is `com.ashzelazny.carlense`. Change it
   to something unique to you (e.g. `com.<your-name>.carlense`) — Apple's free
   tier rejects bundle IDs that have been claimed by anyone else.

Xcode will provision a 7-day development certificate. Re-plug the phone and
re-run after 7 days to refresh; no other ceremony.

---

## Install on iPhone

1. Plug the iPhone into the Mac via USB.
2. In Xcode's destination dropdown, pick your iPhone (not a simulator — the
   simulator can't access the real camera and Core ML runs there in CPU-only
   mode, which is slow).
3. Hit **⌘R** to build and run.
4. The first time: **Settings → General → VPN & Device Management → Trust your
   developer cert**. Then go back to the home screen and tap the **CarLense**
   icon.

---

## What to expect on first run

- The app asks for camera permission → tap **Allow**.
- The full-screen camera preview appears.
- Point the phone at a car. After ~200 ms (model load), a yellow / orange / red
  bounding box overlays each visible car. About once a second, the label
  underneath each box updates with a predicted year/make/model and a
  confidence percentage.
- Top-of-screen status strip shows current FPS and last classifier latency.

Color coding:

| Confidence | Box color |
|------------|-----------|
| ≥ 70%      | Yellow    |
| 40-70%     | Orange    |
| < 40%      | Red       |

---

## Known limitations

- **Battery drain**: ~15-20%/hour at sustained 30fps detection. Don't leave it
  running all day on a single charge.
- **Flicker on mid-range phones**: the label may swap between two close trims
  (e.g. Civic vs Civic Coupe). Flagship iPhones (13 Pro and newer) hold steady.
  A v1.1 tracker + temporal smoother fixes this.
- **Confidently-wrong labels on rare cars**: the model is ~81% top-1 on common
  cars; rarer trims may show a confidently-wrong label. That's the model
  speaking, not a UI bug. Re-rank with cloud LLM is on the roadmap.
- **Bounding box accuracy near screen edges**: the camera preview is
  `.resizeAspectFill`, so when the camera buffer is cropped to fit, boxes
  near the edges may drift a few pixels. We use the full camera buffer for
  detection regardless; only the visual overlay is approximate.
- **Portrait only**: rotating to landscape will misalign overlays. The Info.plist
  pins to portrait to keep this from happening accidentally.

---

## Project layout

```
app/ios/CarLense/
├── project.yml                         # XcodeGen spec (source of truth for the .xcodeproj)
├── README.md                           # ← you are here
└── CarLense/
    ├── CarLenseApp.swift               # @main SwiftUI entry point
    ├── ContentView.swift               # Root view (camera + overlay + status)
    ├── Camera/
    │   ├── CameraController.swift      # AVCaptureSession wrapper
    │   └── CameraView.swift            # UIViewRepresentable preview layer
    ├── Vision/
    │   └── VehicleDetector.swift       # VNRecognizeObjectsRequest, filtered to vehicles
    ├── Classifier/
    │   ├── Classifier.swift            # Preprocess → Core ML → top-K orchestrator
    │   ├── Preprocessing.swift         # CIImage crop+resize+normalize → MLMultiArray
    │   ├── PrototypeBank.swift         # FP16 → FP32 load, SGEMV top-K
    │   └── ClassCatalog.swift          # class_names.json loader
    ├── Models/
    │   ├── Detection.swift             # One detected car (bbox + score)
    │   └── Prediction.swift            # One classifier hypothesis
    ├── Overlay/
    │   └── OverlayView.swift           # SwiftUI bounding-box + label renderer
    ├── State/
    │   └── ViewModel.swift             # @MainActor pipeline orchestrator
    ├── Info.plist                      # Camera usage description, portrait lock, etc.
    └── Resources/                      # ← drop model.mlpackage / prototypes.bin / etc. here
```

---

## Re-running after engine updates

Whenever the engine ships a new `dist/ios/` bundle:

```bash
# From repo root
cp -R dist/ios/model.mlpackage      app/ios/CarLense/CarLense/Resources/
cp    dist/ios/prototypes.bin       app/ios/CarLense/CarLense/Resources/
cp    dist/ios/class_names.json     app/ios/CarLense/CarLense/Resources/
cp    dist/ios/preprocessing.json   app/ios/CarLense/CarLense/Resources/

# Then in Xcode: Product → Clean Build Folder (⇧⌘K) then ⌘R
```

Clean Build is important — Xcode caches compiled `.mlmodelc` outputs and won't
re-compile an updated `.mlpackage` without it.
