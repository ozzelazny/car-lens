// Prediction.swift
//
// A single classifier hypothesis for a detected car. The classifier returns
// top-K; the UI currently shows only top-1 but we keep the full list for
// debugging and future "alternates" UI.

import Foundation

struct Prediction: Hashable {
    /// Raw class id from class_names.json `class_ids`, e.g. "2014|honda|civic".
    let classId: String

    /// Pretty display string from class_names.json `display_names`,
    /// e.g. "2014-2017 Honda Civic".
    let displayName: String

    /// Softmax probability over the 6,423-class prototype bank, 0..1.
    let confidence: Float
}
