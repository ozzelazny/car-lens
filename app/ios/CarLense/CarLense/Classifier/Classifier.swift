// Classifier.swift
//
// Coordinates the full classify-one-bbox pipeline:
//
//   pixelBuffer + bbox
//     -> Preprocessor (crop, resize, normalize) -> MLMultiArray [1,3,224,224]
//     -> MLModel "input"  ->  "var_685" [1, 512] L2-normalized embedding
//     -> PrototypeBank.topK (SGEMV against 6423×512 FP32 prototypes)
//     -> softmax with temperature
//     -> [Prediction]
//
// Core ML I/O contract (verified against dist/ios/model.mlpackage/Data/.../model.mlmodel):
//   input  name = "input"    shape = [1, 3, 224, 224]  dtype = float32
//   output name = "var_685"  shape = [1, 512]          dtype = float32 (L2-normalized)
//
// `var_685` is coremltools' auto-assigned name for the final tensor since the
// engine's export-mobile path doesn't rename it. If a future engine rebuild
// renames the output, just update OUTPUT_NAME below.

import CoreML
import CoreVideo
import Foundation

/// Tunable knobs that affect classification behavior.
struct ClassifierConfig {
    /// Softmax temperature. Higher = sharper top-1. With L2-normalized cosine
    /// scores in [-1, 1], a temperature of ~10 puts top-1 confidence in a
    /// readable 30-95% band on real photos. Tune after on-device telemetry.
    var softmaxTemperature: Float = 10.0

    /// How many predictions to return per classify() call.
    var topK: Int = 3
}

enum ClassifierError: Error {
    case modelMissing
    case unexpectedOutput(String)
}

final class Classifier {
    /// Core ML output tensor name. See file header comment.
    private static let OUTPUT_NAME = "var_685"
    private static let INPUT_NAME  = "input"

    let preprocessor: Preprocessor
    let bank: PrototypeBank
    let catalog: ClassCatalog
    var config: ClassifierConfig

    private let model: MLModel

    init(preprocessor: Preprocessor,
         bank: PrototypeBank,
         catalog: ClassCatalog,
         config: ClassifierConfig = .init()) throws {
        self.preprocessor = preprocessor
        self.bank = bank
        self.catalog = catalog
        self.config = config

        // Locate the compiled model. Xcode compiles `model.mlpackage` into
        // `model.mlmodelc` (a directory bundle) at build time and ships it
        // inside the app. Bundle.main.url(forResource:withExtension:) returns
        // the compiled .mlmodelc directory.
        guard let url = Bundle.main.url(forResource: "model", withExtension: "mlmodelc") else {
            throw ClassifierError.modelMissing
        }
        let mlConfig = MLModelConfiguration()
        // Let Core ML pick whichever of CPU/GPU/Neural Engine is fastest for
        // each op. On iPhone 13+ the NE handles ~all conv layers.
        mlConfig.computeUnits = .all
        self.model = try MLModel(contentsOf: url, configuration: mlConfig)
    }

    /// Top-level convenience: load everything from the bundle. Throws on any
    /// missing asset so the app fails fast with a useful error in the UI.
    static func loadFromBundle() throws -> Classifier {
        let prep = try Preprocessor.loadFromBundle()
        // 512 is fixed by preprocessing.json's `embedding_dim`. If a future
        // export bumps this we'd plumb it through; for the MVP we hard-code
        // it (the engine's prototype writer asserts the same number).
        let bank = try PrototypeBank.loadFromBundle(embeddingDim: 512)
        let catalog = try ClassCatalog.loadFromBundle()
        if catalog.count != bank.numClasses {
            // Don't crash — but the prototypes and class names came out of
            // step. Most likely cause: someone refreshed prototypes.bin but
            // forgot class_names.json (or vice versa). Surface a clear error.
            throw ClassifierError.unexpectedOutput(
                "class count mismatch: catalog=\(catalog.count) prototypes=\(bank.numClasses)"
            )
        }
        return try Classifier(preprocessor: prep, bank: bank, catalog: catalog)
    }

    /// Run the full pipeline. Returns the top-K predictions by softmax probability.
    /// `bbox` is in Vision-space normalized coords (origin bottom-left, y-up).
    func classify(pixelBuffer: CVPixelBuffer, bbox: CGRect?) throws -> [Prediction] {
        let array = try preprocessor.preprocess(pixelBuffer: pixelBuffer, bbox: bbox)

        let provider = try MLDictionaryFeatureProvider(dictionary: [Self.INPUT_NAME: array])
        let result = try model.prediction(from: provider)

        guard let featureValue = result.featureValue(for: Self.OUTPUT_NAME),
              let outArray = featureValue.multiArrayValue
        else {
            throw ClassifierError.unexpectedOutput(
                "expected output named '\(Self.OUTPUT_NAME)'; available: \(result.featureNames)"
            )
        }

        let embedding = Self.copyFloat32(from: outArray)

        // Cosine similarity == dot product since the engine L2-normalizes both
        // the model output and the prototype rows.
        let topScores = bank.topK(embedding: embedding, k: config.topK)

        // Soft-max with temperature over the full score vector would be the
        // "correct" thing, but we only need confidences for the top-K results
        // — and we want top-1 confidence to be high when one prototype is the
        // clear winner. A local soft-max over just the top-K achieves that
        // and is dramatically cheaper than over all 6,423. The interpretation
        // is "given that the right answer is in this top-K, here's how
        // confident we are in each rank" — fine for an overlay label.
        let probs = Self.softmaxOverScores(topScores.map { $0.score },
                                           temperature: config.softmaxTemperature)

        var out: [Prediction] = []
        out.reserveCapacity(topScores.count)
        for (i, item) in topScores.enumerated() {
            guard let entry = catalog[item.classIdx] else { continue }
            out.append(Prediction(classId: entry.classId,
                                  displayName: entry.displayName,
                                  confidence: probs[i]))
        }
        return out
    }

    // MARK: - helpers

    /// Copy an MLMultiArray of shape [1, 512] (or [512]) into a contiguous [Float].
    /// Defends against non-contiguous strides by walking the array via subscript.
    private static func copyFloat32(from array: MLMultiArray) -> [Float] {
        let count = array.count
        if array.dataType == .float32, array.strides.allSatisfy({ $0.intValue > 0 }),
           array.strides[array.strides.count - 1].intValue == 1 {
            // Fast path: dense float32 with unit innermost stride.
            let ptr = array.dataPointer.bindMemory(to: Float32.self, capacity: count)
            return Array(UnsafeBufferPointer(start: ptr, count: count))
        }
        // Slow but correct path.
        var out = [Float](repeating: 0, count: count)
        for i in 0..<count {
            out[i] = Float(truncating: array[i])
        }
        return out
    }

    /// Numerically-stable softmax with temperature over a small score vector.
    private static func softmaxOverScores(_ scores: [Float], temperature: Float) -> [Float] {
        guard !scores.isEmpty else { return [] }
        let scaled = scores.map { $0 * temperature }
        let m = scaled.max() ?? 0
        let exps = scaled.map { expf($0 - m) }
        let sum = exps.reduce(0, +)
        guard sum > 0 else {
            return Array(repeating: 1.0 / Float(scores.count), count: scores.count)
        }
        return exps.map { $0 / sum }
    }
}
