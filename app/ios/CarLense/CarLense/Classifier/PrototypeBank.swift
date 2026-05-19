// PrototypeBank.swift
//
// Loads `prototypes.bin` — raw little-endian FP16 bytes, shape [num_classes, 512].
// num_classes is currently 6,423 but we determine it from the byte count to
// stay robust across model rebuilds.
//
// Storage policy: we keep prototypes as FP32 in memory for fast SGEMV against
// the FP32 embedding. The FP16→FP32 upcast happens once at startup. 6,423 ×
// 512 × 4 = ~13 MB — well within budget.
//
// Similarity: dot product (both sides L2-normalized per preprocessing.json).
// We use Accelerate's `cblas_sgemv` for the 6423×512 · 512 matrix-vector
// multiply: typical ~0.5-1.5 ms on iPhone 13+ Neural Engine class CPUs.

import Accelerate
import Foundation

enum PrototypeBankError: Error {
    case fileMissing
    case invalidSize(Int)
}

final class PrototypeBank {
    /// Flat FP32 row-major buffer, length = numClasses * embeddingDim.
    /// Stored as a contiguous Swift array so Accelerate can use the underlying
    /// pointer directly without copies.
    private let prototypesFP32: [Float]

    /// Number of class prototypes, e.g. 6423.
    let numClasses: Int

    /// Embedding dimensionality, e.g. 512.
    let embeddingDim: Int

    init(prototypesFP32: [Float], numClasses: Int, embeddingDim: Int) {
        self.prototypesFP32 = prototypesFP32
        self.numClasses = numClasses
        self.embeddingDim = embeddingDim
    }

    /// Loads `prototypes.bin` from the app bundle. The embedding dim is taken
    /// from the spec (typically 512) and the class count is derived from
    /// file size / (dim * 2 bytes per FP16).
    static func loadFromBundle(embeddingDim: Int = 512) throws -> PrototypeBank {
        guard let url = Bundle.main.url(forResource: "prototypes", withExtension: "bin") else {
            throw PrototypeBankError.fileMissing
        }
        let data = try Data(contentsOf: url)
        let halfCount = data.count / MemoryLayout<UInt16>.size
        guard halfCount > 0, halfCount % embeddingDim == 0 else {
            throw PrototypeBankError.invalidSize(data.count)
        }
        let numClasses = halfCount / embeddingDim

        // Reinterpret bytes as Float16 and upcast to Float32 in one pass with
        // Accelerate's vImageConvert. Keeps us off the slow scalar path.
        var fp32 = [Float](repeating: 0, count: halfCount)

        data.withUnsafeBytes { (rawBuf: UnsafeRawBufferPointer) in
            guard let src = rawBuf.baseAddress else { return }
            let srcF16 = src.bindMemory(to: UInt16.self, capacity: halfCount)

            var srcBuf = vImage_Buffer(
                data: UnsafeMutableRawPointer(mutating: srcF16),
                height: 1,
                width: vImagePixelCount(halfCount),
                rowBytes: halfCount * MemoryLayout<UInt16>.size
            )
            fp32.withUnsafeMutableBufferPointer { dstPtr in
                var dstBuf = vImage_Buffer(
                    data: dstPtr.baseAddress,
                    height: 1,
                    width: vImagePixelCount(halfCount),
                    rowBytes: halfCount * MemoryLayout<Float>.size
                )
                // Planar16F -> PlanarF (single-channel float16 to float32).
                _ = vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, 0)
            }
        }

        return PrototypeBank(prototypesFP32: fp32,
                             numClasses: numClasses,
                             embeddingDim: embeddingDim)
    }

    /// Top-K classes by cosine similarity to `embedding`.
    /// Assumes `embedding` is L2-normalized (the export-mobile model does this
    /// inside its forward — see export.mobile._WrappedNormalized). Likewise
    /// for the prototypes (the engine normalizes before writing prototypes.bin).
    /// Therefore cosine sim == dot product.
    ///
    /// - Parameter embedding: FP32 vector of length `embeddingDim`.
    /// - Parameter k: number of top results to return.
    /// - Returns: (classIdx, score) sorted by descending score.
    func topK(embedding: [Float], k: Int) -> [(classIdx: Int, score: Float)] {
        precondition(embedding.count == embeddingDim,
                     "embedding length \(embedding.count) != \(embeddingDim)")
        var scores = [Float](repeating: 0, count: numClasses)

        // scores = prototypes @ embedding
        //   prototypes: [numClasses, embeddingDim] row-major
        //   embedding:  [embeddingDim]
        //   result:     [numClasses]
        prototypesFP32.withUnsafeBufferPointer { protoPtr in
            embedding.withUnsafeBufferPointer { embPtr in
                scores.withUnsafeMutableBufferPointer { scorePtr in
                    cblas_sgemv(
                        CblasRowMajor,
                        CblasNoTrans,
                        Int32(numClasses),
                        Int32(embeddingDim),
                        1.0,
                        protoPtr.baseAddress!,
                        Int32(embeddingDim),
                        embPtr.baseAddress!,
                        1,
                        0.0,
                        scorePtr.baseAddress!,
                        1
                    )
                }
            }
        }

        // Partial top-K via an O(n log k) heap-style pass. For k=3..10 over
        // n=6423 a full sort is fine (microseconds), so we just sort.
        let indices = (0..<numClasses).sorted { scores[$0] > scores[$1] }
        let limit = min(k, numClasses)
        var out: [(classIdx: Int, score: Float)] = []
        out.reserveCapacity(limit)
        for i in 0..<limit {
            let idx = indices[i]
            out.append((classIdx: idx, score: scores[idx]))
        }
        return out
    }
}
