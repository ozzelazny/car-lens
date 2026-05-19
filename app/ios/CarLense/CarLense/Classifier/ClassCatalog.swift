// ClassCatalog.swift
//
// Loads `class_names.json` shipped in the app bundle.
//
// Schema (matches dist/ios/class_names.json):
//   {
//     "class_ids":     ["1900|acura|integra", ...],     // canonical key
//     "display_names": ["1900-1903 Acura Integra", ...] // pretty label for UI
//   }
// Both arrays are the same length and ordered to match the prototypes.bin rows.

import Foundation

enum ClassCatalogError: Error {
    case fileMissing
    case malformed(String)
}

struct ClassCatalog {
    /// Raw class id at row i in prototypes.bin (e.g. "2014|honda|civic").
    let classIds: [String]

    /// Pretty display name at row i (e.g. "2014-2017 Honda Civic").
    let displayNames: [String]

    var count: Int { classIds.count }

    static func loadFromBundle() throws -> ClassCatalog {
        guard let url = Bundle.main.url(forResource: "class_names", withExtension: "json") else {
            throw ClassCatalogError.fileMissing
        }
        let data = try Data(contentsOf: url)
        struct Wire: Decodable {
            let class_ids: [String]
            let display_names: [String]
        }
        let wire = try JSONDecoder().decode(Wire.self, from: data)
        guard wire.class_ids.count == wire.display_names.count else {
            throw ClassCatalogError.malformed(
                "class_ids count \(wire.class_ids.count) != display_names count \(wire.display_names.count)"
            )
        }
        return ClassCatalog(classIds: wire.class_ids, displayNames: wire.display_names)
    }

    subscript(index: Int) -> (classId: String, displayName: String)? {
        guard index >= 0, index < classIds.count else { return nil }
        return (classIds[index], displayNames[index])
    }
}
