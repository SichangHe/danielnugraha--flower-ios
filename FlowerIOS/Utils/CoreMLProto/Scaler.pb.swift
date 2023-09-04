// DO NOT EDIT.
// swift-format-ignore-file
//
// Generated by the Swift generator plugin for the protocol buffer compiler.
// Source: Scaler.proto
//
// For information on using the generated types, please see the documentation:
//   https://github.com/apple/swift-protobuf/

// Copyright (c) 2017, Apple Inc. All rights reserved.
//
// Use of this source code is governed by a BSD-3-clause license that can be
// found in LICENSE.txt or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import SwiftProtobuf

// If the compiler emits an error on this type, it is because this file
// was generated by a version of the `protoc` Swift plug-in that is
// incompatible with the version of SwiftProtobuf to which you are linking.
// Please ensure that you are building against the same version of the API
// that was used to generate this file.
private struct _GeneratedWithProtocGenSwiftVersion: SwiftProtobuf.ProtobufAPIVersionCheck {
    struct _2: SwiftProtobuf.ProtobufAPIVersion_2 {}
    typealias Version = _2
}

/// *
/// A scaling operation.
///
/// This function has the following formula:
///
/// .. math::
///     f(x) = scaleValue \cdot (x + shiftValue)
///
/// If the ``scaleValue`` is not given, the default value 1 is used.
/// If the ``shiftValue`` is not given, the default value 0 is used.
///
/// If ``scaleValue`` and ``shiftValue`` are each a single value
/// and the input is an array, then the scale and shift are applied
/// to each element of the array.
///
/// If the input is an integer, then it is converted to a double to
/// perform the scaling operation. If the output type is an integer,
/// then it is cast to an integer. If that cast is lossy, then an
/// error is generated.
struct CoreML_Specification_Scaler {
    // SwiftProtobuf.Message conformance is added in an extension below. See the
    // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
    // methods supported on all messages.

    var shiftValue: [Double] = []

    var scaleValue: [Double] = []

    var unknownFields = SwiftProtobuf.UnknownStorage()

    init() {}
}

// MARK: - Code below here is support for the SwiftProtobuf runtime.

private let _protobuf_package = "CoreML.Specification"

extension CoreML_Specification_Scaler: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
    static let protoMessageName: String = _protobuf_package + ".Scaler"
    static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
        1: .same(proto: "shiftValue"),
        2: .same(proto: "scaleValue"),
    ]

    mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
        while let fieldNumber = try decoder.nextFieldNumber() {
            // The use of inline closures is to circumvent an issue where the compiler
            // allocates stack space for every case branch when no optimizations are
            // enabled. https://github.com/apple/swift-protobuf/issues/1034
            switch fieldNumber {
            case 1: try try decoder.decodeRepeatedDoubleField(value: &shiftValue)
            case 2: try try decoder.decodeRepeatedDoubleField(value: &scaleValue)
            default: break
            }
        }
    }

    func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
        if !shiftValue.isEmpty {
            try visitor.visitPackedDoubleField(value: shiftValue, fieldNumber: 1)
        }
        if !scaleValue.isEmpty {
            try visitor.visitPackedDoubleField(value: scaleValue, fieldNumber: 2)
        }
        try unknownFields.traverse(visitor: &visitor)
    }

    static func == (lhs: CoreML_Specification_Scaler, rhs: CoreML_Specification_Scaler) -> Bool {
        if lhs.shiftValue != rhs.shiftValue { return false }
        if lhs.scaleValue != rhs.scaleValue { return false }
        if lhs.unknownFields != rhs.unknownFields { return false }
        return true
    }
}
