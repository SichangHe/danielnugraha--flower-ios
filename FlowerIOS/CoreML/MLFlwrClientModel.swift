//
//  MLFlwrClientModel.swift
//
//
//  Created by Daniel Nugraha on 20.01.23.
//

import CoreML
import Foundation
import os

/// Container for train and test dataset.
public struct MLDataLoader {
    public let trainBatchProvider: MLBatchProvider
    public let testBatchProvider: MLBatchProvider

    public init(trainBatchProvider: MLBatchProvider, testBatchProvider: MLBatchProvider) {
        self.trainBatchProvider = trainBatchProvider
        self.testBatchProvider = testBatchProvider
    }
}

/// Container for neural network layer information.
public struct MLLayerWrapper {
    let shape: [Int16]
    let name: String
    var weights: [Float]
    let isUpdatable: Bool

    public init(shape: [Int16], name: String, weights: [Float], isUpdatable: Bool) {
        self.shape = shape
        self.name = name
        self.weights = weights
        self.isUpdatable = isUpdatable
    }
}

struct MLResult {
    let loss: Double
    let numSamples: Int
    let accuracy: Double
}

/// A class responsible for loading and retrieving model parameters to and from the CoreML model.
@available(iOS 14.0, *)
public class MLParameter {
    private var parameterConverter = ParameterConverter.shared

    var layerWrappers: [MLLayerWrapper]
    private let log = Logger(subsystem: Bundle.main.bundleIdentifier ?? "flwr.Flower",
                             category: String(describing: MLParameter.self))

    /// Inits MLParameter class that contains information about the model parameters and implements routines for their update and transformation.
    ///
    /// - Parameters:
    ///   - layerWrappers: Information about the layer provided with primitive data types.
    public init(layerWrappers: [MLLayerWrapper]) {
        self.layerWrappers = layerWrappers
    }

    /// Converts the Parameters structure to MLModelConfiguration to interface with CoreML.
    ///
    /// - Parameters:
    ///   - parameters: The parameters of the model passed as Parameters struct.
    /// - Returns: Specification of the machine learning model configuration in the CoreML structure.
    public func parametersToWeights(parameters: Parameters) -> MLModelConfiguration {
        let config = MLModelConfiguration()
        if config.parameters == nil {
            config.parameters = [:]
        }

        guard parameters.tensors.count == layerWrappers.count else {
            log.info("parameters received is not valid")
            return config
        }

        for (index, data) in parameters.tensors.enumerated() {
            let expectedNumberOfElements = layerWrappers[index].shape.map { Int($0) }.reduce(1, *)
            let weightsArray = data.toArray(type: Float.self)
            guard weightsArray.count == expectedNumberOfElements else {
                log.info("array received has wrong number of elements")
                continue
            }

            layerWrappers[index].weights = weightsArray
            if layerWrappers[index].isUpdatable {
                let name = layerWrappers[index].name
                let shapedArray = MLShapedArray(scalars: weightsArray, shape: layerWrappers[index].shape.map { Int($0) })
                let layerParams = MLMultiArray(shapedArray)
                log.error("MLClient: layerParams for \(name) shape: \(layerParams.shape) count: \(layerParams.count) is float: \(layerParams.dataType == .float).")
                let weightsShape = layerParams.shape.map { Int16(truncating: $0) }
                guard weightsShape == layerWrappers[index].shape else {
                    log.info("shape not the same")
                    continue
                }
                let paramKey = MLParameterKey.weights.scoped(to: name)
                config.parameters![paramKey] = layerParams
            }
        }

        return config
    }

    /// Returns the weights of the current layer wrapper in parameter format
    ///
    /// - Returns: The weights of the current layer wrapper in parameter format
    public func weightsToParameters() -> Parameters {
        let dataArray = layerWrappers.map { Data(fromArray: $0.weights) }
        return Parameters(tensors: dataArray, tensorType: "ndarray")
    }

    /// Updates the layers given the CoreML update context
    ///
    /// - Parameters:
    ///   - context: The context of the update procedure of the CoreML model.
    public func updateLayerWrappers(context: MLUpdateContext) {
        for (index, layer) in layerWrappers.enumerated() {
            if layer.isUpdatable {
                let paramKey = MLParameterKey.weights.scoped(to: layer.name)
                if let weightsMultiArray = try? context.model.parameterValue(for: paramKey) as? MLMultiArray {
                    let weightsShape = Array(weightsMultiArray.shape.map { Int16(truncating: $0) }.drop(while: { $0 < 2 }))
                    guard weightsShape == layer.shape else {
                        log.info("shape \(weightsShape) is not the same as \(layer.shape)")
                        continue
                    }

                    if let pointer = try? UnsafeBufferPointer<Float>(weightsMultiArray) {
                        let array = pointer.compactMap { $0 }
                        layerWrappers[index].weights = array
                    }
                }
            }
        }
    }
}
