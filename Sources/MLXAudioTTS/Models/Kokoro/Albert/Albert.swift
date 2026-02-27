//
//  Albert.swift
//  MLXAudio
//
//  Ported from mlx-audio Kokoro implementation
//

import Foundation
import MLX
import MLXNN

// MARK: - AlbertEmbeddings

class AlbertEmbeddings {
    let wordEmbeddings: Embedding
    let positionEmbeddings: Embedding
    let tokenTypeEmbeddings: Embedding
    let layerNorm: LayerNorm

    init(weights: [String: MLXArray], config: AlbertModelArgs) {
        wordEmbeddings = Embedding(weight: weights["bert.embeddings.word_embeddings.weight"]!)
        positionEmbeddings = Embedding(weight: weights["bert.embeddings.position_embeddings.weight"]!)
        tokenTypeEmbeddings = Embedding(weight: weights["bert.embeddings.token_type_embeddings.weight"]!)
        layerNorm = LayerNorm(dimensions: config.embeddingSize, eps: config.layerNormEps)
        let layerNormWeights = weights["bert.embeddings.LayerNorm.weight"]!
        let layerNormBiases = weights["bert.embeddings.LayerNorm.bias"]!

        guard layerNormBiases.count == config.embeddingSize, layerNormWeights.count == config.embeddingSize else {
            fatalError("Wrong shape for AlbertEmbeddings LayerNorm bias or weights!")
        }

        for i in 0..<layerNormBiases.shape[0] {
            layerNorm.bias![i] = layerNormBiases[i]
            layerNorm.weight![i] = layerNormWeights[i]
        }
    }

    func callAsFunction(
        _ inputIds: MLXArray,
        tokenTypeIds: MLXArray? = nil,
        positionIds: MLXArray? = nil
    ) -> MLXArray {
        let seqLength = inputIds.shape[1]

        let positionIdsUsed: MLXArray
        if let positionIds = positionIds {
            positionIdsUsed = positionIds
        } else {
            positionIdsUsed = MLX.expandedDimensions(MLXArray(0..<seqLength), axes: [0])
        }

        let tokenTypeIdsUsed: MLXArray
        if let tokenTypeIds = tokenTypeIds {
            tokenTypeIdsUsed = tokenTypeIds
        } else {
            tokenTypeIdsUsed = MLXArray.zeros(like: inputIds)
        }

        let wordsEmbeddings = wordEmbeddings(inputIds)
        let positionEmbeddingsResult = positionEmbeddings(positionIdsUsed)
        let tokenTypeEmbeddingsResult = tokenTypeEmbeddings(tokenTypeIdsUsed)
        var embeddings = wordsEmbeddings + positionEmbeddingsResult + tokenTypeEmbeddingsResult
        embeddings = layerNorm(embeddings)
        return embeddings
    }
}

// MARK: - AlbertSelfAttention

class AlbertSelfAttention {
    let numAttentionHeads: Int
    let attentionHeadSize: Int
    let allHeadSize: Int

    let query: Linear
    let key: Linear
    let value: Linear
    let dense: Linear
    let layerNorm: LayerNorm

    init(weights: [String: MLXArray], config: AlbertModelArgs, layerNum: Int, innerGroupNum: Int) {
        numAttentionHeads = config.numAttentionHeads
        attentionHeadSize = config.hiddenSize / config.numAttentionHeads
        allHeadSize = numAttentionHeads * attentionHeadSize

        query = Linear(
            weight: weights["bert.encoder.albert_layer_groups.\(layerNum).albert_layers.\(innerGroupNum).attention.query.weight"]!,
            bias: weights["bert.encoder.albert_layer_groups.\(layerNum).albert_layers.\(innerGroupNum).attention.query.bias"]!
        )
        key = Linear(
            weight: weights["bert.encoder.albert_layer_groups.\(layerNum).albert_layers.\(innerGroupNum).attention.key.weight"]!,
            bias: weights["bert.encoder.albert_layer_groups.\(layerNum).albert_layers.\(innerGroupNum).attention.key.bias"]!
        )
        value = Linear(
            weight: weights["bert.encoder.albert_layer_groups.\(layerNum).albert_layers.\(innerGroupNum).attention.value.weight"]!,
            bias: weights["bert.encoder.albert_layer_groups.\(layerNum).albert_layers.\(innerGroupNum).attention.value.bias"]
        )
        dense = Linear(
            weight: weights["bert.encoder.albert_layer_groups.\(layerNum).albert_layers.\(innerGroupNum).attention.dense.weight"]!,
            bias: weights["bert.encoder.albert_layer_groups.\(layerNum).albert_layers.\(innerGroupNum).attention.dense.bias"]!
        )

        layerNorm = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)

        let layerNormWeights = weights["bert.encoder.albert_layer_groups.\(layerNum).albert_layers.\(innerGroupNum).attention.LayerNorm.weight"]!
        let layerNormBiases = weights["bert.encoder.albert_layer_groups.\(layerNum).albert_layers.\(innerGroupNum).attention.LayerNorm.bias"]!

        guard layerNormWeights.count == config.hiddenSize, layerNormBiases.count == config.hiddenSize else {
            fatalError("Wrong shape for AlbertSelfAttention LayerNorm bias or weights!")
        }

        for i in 0..<layerNormBiases.shape[0] {
            layerNorm.bias![i] = layerNormBiases[i]
            layerNorm.weight![i] = layerNormWeights[i]
        }
    }

    func transposeForScores(_ x: MLXArray) -> MLXArray {
        let shape = x.shape
        var newShape: [Int] = []

        for i in 0..<(shape.count - 1) {
            newShape.append(shape[i])
        }

        newShape.append(numAttentionHeads)
        newShape.append(attentionHeadSize)

        let reshaped = x.reshaped(newShape)
        return reshaped.transposed(0, 2, 1, 3)
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        attentionMask: MLXArray? = nil
    ) -> MLXArray {
        let mixedQueryLayer = query(hiddenStates)
        let mixedKeyLayer = key(hiddenStates)
        let mixedValueLayer = value(hiddenStates)

        let queryLayer = transposeForScores(mixedQueryLayer)
        let keyLayer = transposeForScores(mixedKeyLayer)
        let valueLayer = transposeForScores(mixedValueLayer)

        let keyLayerTransposed = keyLayer.transposed(0, 1, 3, 2)
        var attentionScores = MLX.matmul(queryLayer, keyLayerTransposed)
        attentionScores = attentionScores / sqrt(Float(attentionHeadSize))

        if let attentionMask = attentionMask {
            attentionScores = attentionScores + attentionMask
        }

        let attentionProbs = MLX.softmax(attentionScores, axis: -1)

        var contextLayer = MLX.matmul(attentionProbs, valueLayer)
        contextLayer = contextLayer.transposed(0, 2, 1, 3)

        var newContextLayerShape: [Int] = []
        let shape = contextLayer.shape

        for i in 0..<(shape.count - 2) {
            newContextLayerShape.append(shape[i])
        }

        newContextLayerShape.append(allHeadSize)

        contextLayer = contextLayer.reshaped(newContextLayerShape)
        contextLayer = dense(contextLayer)
        contextLayer = layerNorm(contextLayer + hiddenStates)

        return contextLayer
    }
}

// MARK: - AlbertLayer

class AlbertLayer {
    let attention: AlbertSelfAttention
    let fullLayerLayerNorm: LayerNorm
    let ffn: Linear
    let ffnOutput: Linear
    let seqLenDim: Int

    init(weights: [String: MLXArray], config: AlbertModelArgs, layerNum: Int, innerGroupNum: Int) {
        attention = AlbertSelfAttention(weights: weights, config: config, layerNum: layerNum, innerGroupNum: innerGroupNum)
        ffn = Linear(
            weight: weights["bert.encoder.albert_layer_groups.\(layerNum).albert_layers.\(innerGroupNum).ffn.weight"]!,
            bias: weights["bert.encoder.albert_layer_groups.\(layerNum).albert_layers.\(innerGroupNum).ffn.bias"]!
        )
        ffnOutput = Linear(
            weight: weights["bert.encoder.albert_layer_groups.\(layerNum).albert_layers.\(innerGroupNum).ffn_output.weight"]!,
            bias: weights["bert.encoder.albert_layer_groups.\(layerNum).albert_layers.\(innerGroupNum).ffn_output.bias"]!
        )
        seqLenDim = 1
        fullLayerLayerNorm = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)

        let fullLayerLayerNormWeights = weights["bert.encoder.albert_layer_groups.\(layerNum).albert_layers.\(innerGroupNum).full_layer_layer_norm.weight"]!
        let fullLayerLayerNormBiases = weights["bert.encoder.albert_layer_groups.\(layerNum).albert_layers.\(innerGroupNum).full_layer_layer_norm.bias"]!

        guard fullLayerLayerNormWeights.count == config.hiddenSize, fullLayerLayerNormBiases.count == config.hiddenSize else {
            fatalError("Wrong shape for AlbertLayer FullLayerLayerNorm bias or weights!")
        }

        for i in 0..<config.hiddenSize {
            fullLayerLayerNorm.weight![i] = fullLayerLayerNormWeights[i]
            fullLayerLayerNorm.bias![i] = fullLayerLayerNormBiases[i]
        }
    }

    func ffChunk(_ attentionOutput: MLXArray) -> MLXArray {
        var ffnOutputArray = ffn(attentionOutput)
        ffnOutputArray = MLXNN.gelu(ffnOutputArray)
        ffnOutputArray = ffnOutput(ffnOutputArray)
        return ffnOutputArray
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        attentionMask: MLXArray? = nil
    ) -> MLXArray {
        let attentionOutput = attention(hiddenStates, attentionMask: attentionMask)
        let ffnOutput = ffChunk(attentionOutput)
        let output = fullLayerLayerNorm(ffnOutput + attentionOutput)
        return output
    }
}

// MARK: - AlbertLayerGroup

class AlbertLayerGroup {
    let albertLayers: [AlbertLayer]

    init(config: AlbertModelArgs, layerNum: Int, weights: [String: MLXArray]) {
        var layers: [AlbertLayer] = []
        for innerGroupNum in 0..<config.innerGroupNum {
            layers.append(AlbertLayer(weights: weights, config: config, layerNum: layerNum, innerGroupNum: innerGroupNum))
        }
        albertLayers = layers
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        attentionMask: MLXArray? = nil
    ) -> MLXArray {
        var output = hiddenStates
        for layer in albertLayers {
            output = layer(output, attentionMask: attentionMask)
        }
        return output
    }
}

// MARK: - AlbertEncoder

class AlbertEncoder {
    let config: AlbertModelArgs
    let embeddingHiddenMappingIn: Linear
    let albertLayerGroups: [AlbertLayerGroup]

    init(weights: [String: MLXArray], config: AlbertModelArgs) {
        self.config = config
        embeddingHiddenMappingIn = Linear(
            weight: weights["bert.encoder.embedding_hidden_mapping_in.weight"]!,
            bias: weights["bert.encoder.embedding_hidden_mapping_in.bias"]!
        )

        var groups: [AlbertLayerGroup] = []
        for layerNum in 0..<config.numHiddenGroups {
            groups.append(AlbertLayerGroup(config: config, layerNum: layerNum, weights: weights))
        }
        albertLayerGroups = groups
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        attentionMask: MLXArray? = nil
    ) -> MLXArray {
        var output = embeddingHiddenMappingIn(hiddenStates)

        for i in 0..<config.numHiddenLayers {
            let groupIdx = i / (config.numHiddenLayers / config.numHiddenGroups)
            output = albertLayerGroups[groupIdx](output, attentionMask: attentionMask)
        }

        return output
    }
}

// MARK: - CustomAlbert

class CustomAlbert {
    let config: AlbertModelArgs
    let embeddings: AlbertEmbeddings
    let encoder: AlbertEncoder
    let pooler: Linear

    init(weights: [String: MLXArray], config: AlbertModelArgs) {
        self.config = config
        embeddings = AlbertEmbeddings(weights: weights, config: config)
        encoder = AlbertEncoder(weights: weights, config: config)
        pooler = Linear(weight: weights["bert.pooler.weight"]!, bias: weights["bert.pooler.bias"]!)
    }

    func callAsFunction(
        _ inputIds: MLXArray,
        tokenTypeIds: MLXArray? = nil,
        attentionMask: MLXArray? = nil
    ) -> (sequenceOutput: MLXArray, pooledOutput: MLXArray) {
        let embeddingOutput = embeddings(inputIds, tokenTypeIds: tokenTypeIds)

        var attentionMaskProcessed: MLXArray?
        if let attentionMask = attentionMask {
            let shape = attentionMask.shape
            let newDims = [shape[0], 1, 1, shape[1]]
            attentionMaskProcessed = attentionMask.reshaped(newDims)
            attentionMaskProcessed = (1.0 - attentionMaskProcessed!) * -10000.0
        }

        let sequenceOutput = encoder(embeddingOutput, attentionMask: attentionMaskProcessed)
        let firstTokenReshaped = sequenceOutput[0..., 0, 0...]
        let pooledOutput = MLX.tanh(pooler(firstTokenReshaped))

        return (sequenceOutput, pooledOutput)
    }
}
