//
//  KokoroLayers.swift
//  MLXAudio
//
//  Ported from mlx-audio Kokoro implementation
//

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Weight Normalization

func computeNorm(
    x: MLXArray,
    p: Int,
    dim: [Int]? = nil,
    keepdim: Bool = false
) -> MLXArray {
    guard p == 1 || p == 2 else {
        fatalError("Only p-norms with p of 1 or 2 are supported")
    }

    let dimensions: [Int]
    if let dim = dim {
        dimensions = dim
    } else {
        dimensions = Array(0..<x.ndim)
    }

    if p == 1 {
        return MLX.sum(MLX.abs(x), axes: dimensions, keepDims: keepdim)
    } else {
        return MLX.sqrt(MLX.sum(x * x, axes: dimensions, keepDims: keepdim))
    }
}

func weightNorm(
    weightV: MLXArray,
    weightG: MLXArray,
    dim: Int? = nil
) -> MLXArray {
    let rank = weightV.shape.count

    var axes: [Int]

    if let dim = dim {
        var adjustedDim = dim
        if dim < 0 {
            adjustedDim += rank
        }

        axes = Array(0..<rank)
        if adjustedDim != -1 {
            axes.removeAll(where: { $0 == adjustedDim })
        }
    } else {
        axes = Array(0..<rank)
    }

    let normV = computeNorm(x: weightV, p: 2, dim: axes, keepdim: true)
    let normalizedWeight = weightV / (normV + 1e-7)
    return normalizedWeight * weightG
}

// MARK: - ConvWeighted

/// Conv1d with weight normalization
class ConvWeighted: Module {
    var weightG: MLXArray
    var weightV: MLXArray
    var bias: MLXArray?

    let stride: Int
    let padding: Int
    let dilation: Int
    let groups: Int

    init(
        weightG: MLXArray,
        weightV: MLXArray,
        bias: MLXArray?,
        stride: Int = 1,
        padding: Int = 1,
        dilation: Int = 1,
        groups: Int = 1
    ) {
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weightG = weightG
        self.weightV = weightV
        self.bias = bias
        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray,
        conv: (MLXArray, MLXArray, Int, Int, Int, Int, StreamOrDevice) -> MLXArray
    ) -> MLXArray {
        let weight = weightNorm(weightV: weightV, weightG: weightG, dim: 0)
        bias = bias?.reshaped([1, 1, -1])

        func applyConv(x: MLXArray, weightToUse: MLXArray) -> MLXArray {
            let result = conv(
                x,
                weightToUse,
                self.stride,
                padding,
                dilation,
                groups,
                .default
            )

            if let bias = bias {
                return result + bias
            }
            return result
        }

        if x.shape.last == weight.shape.last || groups > 1 {
            return applyConv(x: x, weightToUse: weight)
        } else {
            return applyConv(x: x, weightToUse: weight.transposed())
        }
    }
}

// MARK: - Conv1dInference

class Conv1dInference {
    public let weight: MLXArray
    public let bias: MLXArray?
    public let padding: Int
    public let dilation: Int
    public let stride: Int
    public let groups: Int

    public init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        dilation: Int = 1,
        groups: Int = 1,
        weight: MLXArray,
        bias: MLXArray? = nil
    ) {
        self.weight = weight
        self.bias = bias
        self.padding = padding
        self.dilation = dilation
        self.stride = stride
        self.groups = groups
    }

    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = conv1d(
            x, weight, stride: stride, padding: padding, dilation: dilation, groups: groups
        )

        if let bias {
            y = y + bias
        }
        return y
    }
}

// MARK: - LayerNormInference

class LayerNormInference: Module {
    public let eps: Float
    public let weight: MLXArray?
    public let bias: MLXArray?

    public init(weight: MLXArray, bias: MLXArray?, eps: Float = 1e-5) {
        self.weight = weight
        self.bias = bias
        self.eps = eps
    }

    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.layerNorm(x, weight: weight, bias: bias, eps: eps)
    }
}

// MARK: - Instance Normalization

class _InstanceNorm {
    let numFeatures: Int
    let eps: Float
    let momentum: Float
    let affine: Bool
    let trackRunningStats: Bool

    var weight: MLXArray?
    var bias: MLXArray?
    var runningMean: MLXArray?
    var runningVar: MLXArray?
    var training: Bool = true

    init(
        numFeatures: Int,
        eps: Float = 1e-5,
        momentum: Float = 0.1,
        affine: Bool = false,
        trackRunningStats: Bool = false
    ) {
        self.numFeatures = numFeatures
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.trackRunningStats = trackRunningStats

        if self.affine {
            weight = MLXArray.ones([numFeatures])
            bias = MLXArray.zeros([numFeatures])
        }

        if self.trackRunningStats {
            runningMean = MLXArray.zeros([numFeatures])
            runningVar = MLXArray.ones([numFeatures])
        }
    }

    func checkInputDim(_ input: MLXArray) {
        fatalError("Subclass must implement checkInputDim")
    }

    func getNoBatchDim() -> Int {
        fatalError("Subclass must implement getNoBatchDim")
    }

    func handleNoBatchInput(_ input: MLXArray) -> MLXArray {
        let expanded = input.expandedDimensions(axis: 0)
        let result = applyInstanceNorm(expanded)
        return result.squeezed(axes: [0])
    }

    func applyInstanceNorm(_ input: MLXArray) -> MLXArray {
        let dims = Array(0..<input.ndim)
        let featureDim = dims[dims.count - getNoBatchDim()]
        let reduceDims = dims.filter { $0 != 0 && $0 != featureDim }

        var mean: MLXArray
        var variance: MLXArray

        if training || !trackRunningStats {
            mean = MLX.mean(input, axes: reduceDims, keepDims: true)
            variance = MLX.variance(input, axes: reduceDims, keepDims: true)

            if trackRunningStats && training, let runningMean = runningMean, let runningVar = runningVar {
                let overallMean = MLX.mean(mean, axes: [0])
                let overallVar = MLX.mean(variance, axes: [0])

                self.runningMean = (1 - momentum) * runningMean + momentum * overallMean
                self.runningVar = (1 - momentum) * runningVar + momentum * overallVar
            }
        } else if let runningMean = runningMean, let runningVar = runningVar {
            var meanShape = Array(repeating: 1, count: input.ndim)
            meanShape[featureDim] = numFeatures
            let varShape = meanShape

            mean = runningMean.reshaped(meanShape)
            variance = runningVar.reshaped(varShape)
        } else {
            fatalError("Running statistics not available")
        }

        let xNorm = (input - mean) / MLX.sqrt(variance + eps)

        if affine, let weight = weight, let bias = bias {
            var weightShape = Array(repeating: 1, count: input.ndim)
            weightShape[featureDim] = numFeatures
            let biasShape = weightShape

            let reshapedWeight = weight.reshaped(weightShape)
            let reshapedBias = bias.reshaped(biasShape)

            return xNorm * reshapedWeight + reshapedBias
        } else {
            return xNorm
        }
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        checkInputDim(input)

        let featureDim = input.ndim - getNoBatchDim()
        if input.shape[featureDim] != numFeatures {
            if affine {
                fatalError("Expected input's size at dim=\(featureDim) to match numFeatures (\(numFeatures)), but got: \(input.shape[featureDim]).")
            }
        }

        if input.ndim == getNoBatchDim() {
            return handleNoBatchInput(input)
        }

        return applyInstanceNorm(input)
    }
}

class InstanceNorm1d: _InstanceNorm {
    override func getNoBatchDim() -> Int {
        return 2
    }

    override func checkInputDim(_ input: MLXArray) {
        if input.ndim != 2, input.ndim != 3 {
            fatalError("Expected 2D or 3D input (got \(input.ndim)D input)")
        }
    }
}

// MARK: - AdaIN1d

class AdaIN1d {
    private let norm: InstanceNorm1d
    private let fc: Linear

    public init(styleDim: Int, numFeatures: Int, fcWeight: MLXArray, fcBias: MLXArray) {
        norm = InstanceNorm1d(numFeatures: numFeatures, affine: false)
        fc = Linear(weight: fcWeight, bias: fcBias)
    }

    public func callAsFunction(_ x: MLXArray, s: MLXArray) -> MLXArray {
        let h = fc(s)
        let hExpanded = h.expandedDimensions(axes: [2])
        let split = hExpanded.split(parts: 2, axis: 1)
        let gamma = split[0]
        let beta = split[1]

        let normalized = norm(x)
        return (1 + gamma) * normalized + beta
    }
}

// MARK: - AdaLayerNorm

class AdaLayerNorm: Module {
    let eps: Float
    let fc: Linear

    init(eps: Float = 1e-5, weight: MLXArray, bias: MLXArray?) {
        self.eps = eps
        fc = Linear(weight: weight, bias: bias)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, _ s: MLXArray) -> MLXArray {
        let h = fc(s)
        let reshaped = h.reshaped([h.shape[0], h.shape[1], 1])
        let split = reshaped.split(parts: 2, axis: 1)
        let gamma = split[0].transposed(2, 0, 1)
        let beta = split[1].transposed(2, 0, 1)

        let mean = MLX.mean(x, axes: [-1], keepDims: true)
        let variance = MLX.variance(x, axes: [-1], keepDims: true)
        let normalized = (x - mean) / MLX.sqrt(variance + eps)

        return (1 + gamma) * normalized + beta
    }
}

// MARK: - UpSample1d

class UpSample1d {
    private let layerType: String
    private let interpolate: Upsample

    init(layerType: String) {
        self.layerType = layerType
        interpolate = Upsample(
            scaleFactor: 2.0,
            mode: .nearest
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if layerType == "none" {
            return x
        } else {
            return interpolate(x)
        }
    }
}

// MARK: - ReflectionPad1d

class ReflectionPad1d: Module {
    let padding: IntOrPair

    init(padding: (Int, Int)) {
        self.padding = IntOrPair([padding.0, padding.1])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return MLX.padded(x, widths: [IntOrPair([0, 0]), IntOrPair([0, 0]), padding])
    }
}

// MARK: - AdainResBlk1d

class AdainResBlk1d {
    let actv: LeakyReLU
    let dimIn: Int
    let upsampleType: String
    let upsample: UpSample1d
    let learned_sc: Bool
    let pool: Module

    var conv1: ConvWeighted!
    var conv2: ConvWeighted!
    var norm1: AdaIN1d!
    var norm2: AdaIN1d!
    var conv1x1: ConvWeighted?

    init(
        weights: [String: MLXArray],
        weightKeyPrefix: String,
        dimIn: Int,
        dimOut: Int,
        styleDim: Int = 64,
        actv: LeakyReLU = LeakyReLU(negativeSlope: 0.2),
        upsample: String = "none"
    ) {
        self.actv = actv
        self.dimIn = dimIn
        upsampleType = upsample
        self.upsample = UpSample1d(layerType: upsample)
        learned_sc = dimIn != dimOut

        if upsample == "none" {
            pool = Identity()
        } else {
            pool = ConvWeighted(
                weightG: weights[weightKeyPrefix + ".pool.weight_g"]!,
                weightV: weights[weightKeyPrefix + ".pool.weight_v"]!,
                bias: weights[weightKeyPrefix + ".pool.bias"]!,
                stride: 2,
                padding: 1,
                groups: dimIn
            )
        }

        buildWeights(weights: weights, weightKeyPrefix: weightKeyPrefix, dimIn: dimIn, dimOut: dimOut, styleDim: styleDim)
    }

    func buildWeights(weights: [String: MLXArray], weightKeyPrefix: String, dimIn: Int, dimOut: Int, styleDim: Int) {
        conv1 = ConvWeighted(
            weightG: weights[weightKeyPrefix + ".conv1.weight_g"]!,
            weightV: weights[weightKeyPrefix + ".conv1.weight_v"]!,
            bias: weights[weightKeyPrefix + ".conv1.bias"]!,
            stride: 1,
            padding: 1
        )

        conv2 = ConvWeighted(
            weightG: weights[weightKeyPrefix + ".conv2.weight_g"]!,
            weightV: weights[weightKeyPrefix + ".conv2.weight_v"]!,
            bias: weights[weightKeyPrefix + ".conv2.bias"]!,
            stride: 1,
            padding: 1
        )

        norm1 = AdaIN1d(
            styleDim: styleDim,
            numFeatures: dimIn,
            fcWeight: weights[weightKeyPrefix + ".norm1.fc.weight"]!,
            fcBias: weights[weightKeyPrefix + ".norm1.fc.bias"]!
        )

        norm2 = AdaIN1d(
            styleDim: styleDim,
            numFeatures: dimIn,
            fcWeight: weights[weightKeyPrefix + ".norm2.fc.weight"]!,
            fcBias: weights[weightKeyPrefix + ".norm2.fc.bias"]!
        )

        if learned_sc {
            conv1x1 = ConvWeighted(
                weightG: weights[weightKeyPrefix + ".conv1x1.weight_g"]!,
                weightV: weights[weightKeyPrefix + ".conv1x1.weight_v"]!,
                bias: nil,
                stride: 1,
                padding: 0
            )
        }
    }

    func shortcut(_ x: MLXArray) -> MLXArray {
        var x = MLX.swappedAxes(x, 2, 1)
        x = upsample(x)
        x = MLX.swappedAxes(x, 2, 1)

        if let conv1x1 = conv1x1 {
            x = MLX.swappedAxes(x, 2, 1)
            x = conv1x1(x, conv: MLX.conv1d)
            x = MLX.swappedAxes(x, 2, 1)
        }

        return x
    }

    func residual(_ x: MLXArray, _ s: MLXArray) -> MLXArray {
        var x = norm1(x, s: s)
        x = actv(x)

        x = MLX.swappedAxes(x, 2, 1)
        if upsampleType != "none" {
            if let idPool = pool as? Identity {
                x = idPool(x)
            } else if let convPool = pool as? ConvWeighted {
                x = convPool.callAsFunction(x, conv: { a, b, c, d, e, f, g in
                    MLX.convTransposed1d(a, b, stride: c, padding: d, dilation: e, outputPadding: 0, groups: f, stream: g)
                })
            }
            x = MLX.padded(x, widths: [IntOrPair([0, 0]), IntOrPair([1, 0]), IntOrPair([0, 0])])
        }
        x = MLX.swappedAxes(x, 2, 1)

        x = MLX.swappedAxes(x, 2, 1)
        x = conv1(x, conv: MLX.conv1d)
        x = MLX.swappedAxes(x, 2, 1)

        x = norm2(x, s: s)
        x = actv(x)

        x = MLX.swappedAxes(x, 2, 1)
        x = conv2(x, conv: MLX.conv1d)
        x = MLX.swappedAxes(x, 2, 1)

        return x
    }

    func callAsFunction(x: MLXArray, s: MLXArray) -> MLXArray {
        let out = residual(x, s)
        let result = (out + shortcut(x)) / sqrt(2.0)
        return result
    }
}

// MARK: - AdaINResBlock1

class AdaINResBlock1 {
    var convs1: [ConvWeighted] = []
    var convs2: [ConvWeighted] = []
    var adain1: [AdaIN1d] = []
    var adain2: [AdaIN1d] = []
    var alpha1: [MLXArray] = []
    var alpha2: [MLXArray] = []

    private func getPadding(kernelSize: Int, dilation: Int = 1) -> Int {
        return Int((kernelSize * dilation - dilation) / 2)
    }

    init(
        weights: [String: MLXArray],
        weightPrefixKey: String,
        channels: Int,
        kernelSize: Int = 3,
        dilation: [Int] = [1, 3, 5],
        styleDim: Int = 64
    ) {
        for i in 0..<3 {
            let dilationValue = dilation[i]
            let conv = ConvWeighted(
                weightG: weights[weightPrefixKey + ".convs1.\(i).weight_g"]!,
                weightV: weights[weightPrefixKey + ".convs1.\(i).weight_v"]!,
                bias: weights[weightPrefixKey + ".convs1.\(i).bias"]!,
                stride: 1,
                padding: getPadding(kernelSize: kernelSize, dilation: dilationValue),
                dilation: dilationValue
            )
            convs1.append(conv)
        }

        for i in 0..<convs1.count {
            let conv = ConvWeighted(
                weightG: weights[weightPrefixKey + ".convs2.\(i).weight_g"]!,
                weightV: weights[weightPrefixKey + ".convs2.\(i).weight_v"]!,
                bias: weights[weightPrefixKey + ".convs2.\(i).bias"]!,
                stride: 1,
                padding: getPadding(kernelSize: kernelSize, dilation: 1),
                dilation: 1
            )
            convs2.append(conv)
        }

        for i in 0..<convs1.count {
            adain1.append(AdaIN1d(
                styleDim: styleDim,
                numFeatures: channels,
                fcWeight: weights[weightPrefixKey + ".adain1.\(i).fc.weight"]!,
                fcBias: weights[weightPrefixKey + ".adain1.\(i).fc.bias"]!
            ))

            adain2.append(AdaIN1d(
                styleDim: styleDim,
                numFeatures: channels,
                fcWeight: weights[weightPrefixKey + ".adain2.\(i).fc.weight"]!,
                fcBias: weights[weightPrefixKey + ".adain2.\(i).fc.bias"]!
            ))
        }

        for i in 0..<convs1.count {
            alpha1.append(weights[weightPrefixKey + ".alpha1.\(i)"]!)
            alpha2.append(weights[weightPrefixKey + ".alpha2.\(i)"]!)
        }
    }

    func callAsFunction(_ x: MLXArray, _ s: MLXArray) -> MLXArray {
        var result = x

        for i in 0..<convs1.count {
            let c1 = convs1[i]
            let c2 = convs2[i]
            let n1 = adain1[i]
            let n2 = adain2[i]
            let a1 = alpha1[i]
            let a2 = alpha2[i]

            var xt = n1(result, s: s)
            xt = xt + (1 / a1) * (MLX.sin(a1 * xt).pow(2))

            xt = MLX.swappedAxes(xt, 2, 1)
            xt = c1(xt, conv: MLX.conv1d)
            xt = MLX.swappedAxes(xt, 2, 1)

            xt = n2(xt, s: s)
            xt = xt + (1 / a2) * (MLX.sin(a2 * xt).pow(2))

            xt = MLX.swappedAxes(xt, 2, 1)
            xt = c2(xt, conv: MLX.conv1d)
            xt = MLX.swappedAxes(xt, 2, 1)

            result = xt + result
        }
        return result
    }
}

// MARK: - LSTM

class LSTM: Module {
    let inputSize: Int
    let hiddenSize: Int
    let hasBias: Bool
    let batchFirst: Bool

    var wxForward: MLXArray
    var whForward: MLXArray
    var biasIhForward: MLXArray?
    var biasHhForward: MLXArray?

    var wxBackward: MLXArray
    var whBackward: MLXArray
    var biasIhBackward: MLXArray?
    var biasHhBackward: MLXArray?

    init(
        inputSize: Int,
        hiddenSize: Int,
        bias: Bool = true,
        batchFirst: Bool = true,
        wxForward: MLXArray,
        whForward: MLXArray,
        biasIhForward: MLXArray? = nil,
        biasHhForward: MLXArray? = nil,
        wxBackward: MLXArray,
        whBackward: MLXArray,
        biasIhBackward: MLXArray? = nil,
        biasHhBackward: MLXArray? = nil
    ) {
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        hasBias = bias
        self.batchFirst = batchFirst

        self.wxForward = wxForward
        self.whForward = whForward
        self.biasIhForward = biasIhForward
        self.biasHhForward = biasHhForward

        self.wxBackward = wxBackward
        self.whBackward = whBackward
        self.biasIhBackward = biasIhBackward
        self.biasHhBackward = biasHhBackward

        super.init()
    }

    private func forwardDirection(
        _ x: MLXArray,
        hidden: MLXArray? = nil,
        cell: MLXArray? = nil
    ) -> (MLXArray, MLXArray) {
        let xProj: MLXArray
        if let biasIhForward = biasIhForward, let biasHhForward = biasHhForward {
            xProj = MLX.addMM(
                biasIhForward + biasHhForward,
                x,
                wxForward.transposed()
            )
        } else {
            xProj = MLX.matmul(x, wxForward.transposed())
        }

        var allHidden: [MLXArray] = []
        var allCell: [MLXArray] = []

        let seqLen = x.shape[x.shape.count - 2]

        var currentHidden = hidden ?? MLXArray.zeros([x.shape[0], hiddenSize])
        var currentCell = cell ?? MLXArray.zeros([x.shape[0], hiddenSize])

        for idx in 0..<seqLen {
            var ifgo = xProj[0..., idx, 0...]
            ifgo = ifgo + MLX.matmul(currentHidden, whForward.transposed())

            let gates = MLX.split(ifgo, parts: 4, axis: -1)
            let i = MLX.sigmoid(gates[0])
            let f = MLX.sigmoid(gates[1])
            let g = MLX.tanh(gates[2])
            let o = MLX.sigmoid(gates[3])

            currentCell = f * currentCell + i * g
            currentHidden = o * MLX.tanh(currentCell)

            allCell.append(currentCell)
            allHidden.append(currentHidden)
        }

        return (MLX.stacked(allHidden, axis: -2), MLX.stacked(allCell, axis: -2))
    }

    private func backwardDirection(
        _ x: MLXArray,
        hidden: MLXArray? = nil,
        cell: MLXArray? = nil
    ) -> (MLXArray, MLXArray) {
        let xProj: MLXArray
        if let biasIhBackward = biasIhBackward, let biasHhBackward = biasHhBackward {
            xProj = MLX.addMM(
                biasIhBackward + biasHhBackward,
                x,
                wxBackward.transposed()
            )
        } else {
            xProj = MLX.matmul(x, wxBackward.transposed())
        }

        var allHidden: [MLXArray] = []
        var allCell: [MLXArray] = []

        let seqLen = x.shape[x.shape.count - 2]

        var currentHidden = hidden ?? MLXArray.zeros([x.shape[0], hiddenSize])
        var currentCell = cell ?? MLXArray.zeros([x.shape[0], hiddenSize])

        for idx in stride(from: seqLen - 1, through: 0, by: -1) {
            var ifgo = xProj[0..., idx, 0...]
            ifgo = ifgo + MLX.matmul(currentHidden, whBackward.transposed())

            let gates = MLX.split(ifgo, parts: 4, axis: -1)
            let i = MLX.sigmoid(gates[0])
            let f = MLX.sigmoid(gates[1])
            let g = MLX.tanh(gates[2])
            let o = MLX.sigmoid(gates[3])

            currentCell = f * currentCell + i * g
            currentHidden = o * MLX.tanh(currentCell)

            allCell.insert(currentCell, at: 0)
            allHidden.insert(currentHidden, at: 0)
        }

        return (MLX.stacked(allHidden, axis: -2), MLX.stacked(allCell, axis: -2))
    }

    func callAsFunction(
        _ x: MLXArray,
        hiddenForward: MLXArray? = nil,
        cellForward: MLXArray? = nil,
        hiddenBackward: MLXArray? = nil,
        cellBackward: MLXArray? = nil
    ) -> (MLXArray, ((MLXArray, MLXArray), (MLXArray, MLXArray))) {
        let input: MLXArray
        if x.ndim == 2 {
            input = x.expandedDimensions(axis: 0)
        } else {
            input = x
        }

        let (forwardHidden, forwardCell) = forwardDirection(
            input,
            hidden: hiddenForward,
            cell: cellForward
        )

        let (backwardHidden, backwardCell) = backwardDirection(
            input,
            hidden: hiddenBackward,
            cell: cellBackward
        )

        let output = MLX.concatenated([forwardHidden, backwardHidden], axis: -1)

        return (
            output,
            (
                (forwardHidden[0..., -1, 0...], forwardCell[0..., -1, 0...]),
                (backwardHidden[0..., 0, 0...], backwardCell[0..., 0, 0...])
            )
        )
    }
}

// MARK: - Interpolate Functions

func interpolate(
    input: MLXArray,
    size: [Int]? = nil,
    scaleFactor: [Float]? = nil,
    mode: String = "nearest",
    alignCorners: Bool? = nil
) -> MLXArray {
    let ndim = input.ndim
    if ndim < 3 {
        fatalError("Expected at least 3D input (N, C, D1), got \(ndim)D")
    }

    let spatialDims = ndim - 2

    if size != nil && scaleFactor != nil {
        fatalError("Only one of size or scaleFactor should be defined")
    } else if size == nil && scaleFactor == nil {
        fatalError("One of size or scaleFactor must be defined")
    }

    var outputSize: [Int] = []
    if let scaleFactor = scaleFactor {
        let factors = scaleFactor.count == 1 ? Array(repeating: scaleFactor[0], count: spatialDims) : scaleFactor

        for i in 0..<spatialDims {
            let currSize = max(1, Int(ceil(Float(input.shape[i + 2]) * factors[i])))
            outputSize.append(currSize)
        }
    } else if let size = size {
        outputSize = size.count == 1 ? Array(repeating: size[0], count: spatialDims) : size
    }

    if spatialDims == 1 {
        return interpolate1d(input: input, size: outputSize[0], mode: mode, alignCorners: alignCorners)
    } else {
        fatalError("Only 1D interpolation currently supported, got \(spatialDims)D")
    }
}

func interpolate1d(
    input: MLXArray,
    size: Int,
    mode: String = "linear",
    alignCorners: Bool? = nil
) -> MLXArray {
    let shape = input.shape
    let batchSize = shape[0]
    let channels = shape[1]
    let inWidth = shape[2]

    let outputSize = max(1, size)
    let inputWidth = max(1, inWidth)

    if mode == "nearest" {
        if outputSize == 1 {
            let indices = MLXArray(converting: [0]).asType(.int32)
            return input[0..., 0..., indices]
        } else {
            let scale = Float(inputWidth) / Float(outputSize)
            let indices = MLX.floor(MLXArray(0..<outputSize).asType(.float32) * scale).asType(.int32)
            let clippedIndices = MLX.clip(indices, min: 0, max: inputWidth - 1)
            return input[0..., 0..., clippedIndices]
        }
    }

    var x: MLXArray
    if alignCorners == true && outputSize > 1 {
        x = MLXArray(0..<outputSize).asType(.float32) * (Float(inputWidth - 1) / Float(outputSize - 1))
    } else {
        if outputSize == 1 {
            x = MLXArray(converting: [0.0]).asType(.float32)
        } else {
            x = MLXArray(0..<outputSize).asType(.float32) * (Float(inputWidth) / Float(outputSize))
            if alignCorners != true {
                x = x + 0.5 * (Float(inputWidth) / Float(outputSize)) - 0.5
            }
        }
    }

    if inputWidth == 1 {
        let outputShape = [batchSize, channels, outputSize]
        return MLX.broadcast(input, to: outputShape)
    }

    let xLow = MLX.floor(x).asType(.int32)
    let xHigh = MLX.minimum(xLow + 1, MLXArray(inputWidth - 1, dtype: .int32))
    let xFrac = x - xLow.asType(.float32)

    let yLow = input[0..., 0..., xLow]
    let yHigh = input[0..., 0..., xHigh]

    let oneMinusXFrac = 1 - xFrac
    let output = yLow * oneMinusXFrac.expandedDimensions(axis: 0).expandedDimensions(axis: 0) +
        yHigh * xFrac.expandedDimensions(axis: 0).expandedDimensions(axis: 0)

    return output
}
