//
//  KokoroDecoder.swift
//  MLXAudio
//
//  Ported from mlx-audio Kokoro implementation
//

import Foundation
import MLX
import MLXFFT
import MLXNN
import MLXRandom

// MARK: - STFT Utilities

/// Hanning window implementation
func hanning(length: Int) -> MLXArray {
    if length == 1 {
        return MLXArray(1.0)
    }

    let n = MLXArray(Array(stride(from: Float(1 - length), to: Float(length), by: 2.0)))
    let factor = .pi / Float(length - 1)
    return 0.5 + 0.5 * cos(n * factor)
}

/// Unwrap phase implementation
func unwrap(p: MLXArray) -> MLXArray {
    let period: Float = 2.0 * .pi
    let discont: Float = period / 2.0

    let pDiff1 = p[0..., 0..<p.shape[1] - 1]
    let pDiff2 = p[0..., 1..<p.shape[1]]

    let pDiff = pDiff2 - pDiff1

    let intervalHigh: Float = period / 2.0
    let intervalLow: Float = -intervalHigh

    var pDiffMod = pDiff - intervalLow
    pDiffMod = (((pDiffMod % period) + period) % period) + intervalLow

    let ddSignArray = MLX.where(pDiff .> 0, intervalHigh, pDiffMod)
    pDiffMod = MLX.where(pDiffMod .== intervalLow, ddSignArray, pDiffMod)

    var phCorrect = pDiffMod - pDiff
    phCorrect = MLX.where(abs(pDiff) .< discont, MLXArray(0.0), phCorrect)

    return MLX.concatenated([p[0..., 0..<1], p[0..., 1...] + phCorrect.cumsum(axis: 1)], axis: 1)
}

func getWindow(window: Any, winLen: Int, nFft: Int) -> MLXArray {
    var w: MLXArray
    if let windowStr = window as? String {
        if windowStr.lowercased() == "hann" {
            w = hanning(length: winLen + 1)[0..<winLen]
        } else {
            fatalError("Only hanning is supported for window, not \(windowStr)")
        }
    } else if let windowArray = window as? MLXArray {
        w = windowArray
    } else {
        fatalError("Window must be a string or MLXArray")
    }

    if w.shape[0] < nFft {
        let padSize = nFft - w.shape[0]
        w = MLX.concatenated([w, MLXArray.zeros([padSize])], axis: 0)
    }
    return w
}

func mlxStft(
    x: MLXArray,
    nFft: Int = 800,
    hopLength: Int? = nil,
    winLength: Int? = nil,
    window: Any = "hann",
    center: Bool = true,
    padMode: String = "reflect"
) -> MLXArray {
    let hopLen = hopLength ?? nFft / 4
    let winLen = winLength ?? nFft

    let w = getWindow(window: window, winLen: winLen, nFft: nFft)

    func pad(_ x: MLXArray, padding: Int, padMode: String = "reflect") -> MLXArray {
        if padMode == "constant" {
            return MLX.padded(x, width: [padding, padding])
        } else if padMode == "reflect" {
            let prefix = x[1..<padding + 1][.stride(by: -1)]
            let suffix = x[-(padding + 1) ..< -1][.stride(by: -1)]
            return MLX.concatenated([prefix, x, suffix])
        } else {
            fatalError("Invalid pad mode \(padMode)")
        }
    }

    var xArray = x

    if center {
        xArray = pad(xArray, padding: nFft / 2, padMode: padMode)
    }

    let numFrames = 1 + (xArray.shape[0] - nFft) / hopLen
    if numFrames <= 0 {
        fatalError("Input is too short")
    }

    let shape: [Int] = [numFrames, nFft]
    let strides: [Int] = [hopLen, 1]

    let frames = MLX.asStrided(xArray, shape, strides: strides)

    let spec = MLXFFT.rfft(frames * w)
    return spec.transposed(1, 0)
}

func mlxIstft(
    x: MLXArray,
    hopLength: Int? = nil,
    winLength: Int? = nil,
    window: Any = "hann"
) -> MLXArray {
    let winLen = winLength ?? ((x.shape[1] - 1) * 2)
    let hopLen = hopLength ?? (winLen / 4)

    let w = getWindow(window: window, winLen: winLen, nFft: winLen)

    let xTransposed = x.transposed(1, 0)
    let t = (xTransposed.shape[0] - 1) * hopLen + winLen
    let windowModLen = 20 / 5

    let wSquared = w * w
    w.eval()
    wSquared.eval()

    let totalWsquared = MLX.concatenated(Array(repeating: wSquared, count: t / winLen))

    xTransposed.eval()

    let output = MLXFFT.irfft(xTransposed, axis: 1) * w
    output.eval()

    var outputs: [MLXArray] = []
    var windowSums: [MLXArray] = []

    for i in 0..<windowModLen {
        let outputStride = output[.stride(from: i, by: windowModLen), .ellipsis].reshaped([-1])
        let windowSumArray = totalWsquared[0..<outputStride.shape[0]]

        outputs.append(MLX.concatenated([
            MLXArray.zeros([i * hopLen]),
            outputStride,
            MLXArray.zeros([max(0, t - i * hopLen - outputStride.shape[0])]),
        ]))

        windowSums.append(MLX.concatenated([
            MLXArray.zeros([i * hopLen]),
            windowSumArray,
            MLXArray.zeros([max(0, t - i * hopLen - windowSumArray.shape[0])]),
        ]))
    }

    var reconstructed = outputs[0]
    var windowSum = windowSums[0]
    for i in 1..<windowModLen {
        reconstructed += outputs[i]
        windowSum += windowSums[i]
    }
    reconstructed.eval()
    windowSum.eval()

    reconstructed =
        reconstructed[winLen / 2..<(reconstructed.shape[0] - winLen / 2)] /
        windowSum[winLen / 2..<(reconstructed.shape[0] - winLen / 2)]
    reconstructed.eval()

    return reconstructed
}

// MARK: - MLXSTFT

class MLXSTFT {
    let filterLength: Int
    let hopLength: Int
    let winLength: Int
    let window: String

    var magnitude: MLXArray?
    var phase: MLXArray?

    init(filterLength: Int = 800, hopLength: Int = 200, winLength: Int = 800, window: String = "hann") {
        self.filterLength = filterLength
        self.hopLength = hopLength
        self.winLength = winLength
        self.window = window
    }

    func transform(inputData: MLXArray) -> (MLXArray, MLXArray) {
        var audioArray = inputData
        if audioArray.ndim == 1 {
            audioArray = audioArray.expandedDimensions(axis: 0)
        }

        var magnitudes: [MLXArray] = []
        var phases: [MLXArray] = []

        for batchIdx in 0..<audioArray.shape[0] {
            let stft = mlxStft(
                x: audioArray[batchIdx],
                nFft: self.filterLength,
                hopLength: self.hopLength,
                winLength: self.winLength,
                window: self.window,
                center: true,
                padMode: "reflect"
            )
            magnitudes.append(MLX.abs(stft))
            phases.append(MLX.atan2(stft.imaginaryPart(), stft.realPart()))
        }

        let magnitudesStacked = MLX.stacked(magnitudes, axis: 0)
        let phasesStacked = MLX.stacked(phases, axis: 0)

        return (magnitudesStacked, phasesStacked)
    }

    func inverse(magnitude: MLXArray, phase: MLXArray) -> MLXArray {
        var reconstructed: [MLXArray] = []

        for batchIdx in 0..<magnitude.shape[0] {
            let phaseCont = unwrap(p: phase[batchIdx])
            let stft = magnitude[batchIdx] * MLX.exp(MLXArray(real: 0, imaginary: 1) * phaseCont)
            stft.eval()

            let audio = mlxIstft(
                x: stft,
                hopLength: hopLength,
                winLength: winLength,
                window: window
            )
            audio.eval()
            reconstructed.append(audio)
        }

        let reconstructedStacked = MLX.stacked(reconstructed, axis: 0)
        return reconstructedStacked.expandedDimensions(axis: 1)
    }

    func callAsFunction(inputData: MLXArray) -> MLXArray {
        let (mag, ph) = transform(inputData: inputData)
        magnitude = mag
        phase = ph
        let reconstruction = inverse(magnitude: mag, phase: ph)
        return reconstruction.expandedDimensions(axis: -2)
    }
}

// MARK: - SineGen

class SineGen {
    private let sineAmp: Float
    private let noiseStd: Float
    private let harmonicNum: Int
    private let dim: Int
    private let samplingRate: Int
    private let voicedThreshold: Float
    private let upsampleScale: Float

    init(
        sampRate: Int,
        upsampleScale: Float,
        harmonicNum: Int = 0,
        sineAmp: Float = 0.1,
        noiseStd: Float = 0.003,
        voicedThreshold: Float = 0
    ) {
        self.sineAmp = sineAmp
        self.noiseStd = noiseStd
        self.harmonicNum = harmonicNum
        dim = harmonicNum + 1
        samplingRate = sampRate
        self.voicedThreshold = voicedThreshold
        self.upsampleScale = upsampleScale
    }

    private func _f02uv(_ f0: MLXArray) -> MLXArray {
        let arr = f0 .> voicedThreshold
        return arr.asType(.float32)
    }

    private func _f02sine(_ f0Values: MLXArray) -> MLXArray {
        var radValues = (f0Values / Float(samplingRate)) % 1

        let randIni = MLXRandom.normal([f0Values.shape[0], f0Values.shape[2]])
        randIni[0..., 0] = MLXArray(0.0)
        radValues[0..<radValues.shape[0], 0, 0..<radValues.shape[2]] = radValues[0..<radValues.shape[0], 0, 0..<radValues.shape[2]] + randIni

        radValues = interpolate(
            input: radValues.transposed(0, 2, 1),
            scaleFactor: [1 / Float(upsampleScale)],
            mode: "linear"
        ).transposed(0, 2, 1)

        var phase = MLX.cumsum(radValues, axis: 1) * 2 * Float.pi
        phase = interpolate(
            input: phase.transposed(0, 2, 1) * Float(upsampleScale),
            scaleFactor: [Float(upsampleScale)],
            mode: "linear"
        ).transposed(0, 2, 1)

        return MLX.sin(phase)
    }

    func callAsFunction(_ f0: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        let range = MLXArray(1...harmonicNum + 1).asType(.float32)
        let fn = f0 * range.reshaped([1, 1, range.shape[0]])

        let sineWaves = _f02sine(fn) * sineAmp
        let uv = _f02uv(f0)
        let noiseAmp = uv * noiseStd + (1 - uv) * sineAmp / 3
        let noise = noiseAmp * MLXRandom.normal(sineWaves.shape)

        let result = sineWaves * uv + noise
        return (result, uv, noise)
    }
}

// MARK: - SourceModuleHnNSF

class SourceModuleHnNSF: Module {
    private let sineAmp: Float
    private let noiseStd: Float
    private let lSinGen: SineGen
    private let lLinear: Linear

    init(
        weights: [String: MLXArray],
        samplingRate: Int,
        upsampleScale: Float,
        harmonicNum: Int = 0,
        sineAmp: Float = 0.1,
        addNoiseStd: Float = 0.003,
        voicedThreshold: Float = 0
    ) {
        self.sineAmp = sineAmp
        noiseStd = addNoiseStd

        lSinGen = SineGen(
            sampRate: samplingRate,
            upsampleScale: upsampleScale,
            harmonicNum: harmonicNum,
            sineAmp: sineAmp,
            noiseStd: addNoiseStd,
            voicedThreshold: voicedThreshold
        )

        lLinear = Linear(
            weight: weights["decoder.generator.m_source.l_linear.weight"]!,
            bias: weights["decoder.generator.m_source.l_linear.bias"]!
        )

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        let (sineWavs, uv, _) = lSinGen(x)
        let sineMerge = tanh(lLinear(sineWavs))

        let noise = MLXRandom.normal(uv.shape) * (sineAmp / 3)

        return (sineMerge, noise, uv)
    }
}

// MARK: - Generator

class Generator {
    let numKernels: Int
    let numUpsamples: Int
    let mSource: SourceModuleHnNSF
    let f0Upsample: Upsample
    let postNFFt: Int
    var noiseConvs: [Conv1dInference]
    var noiseRes: [AdaINResBlock1]
    var ups: [ConvWeighted]
    var resBlocks: [AdaINResBlock1]
    let convPost: ConvWeighted
    let reflectionPad: ReflectionPad1d
    let stft: MLXSTFT

    init(
        weights: [String: MLXArray],
        styleDim: Int,
        resblockKernelSizes: [Int],
        upsampleRates: [Int],
        upsampleInitialChannel: Int,
        resblockDilationSizes: [[Int]],
        upsampleKernelSizes: [Int],
        genIstftNFft: Int,
        genIstftHopSize: Int
    ) {
        numKernels = resblockKernelSizes.count
        numUpsamples = upsampleRates.count

        let upsampleScaleNum = MLX.product(MLXArray(upsampleRates)) * genIstftHopSize
        let upsampleScaleNumVal: Int = upsampleScaleNum.item()

        mSource = SourceModuleHnNSF(
            weights: weights,
            samplingRate: 24000,
            upsampleScale: upsampleScaleNum.item(),
            harmonicNum: 8,
            voicedThreshold: 10
        )

        f0Upsample = Upsample(scaleFactor: .float(Float(upsampleScaleNumVal)))

        noiseConvs = []
        noiseRes = []
        ups = []

        for (i, (u, k)) in zip(upsampleRates, upsampleKernelSizes).enumerated() {
            ups.append(
                ConvWeighted(
                    weightG: weights["decoder.generator.ups.\(i).weight_g"]!,
                    weightV: weights["decoder.generator.ups.\(i).weight_v"]!,
                    bias: weights["decoder.generator.ups.\(i).bias"]!,
                    stride: u,
                    padding: (k - u) / 2
                )
            )
        }

        resBlocks = []
        for i in 0..<ups.count {
            let ch = upsampleInitialChannel / Int(pow(2.0, Double(i + 1)))
            for (j, (k, d)) in zip(resblockKernelSizes, resblockDilationSizes).enumerated() {
                resBlocks.append(
                    AdaINResBlock1(
                        weights: weights,
                        weightPrefixKey: "decoder.generator.resblocks.\((i * resblockKernelSizes.count) + j)",
                        channels: ch,
                        kernelSize: k,
                        dilation: d,
                        styleDim: styleDim
                    )
                )
            }

            let cCur = ch
            if i + 1 < upsampleRates.count {
                let strideF0: Int = MLX.product(MLXArray(upsampleRates)[(i + 1)...]).item()
                noiseConvs.append(
                    Conv1dInference(
                        inputChannels: genIstftNFft + 2,
                        outputChannels: cCur,
                        kernelSize: strideF0 * 2,
                        stride: strideF0,
                        padding: (strideF0 + 1) / 2,
                        weight: weights["decoder.generator.noise_convs.\(i).weight"]!,
                        bias: weights["decoder.generator.noise_convs.\(i).bias"]!
                    )
                )

                noiseRes.append(
                    AdaINResBlock1(
                        weights: weights,
                        weightPrefixKey: "decoder.generator.noise_res.\(i)",
                        channels: cCur,
                        kernelSize: 7,
                        dilation: [1, 3, 5],
                        styleDim: styleDim
                    )
                )
            } else {
                noiseConvs.append(
                    Conv1dInference(
                        inputChannels: genIstftNFft + 2,
                        outputChannels: cCur,
                        kernelSize: 1,
                        weight: weights["decoder.generator.noise_convs.\(i).weight"]!,
                        bias: weights["decoder.generator.noise_convs.\(i).bias"]!
                    )
                )
                noiseRes.append(
                    AdaINResBlock1(
                        weights: weights,
                        weightPrefixKey: "decoder.generator.noise_res.\(i)",
                        channels: cCur,
                        kernelSize: 11,
                        dilation: [1, 3, 5],
                        styleDim: styleDim
                    )
                )
            }
        }

        postNFFt = genIstftNFft

        convPost = ConvWeighted(
            weightG: weights["decoder.generator.conv_post.weight_g"]!,
            weightV: weights["decoder.generator.conv_post.weight_v"]!,
            bias: weights["decoder.generator.conv_post.bias"]!,
            stride: 1,
            padding: 3
        )

        reflectionPad = ReflectionPad1d(padding: (1, 0))

        stft = MLXSTFT(
            filterLength: genIstftNFft,
            hopLength: genIstftHopSize,
            winLength: genIstftNFft
        )
    }

    func callAsFunction(_ x: MLXArray, _ s: MLXArray, _ F0Curve: MLXArray) -> MLXArray {
        var f0New = F0Curve[.newAxis, 0..., 0...].transposed(0, 2, 1)
        f0New = f0Upsample(f0New)

        var (harSource, _, _) = mSource(f0New)

        harSource = MLX.squeezed(harSource.transposed(0, 2, 1), axis: 1)
        let (harSpec, harPhase) = stft.transform(inputData: harSource)
        var har = MLX.concatenated([harSpec, harPhase], axis: 1)
        har = MLX.swappedAxes(har, 2, 1)

        var newX = x
        for i in 0..<numUpsamples {
            newX = LeakyReLU(negativeSlope: 0.1)(newX)
            var xSource = noiseConvs[i](har)
            xSource = MLX.swappedAxes(xSource, 2, 1)
            xSource = noiseRes[i](xSource, s)

            newX = MLX.swappedAxes(newX, 2, 1)
            let upsi = ups[i]
            newX = upsi.callAsFunction(newX, conv: { a, b, c, d, e, f, g in
                MLX.convTransposed1d(a, b, stride: c, padding: d, dilation: e, outputPadding: 0, groups: f, stream: g)
            })
            newX = MLX.swappedAxes(newX, 2, 1)

            if i == numUpsamples - 1 {
                newX = reflectionPad(newX)
            }
            newX = newX + xSource

            var xs: MLXArray?
            for j in 0..<numKernels {
                if xs == nil {
                    xs = resBlocks[i * numKernels + j](newX, s)
                } else {
                    let temp = resBlocks[i * numKernels + j](newX, s)
                    xs = xs! + temp
                }
            }
            newX = xs! / numKernels
        }

        newX = LeakyReLU(negativeSlope: 0.01)(newX)

        newX = MLX.swappedAxes(newX, 2, 1)
        newX = convPost(newX, conv: MLX.conv1d)
        newX = MLX.swappedAxes(newX, 2, 1)

        let spec = MLX.exp(newX[0..., 0..<(postNFFt / 2 + 1), 0...])
        let phase = MLX.sin(newX[0..., (postNFFt / 2 + 1)..., 0...])

        spec.eval()
        phase.eval()

        let result = stft.inverse(magnitude: spec, phase: phase)
        result.eval()

        return result
    }
}

// MARK: - Decoder

class Decoder {
    private let encode: AdainResBlk1d
    private var decode: [AdainResBlk1d] = []
    private let F0Conv: ConvWeighted
    private let NConv: ConvWeighted
    private let asrRes: [ConvWeighted]
    private let generator: Generator

    init(
        weights: [String: MLXArray],
        dimIn: Int,
        styleDim: Int,
        dimOut: Int,
        resblockKernelSizes: [Int],
        upsampleRates: [Int],
        upsampleInitialChannel: Int,
        resblockDilationSizes: [[Int]],
        upsampleKernelSizes: [Int],
        genIstftNFft: Int,
        genIstftHopSize: Int
    ) {
        encode = AdainResBlk1d(
            weights: weights,
            weightKeyPrefix: "decoder.encode",
            dimIn: dimIn + 2,
            dimOut: 1024,
            styleDim: styleDim
        )

        decode.append(AdainResBlk1d(
            weights: weights,
            weightKeyPrefix: "decoder.decode.0",
            dimIn: 1024 + 2 + 64,
            dimOut: 1024,
            styleDim: styleDim
        ))
        decode.append(AdainResBlk1d(
            weights: weights,
            weightKeyPrefix: "decoder.decode.1",
            dimIn: 1024 + 2 + 64,
            dimOut: 1024,
            styleDim: styleDim
        ))
        decode.append(AdainResBlk1d(
            weights: weights,
            weightKeyPrefix: "decoder.decode.2",
            dimIn: 1024 + 2 + 64,
            dimOut: 1024,
            styleDim: styleDim
        ))
        decode.append(AdainResBlk1d(
            weights: weights,
            weightKeyPrefix: "decoder.decode.3",
            dimIn: 1024 + 2 + 64,
            dimOut: 512,
            styleDim: styleDim,
            upsample: "true"
        ))

        F0Conv = ConvWeighted(
            weightG: weights["decoder.F0_conv.weight_g"]!,
            weightV: weights["decoder.F0_conv.weight_v"]!,
            bias: weights["decoder.F0_conv.bias"]!,
            stride: 2,
            padding: 1,
            groups: 1
        )
        NConv = ConvWeighted(
            weightG: weights["decoder.N_conv.weight_g"]!,
            weightV: weights["decoder.N_conv.weight_v"]!,
            bias: weights["decoder.N_conv.bias"]!,
            stride: 2,
            padding: 1,
            groups: 1
        )

        asrRes = [ConvWeighted(
            weightG: weights["decoder.asr_res.0.weight_g"]!,
            weightV: weights["decoder.asr_res.0.weight_v"]!,
            bias: weights["decoder.asr_res.0.bias"]!,
            padding: 0
        )]

        generator = Generator(
            weights: weights,
            styleDim: styleDim,
            resblockKernelSizes: resblockKernelSizes,
            upsampleRates: upsampleRates,
            upsampleInitialChannel: upsampleInitialChannel,
            resblockDilationSizes: resblockDilationSizes,
            upsampleKernelSizes: upsampleKernelSizes,
            genIstftNFft: genIstftNFft,
            genIstftHopSize: genIstftHopSize
        )
    }

    func callAsFunction(asr: MLXArray, F0Curve: MLXArray, N: MLXArray, s: MLXArray) -> MLXArray {
        let F0CurveSwapped = MLX.swappedAxes(F0Curve.reshaped([F0Curve.shape[0], 1, F0Curve.shape[1]]), 2, 1)
        let F0 = MLX.swappedAxes(F0Conv(F0CurveSwapped, conv: MLX.conv1d), 2, 1)

        let NSwapped = MLX.swappedAxes(N.reshaped([N.shape[0], 1, N.shape[1]]), 2, 1)
        let NProcessed = MLX.swappedAxes(NConv(NSwapped, conv: MLX.conv1d), 2, 1)

        var x = MLX.concatenated([asr, F0, NProcessed], axis: 1)
        x = encode(x: x, s: s)

        let asrResidual = MLX.swappedAxes(asrRes[0](MLX.swappedAxes(asr, 2, 1), conv: MLX.conv1d), 2, 1)
        var res = true

        x.eval()

        for block in decode {
            if res {
                x = MLX.concatenated([x, asrResidual, F0, NProcessed], axis: 1)
            }
            x = block(x: x, s: s)

            if block.upsampleType != "none" {
                res = false
            }
        }

        x.eval()

        return generator(x, s, F0Curve)
    }
}
