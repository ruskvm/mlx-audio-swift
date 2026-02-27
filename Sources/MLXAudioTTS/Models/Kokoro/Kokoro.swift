//
//  Kokoro.swift
//  MLXAudio
//
//  Ported from mlx-audio Kokoro implementation
//

import Foundation
import Hub
@preconcurrency import MLX
import MLXNN

/// Default HuggingFace repository for Kokoro
public let kokoroDefaultRepo = "prince-canuma/Kokoro-82M"
public let chineseKokoroRepo = "FluidInference/kokoro-82m-v1.1-zh-mlx"

// MARK: - Kokoro TTS Model

/// Main class that encapsulates the whole Kokoro text-to-speech pipeline
public class Kokoro: @unchecked Sendable {

    public enum KokoroError: Error {
        case tooManyTokens
        case sentenceSplitError
        case modelNotInitialized
        case chineseG2PNotInitialized
        case voiceNotLoaded
    }

    // Model components
    private var bert: CustomAlbert!
    private var bertEncoder: Linear!
    private var durationEncoder: DurationEncoder!
    private var predictorLSTM: LSTM!
    private var durationProj: Linear!
    private var prosodyPredictor: ProsodyPredictor!
    private var textEncoder: TextEncoder!
    private var decoder: Decoder!
    private var eSpeakEngine: ESpeakNGEngine!
    private var kokoroTokenizer: KokoroTokenizer!

    private var chosenVoice: KokoroVoice?
    private var voice: MLXArray!

    private var isModelInitialized = false
    private var customURL: URL?

    public let config: KokoroConfiguration

    /// Callback type for streaming audio generation
    public typealias AudioChunkCallback = @Sendable (MLXArray) -> Void

    /// Initializes with default configuration and optional custom model URL
    public init(config: KokoroConfiguration = KokoroConfiguration(), customURL: URL? = nil) {
        self.config = config
        self.customURL = customURL
    }

    /// Download Kokoro model from HuggingFace and initialize
    public static func fromHub(
        repoId: String = kokoroDefaultRepo,
        progressHandler: (@Sendable (Progress) -> Void)? = nil
    ) async throws -> Kokoro {
        print("[Kokoro] Downloading model from \(repoId)...")

        let repo = Hub.Repo(id: repoId)
        let snapshotURL = try await HubApi.shared.snapshot(
            from: repo,
            matching: ["*.safetensors"],
            progressHandler: progressHandler ?? { _ in }
        )

        let modelURL = snapshotURL.appending(path: "kokoro-v1_0.safetensors")
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw KokoroError.modelNotInitialized
        }

        print("[Kokoro] Model downloaded to \(modelURL.path)")

        return Kokoro(customURL: modelURL)
    }

    /// Reset the model to free up memory
    public func resetModel(preserveTextProcessing: Bool = true) {
        bert = nil
        bertEncoder = nil
        durationEncoder = nil
        predictorLSTM = nil
        durationProj = nil
        prosodyPredictor = nil
        textEncoder = nil
        decoder = nil
        voice = nil
        chosenVoice = nil
        isModelInitialized = false

        if !preserveTextProcessing {
            eSpeakEngine = nil
            kokoroTokenizer = nil
        }

        autoreleasepool { }
    }

    /// Ensure model is initialized
    private func ensureModelInitialized() throws {
        if isModelInitialized {
            return
        }

        if eSpeakEngine == nil {
            eSpeakEngine = try ESpeakNGEngine()
        }

        if kokoroTokenizer == nil {
            kokoroTokenizer = KokoroTokenizer(engine: eSpeakEngine)
        }

        autoreleasepool {
            let sanitizedWeights = KokoroWeightLoader.loadWeights(url: self.customURL)

            bert = CustomAlbert(weights: sanitizedWeights, config: config.albertConfig)
            bertEncoder = Linear(weight: sanitizedWeights["bert_encoder.weight"]!, bias: sanitizedWeights["bert_encoder.bias"]!)
            durationEncoder = DurationEncoder(weights: sanitizedWeights, dModel: config.dModel, styDim: config.styleDim, nlayers: config.predictorNLayers)

            predictorLSTM = LSTM(
                inputSize: config.dModel + config.styleDim,
                hiddenSize: config.dModel / 2,
                wxForward: sanitizedWeights["predictor.lstm.weight_ih_l0"]!,
                whForward: sanitizedWeights["predictor.lstm.weight_hh_l0"]!,
                biasIhForward: sanitizedWeights["predictor.lstm.bias_ih_l0"]!,
                biasHhForward: sanitizedWeights["predictor.lstm.bias_hh_l0"]!,
                wxBackward: sanitizedWeights["predictor.lstm.weight_ih_l0_reverse"]!,
                whBackward: sanitizedWeights["predictor.lstm.weight_hh_l0_reverse"]!,
                biasIhBackward: sanitizedWeights["predictor.lstm.bias_ih_l0_reverse"]!,
                biasHhBackward: sanitizedWeights["predictor.lstm.bias_hh_l0_reverse"]!
            )

            durationProj = Linear(
                weight: sanitizedWeights["predictor.duration_proj.linear_layer.weight"]!,
                bias: sanitizedWeights["predictor.duration_proj.linear_layer.bias"]!
            )

            prosodyPredictor = ProsodyPredictor(
                weights: sanitizedWeights,
                styleDim: config.styleDim,
                dHid: config.dModel
            )

            textEncoder = TextEncoder(
                weights: sanitizedWeights,
                channels: config.channels,
                kernelSize: config.kernelSize,
                depth: config.depth,
                nSymbols: config.nSymbols
            )

            decoder = Decoder(
                weights: sanitizedWeights,
                dimIn: config.decoderConfig.dimIn,
                styleDim: config.decoderConfig.styleDim,
                dimOut: config.decoderConfig.dimOut,
                resblockKernelSizes: config.decoderConfig.resblockKernelSizes,
                upsampleRates: config.decoderConfig.upsampleRates,
                upsampleInitialChannel: config.decoderConfig.upsampleInitialChannel,
                resblockDilationSizes: config.decoderConfig.resblockDilationSizes,
                upsampleKernelSizes: config.decoderConfig.upsampleKernelSizes,
                genIstftNFft: config.decoderConfig.genIstftNFft,
                genIstftHopSize: config.decoderConfig.genIstftHopSize
            )
        }

        isModelInitialized = true
    }

    /// Generate audio for a single sentence
    public func generateAudioForSentence(voice: KokoroVoice, text: String, speed: Float = 1.0) throws -> MLXArray {
        try ensureModelInitialized()

        if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return MLXArray.zeros([1])
        }

        return try autoreleasepool { () -> MLXArray in
            if chosenVoice != voice {
                autoreleasepool {
                    self.voice = VoiceLoader.loadVoice(voice)
                    self.voice?.eval()
                }

                if !voice.isChinese {
                    try kokoroTokenizer.setLanguage(for: voice)
                }
                chosenVoice = voice
            }

            do {
                let phonemizedResult = try kokoroTokenizer.phonemize(text)
                let inputIds = PhonemeTokenizer.tokenize(phonemizedText: phonemizedResult.phonemes)

                guard inputIds.count <= config.maxTokenCount else {
                    throw KokoroError.tooManyTokens
                }

                return try self.processTokensToAudio(inputIds: inputIds, speed: speed)
            } catch {
                var errorAudioData = [Float](repeating: 0.0, count: 4800)
                for i in 0..<4800 {
                    let t = Float(i) / Float(config.sampleRate)
                    let freq: Float = 880.0
                    errorAudioData[i] = sin(Float(2.0 * .pi * freq) * t) * 0.3
                }
                return MLXArray(errorAudioData)
            }
        }
    }

    /// Generate audio with streaming chunks
    public func generateAudio(voice: KokoroVoice, text: String, speed: Float = 1.0, chunkCallback: @escaping @Sendable AudioChunkCallback) throws {
        try ensureModelInitialized()

        let sentences = SentenceTokenizer.splitIntoSentences(text: text)
        if sentences.isEmpty {
            throw KokoroError.sentenceSplitError
        }

        // Capture model reference for Task.detached - model is @unchecked Sendable
        let model = self
        Task.detached { @Sendable in
            await MainActor.run { model.voice = nil }

            for sentence in sentences {
                autoreleasepool {
                    do {
                        let audio = try model.generateAudioForSentence(voice: voice, text: sentence, speed: speed)
                        audio.eval()

                        // Callback is @Sendable so can be called directly
                        chunkCallback(audio)

                        autoreleasepool {
                            _ = audio
                        }
                    } catch {
                        // Handle error silently
                    }
                }
                Memory.clearCache()
            }

            if sentences.count > 5 {
                try? await Task.sleep(for: .seconds(2))
                model.resetModel()
            }
        }
    }

    /// Process tokens to audio
    private func processTokensToAudio(inputIds: [Int], speed: Float) throws -> MLXArray {
        return try generateAudioForTokens(inputIds: inputIds, speed: speed)
    }

    /// Generate audio from token IDs
    private func generateAudioForTokens(inputIds: [Int], speed: Float) throws -> MLXArray {
        return try autoreleasepool { () -> MLXArray in
            try autoreleasepool {
                let paddedInputIdsBase = [0] + inputIds + [0]
                let paddedInputIds = MLXArray(paddedInputIdsBase).expandedDimensions(axes: [0])
                paddedInputIds.eval()

                let inputLengths = MLXArray(paddedInputIds.dim(-1))
                inputLengths.eval()

                let inputLengthMax: Int = MLX.max(inputLengths).item()
                var textMask = MLXArray(0..<inputLengthMax)
                textMask.eval()

                textMask = textMask + 1 .> inputLengths
                textMask.eval()

                textMask = textMask.expandedDimensions(axes: [0])
                textMask.eval()

                let swiftTextMask: [Bool] = textMask.asArray(Bool.self)
                let swiftTextMaskInt = swiftTextMask.map { !$0 ? 1 : 0 }
                let attentionMask = MLXArray(swiftTextMaskInt).reshaped(textMask.shape)
                attentionMask.eval()

                return try autoreleasepool { () -> MLXArray in
                    guard let bert = bert, let bertEncoder = bertEncoder else {
                        throw KokoroError.modelNotInitialized
                    }

                    let (bertDur, _) = bert(paddedInputIds, attentionMask: attentionMask)
                    bertDur.eval()

                    autoreleasepool { _ = attentionMask }

                    let dEn = bertEncoder(bertDur).transposed(0, 2, 1)
                    dEn.eval()

                    autoreleasepool { _ = bertDur }

                    var refS: MLXArray
                    do {
                        guard let voice = voice else {
                            throw KokoroError.voiceNotLoaded
                        }
                        refS = voice[min(inputIds.count - 1, voice.shape[0] - 1), 0...1, 0...]
                    } catch {
                        guard let voice = voice else {
                            throw KokoroError.voiceNotLoaded
                        }
                        refS = voice[0, 0...1, 0...]
                    }
                    refS.eval()

                    let s = refS[0...1, 128...]
                    s.eval()

                    return try autoreleasepool { () -> MLXArray in
                        guard let durationEncoder = durationEncoder,
                              let predictorLSTM = predictorLSTM,
                              let durationProj = durationProj
                        else {
                            throw KokoroError.modelNotInitialized
                        }

                        let d = durationEncoder(dEn, style: s, textLengths: inputLengths, m: textMask)
                        d.eval()

                        autoreleasepool {
                            _ = dEn
                            _ = textMask
                        }

                        let (x, _) = predictorLSTM(d)
                        x.eval()

                        let duration = durationProj(x)
                        duration.eval()

                        autoreleasepool { _ = x }

                        let durationSigmoid = MLX.sigmoid(duration).sum(axis: -1) / speed
                        durationSigmoid.eval()

                        autoreleasepool { _ = duration }

                        let predDur = MLX.clip(durationSigmoid.round(), min: 1).asType(.int32)[0]
                        predDur.eval()

                        autoreleasepool { _ = durationSigmoid }

                        var allIndices: [MLXArray] = []
                        let chunkSize = 50

                        for startIdx in stride(from: 0, to: predDur.shape[0], by: chunkSize) {
                            autoreleasepool {
                                let endIdx = min(startIdx + chunkSize, predDur.shape[0])
                                let chunkIndices = predDur[startIdx..<endIdx]

                                let indices = MLX.concatenated(
                                    chunkIndices.enumerated().map { i, n in
                                        let nSize: Int = n.item()
                                        let arrayIndex = MLXArray([i + startIdx])
                                        arrayIndex.eval()
                                        let repeated = MLX.repeated(arrayIndex, count: nSize)
                                        repeated.eval()
                                        return repeated
                                    }
                                )
                                indices.eval()
                                allIndices.append(indices)
                            }
                        }

                        let indices = MLX.concatenated(allIndices)
                        indices.eval()

                        allIndices.removeAll()

                        let indicesShape = indices.shape[0]
                        let inputIdsShape = paddedInputIds.shape[1]

                        var rowIndices: [Int] = []
                        var colIndices: [Int] = []
                        var values: [Float] = []

                        let estimatedNonZeros = min(indicesShape, inputIdsShape * 5)
                        rowIndices.reserveCapacity(estimatedNonZeros)
                        colIndices.reserveCapacity(estimatedNonZeros)
                        values.reserveCapacity(estimatedNonZeros)

                        let batchSize = 256
                        for startIdx in stride(from: 0, to: indicesShape, by: batchSize) {
                            autoreleasepool {
                                let endIdx = min(startIdx + batchSize, indicesShape)
                                for i in startIdx..<endIdx {
                                    let indiceValue: Int = indices[i].item()
                                    if indiceValue < inputIdsShape {
                                        rowIndices.append(indiceValue)
                                        colIndices.append(i)
                                        values.append(1.0)
                                    }
                                }
                            }
                        }

                        autoreleasepool {
                            _ = indices
                            _ = predDur
                        }

                        var swiftPredAlnTrg = [Float](repeating: 0.0, count: inputIdsShape * indicesShape)
                        let matrixBatchSize = 1000
                        for startIdx in stride(from: 0, to: rowIndices.count, by: matrixBatchSize) {
                            autoreleasepool {
                                let endIdx = min(startIdx + matrixBatchSize, rowIndices.count)
                                for i in startIdx..<endIdx {
                                    let row = rowIndices[i]
                                    let col = colIndices[i]
                                    if row < inputIdsShape && col < indicesShape {
                                        swiftPredAlnTrg[row * indicesShape + col] = 1.0
                                    }
                                }
                            }
                        }

                        let predAlnTrg = MLXArray(swiftPredAlnTrg).reshaped([inputIdsShape, indicesShape])
                        predAlnTrg.eval()

                        swiftPredAlnTrg = []

                        autoreleasepool {
                            rowIndices = []
                            colIndices = []
                            values = []
                        }

                        let predAlnTrgBatched = predAlnTrg.expandedDimensions(axis: 0)
                        predAlnTrgBatched.eval()

                        let en = d.transposed(0, 2, 1).matmul(predAlnTrgBatched)
                        en.eval()

                        autoreleasepool {
                            _ = d
                            _ = predAlnTrgBatched
                        }

                        return try autoreleasepool { () -> MLXArray in
                            guard let prosodyPredictor = prosodyPredictor,
                                  let textEncoder = textEncoder,
                                  let decoder = decoder
                            else {
                                throw KokoroError.modelNotInitialized
                            }

                            let (F0Pred, NPred) = prosodyPredictor.F0NTrain(x: en, s: s)
                            F0Pred.eval()
                            NPred.eval()

                            autoreleasepool { _ = en }

                            let tEn = textEncoder(paddedInputIds, inputLengths: inputLengths, m: textMask)
                            tEn.eval()

                            autoreleasepool {
                                _ = paddedInputIds
                                _ = inputLengths
                            }

                            let asr = MLX.matmul(tEn, predAlnTrg)
                            asr.eval()

                            autoreleasepool {
                                _ = tEn
                                _ = predAlnTrg
                            }

                            let voiceS = refS[0...1, 0...127]
                            voiceS.eval()

                            autoreleasepool { _ = refS }

                            let audio = decoder(asr: asr, F0Curve: F0Pred, N: NPred, s: voiceS)[0]
                            audio.eval()

                            autoreleasepool {
                                _ = asr
                                _ = F0Pred
                                _ = NPred
                                _ = voiceS
                                _ = s
                            }

                            let audioShape = audio.shape

                            let totalSamples: Int
                            if audioShape.count == 1 {
                                totalSamples = audioShape[0]
                            } else if audioShape.count == 2 {
                                totalSamples = audioShape[1]
                            } else {
                                totalSamples = 0
                            }

                            if totalSamples <= 1 {
                                var errorAudioData = [Float](repeating: 0.0, count: config.sampleRate)

                                for i in stride(from: 0, to: config.sampleRate, by: 100) {
                                    let endIdx = min(i + 100, config.sampleRate)
                                    for j in i..<endIdx {
                                        let t = Float(j) / Float(config.sampleRate)
                                        let freq = (Int(t * 2) % 2 == 0) ? 500.0 : 800.0
                                        errorAudioData[j] = sin(Float(2.0 * .pi * freq) * t) * 0.5
                                    }
                                }

                                let fallbackAudio = MLXArray(errorAudioData)
                                fallbackAudio.eval()
                                return fallbackAudio
                            }

                            return audio
                        }
                    }
                }
            }
        }
    }
}

// MARK: - DurationEncoder

class DurationEncoder {
    var lstms: [Module] = []

    init(weights: [String: MLXArray], dModel: Int, styDim: Int, nlayers: Int) {
        for i in 0..<nlayers {
            if i % 2 == 0 {
                lstms.append(
                    LSTM(
                        inputSize: dModel + styDim,
                        hiddenSize: dModel / 2,
                        wxForward: weights["predictor.text_encoder.lstms.\(i).weight_ih_l0"]!,
                        whForward: weights["predictor.text_encoder.lstms.\(i).weight_hh_l0"]!,
                        biasIhForward: weights["predictor.text_encoder.lstms.\(i).bias_ih_l0"]!,
                        biasHhForward: weights["predictor.text_encoder.lstms.\(i).bias_hh_l0"]!,
                        wxBackward: weights["predictor.text_encoder.lstms.\(i).weight_ih_l0_reverse"]!,
                        whBackward: weights["predictor.text_encoder.lstms.\(i).weight_hh_l0_reverse"]!,
                        biasIhBackward: weights["predictor.text_encoder.lstms.\(i).bias_ih_l0_reverse"]!,
                        biasHhBackward: weights["predictor.text_encoder.lstms.\(i).bias_hh_l0_reverse"]!
                    )
                )
            } else {
                lstms.append(
                    AdaLayerNorm(
                        weight: weights["predictor.text_encoder.lstms.\(i).fc.weight"]!,
                        bias: weights["predictor.text_encoder.lstms.\(i).fc.bias"]!
                    )
                )
            }
        }
    }

    func callAsFunction(_ x: MLXArray, style: MLXArray, textLengths: MLXArray, m: MLXArray) -> MLXArray {
        var x = x.transposed(2, 0, 1)
        let s = MLX.broadcast(style, to: [x.shape[0], x.shape[1], style.shape[style.shape.count - 1]])
        x = MLX.concatenated([x, s], axis: -1)
        x = MLX.where(m.expandedDimensions(axes: [-1]).transposed(1, 0, 2), MLXArray.zeros(like: x), x)
        x = x.transposed(1, 2, 0)

        for block in lstms {
            if let adaLayerNorm = block as? AdaLayerNorm {
                x = adaLayerNorm(x.transposed(0, 2, 1), style).transposed(0, 2, 1)
                x = MLX.concatenated([x, s.transposed(1, 2, 0)], axis: 1)
                x = MLX.where(m.expandedDimensions(axes: [-1]).transposed(0, 2, 1), MLXArray.zeros(like: x), x)
            } else if let lstm = block as? LSTM {
                x = x.transposed(0, 2, 1)[0]
                let (lstmOutput, _) = lstm(x)
                x = lstmOutput.transposed(0, 2, 1)
                let xPad = MLXArray.zeros([x.shape[0], x.shape[1], m.shape[m.shape.count - 1]])
                xPad[0..<x.shape[0], 0..<x.shape[1], 0..<x.shape[2]] = x
                x = xPad
            }
        }

        return x.transposed(0, 2, 1)
    }
}

// MARK: - TextEncoder

class TextEncoder {
    let embedding: Embedding
    let cnn: [[Module]]
    let lstm: LSTM

    init(weights: [String: MLXArray], channels: Int, kernelSize: Int, depth: Int, nSymbols: Int, actv: Module = LeakyReLU(negativeSlope: 0.2)) {
        embedding = Embedding(weight: weights["text_encoder.embedding.weight"]!)
        let padding = (kernelSize - 1) / 2

        var cnnLayers: [[Module]] = []
        for i in 0..<depth {
            cnnLayers.append([
                ConvWeighted(
                    weightG: weights["text_encoder.cnn.\(i).0.weight_g"]!,
                    weightV: weights["text_encoder.cnn.\(i).0.weight_v"]!,
                    bias: weights["text_encoder.cnn.\(i).0.bias"]!,
                    padding: padding
                ),
                LayerNormInference(
                    weight: weights["text_encoder.cnn.\(i).1.gamma"]!,
                    bias: weights["text_encoder.cnn.\(i).1.beta"]!
                ),
                actv,
            ])
        }
        cnn = cnnLayers

        lstm = LSTM(
            inputSize: channels,
            hiddenSize: channels / 2,
            wxForward: weights["text_encoder.lstm.weight_ih_l0"]!,
            whForward: weights["text_encoder.lstm.weight_hh_l0"]!,
            biasIhForward: weights["text_encoder.lstm.bias_ih_l0"]!,
            biasHhForward: weights["text_encoder.lstm.bias_hh_l0"]!,
            wxBackward: weights["text_encoder.lstm.weight_ih_l0_reverse"]!,
            whBackward: weights["text_encoder.lstm.weight_hh_l0_reverse"]!,
            biasIhBackward: weights["text_encoder.lstm.bias_ih_l0_reverse"]!,
            biasHhBackward: weights["text_encoder.lstm.bias_hh_l0_reverse"]!
        )
    }

    public func callAsFunction(_ x: MLXArray, inputLengths: MLXArray, m: MLXArray) -> MLXArray {
        var features = embedding(x)
        features = features.transposed(0, 2, 1)
        let mask = m.expandedDimensions(axis: 1)
        features = MLX.where(mask, 0.0, features)

        for convBlock in cnn {
            for layer in convBlock {
                if layer is ConvWeighted || layer is LayerNormInference {
                    features = MLX.swappedAxes(features, 2, 1)
                    if let conv = layer as? ConvWeighted {
                        features = conv(features, conv: MLX.conv1d)
                    } else if let norm = layer as? LayerNormInference {
                        features = norm(features)
                    }
                    features = MLX.swappedAxes(features, 2, 1)
                } else if let activation = layer as? LeakyReLU {
                    features = activation(features)
                } else {
                    fatalError("Unsupported layer type")
                }
                features = MLX.where(mask, 0.0, features)
            }
        }

        features = MLX.swappedAxes(features, 2, 1)
        let (lstmOutput, _) = lstm(features)
        features = MLX.swappedAxes(lstmOutput, 2, 1)

        return MLX.where(mask, 0.0, features)
    }
}

// MARK: - ProsodyPredictor

class ProsodyPredictor {
    let shared: LSTM
    let F0: [AdainResBlk1d]
    let N: [AdainResBlk1d]
    let F0Proj: Conv1dInference
    let NProj: Conv1dInference

    public init(weights: [String: MLXArray], styleDim: Int, dHid: Int) {
        shared = LSTM(
            inputSize: dHid + styleDim,
            hiddenSize: dHid / 2,
            wxForward: weights["predictor.shared.weight_ih_l0"]!,
            whForward: weights["predictor.shared.weight_hh_l0"]!,
            biasIhForward: weights["predictor.shared.bias_ih_l0"]!,
            biasHhForward: weights["predictor.shared.bias_hh_l0"]!,
            wxBackward: weights["predictor.shared.weight_ih_l0_reverse"]!,
            whBackward: weights["predictor.shared.weight_hh_l0_reverse"]!,
            biasIhBackward: weights["predictor.shared.bias_ih_l0_reverse"]!,
            biasHhBackward: weights["predictor.shared.bias_hh_l0_reverse"]!
        )

        F0 = [
            AdainResBlk1d(weights: weights, weightKeyPrefix: "predictor.F0.0", dimIn: dHid, dimOut: dHid, styleDim: styleDim),
            AdainResBlk1d(weights: weights, weightKeyPrefix: "predictor.F0.1", dimIn: dHid, dimOut: dHid / 2, styleDim: styleDim, upsample: "true"),
            AdainResBlk1d(weights: weights, weightKeyPrefix: "predictor.F0.2", dimIn: dHid / 2, dimOut: dHid / 2, styleDim: styleDim),
        ]

        N = [
            AdainResBlk1d(weights: weights, weightKeyPrefix: "predictor.N.0", dimIn: dHid, dimOut: dHid, styleDim: styleDim),
            AdainResBlk1d(weights: weights, weightKeyPrefix: "predictor.N.1", dimIn: dHid, dimOut: dHid / 2, styleDim: styleDim, upsample: "true"),
            AdainResBlk1d(weights: weights, weightKeyPrefix: "predictor.N.2", dimIn: dHid / 2, dimOut: dHid / 2, styleDim: styleDim),
        ]

        F0Proj = Conv1dInference(
            inputChannels: dHid / 2,
            outputChannels: 1,
            kernelSize: 1,
            padding: 0,
            weight: weights["predictor.F0_proj.weight"]!,
            bias: weights["predictor.F0_proj.bias"]!
        )

        NProj = Conv1dInference(
            inputChannels: dHid / 2,
            outputChannels: 1,
            kernelSize: 1,
            padding: 0,
            weight: weights["predictor.N_proj.weight"]!,
            bias: weights["predictor.N_proj.bias"]!
        )
    }

    func F0NTrain(x: MLXArray, s: MLXArray) -> (MLXArray, MLXArray) {
        let (x1, _) = shared(x.transposed(0, 2, 1))

        var F0Val = x1.transposed(0, 2, 1)
        for block in F0 {
            F0Val = block(x: F0Val, s: s)
        }
        F0Val = MLX.swappedAxes(F0Val, 2, 1)
        F0Val = F0Proj(F0Val)
        F0Val = MLX.swappedAxes(F0Val, 2, 1)

        var NVal = x1.transposed(0, 2, 1)
        for block in N {
            NVal = block(x: NVal, s: s)
        }
        NVal = MLX.swappedAxes(NVal, 2, 1)
        NVal = NProj(NVal)
        NVal = MLX.swappedAxes(NVal, 2, 1)

        return (F0Val.squeezed(axis: 1), NVal.squeezed(axis: 1))
    }
}

// MARK: - VoiceLoader

class VoiceLoader {
    private init() {}

    static var availableVoices: [KokoroVoice] {
        Array(KokoroVoice.allCases)
    }

    static func loadVoice(_ voice: KokoroVoice) -> MLXArray {
        let file = voice.fileName
        let filePath = Bundle.module.path(forResource: file, ofType: "json")!
        return try! read3DArrayFromJson(file: filePath, shape: [510, 1, 256])!
    }

    private static func read3DArrayFromJson(file: String, shape: [Int]) throws -> MLXArray? {
        guard shape.count == 3 else { return nil }

        let data = try Data(contentsOf: URL(fileURLWithPath: file))
        let jsonObject = try JSONSerialization.jsonObject(with: data, options: [])

        var aa = Array(repeating: Float(0.0), count: shape[0] * shape[1] * shape[2])
        var aaIndex = 0

        if let nestedArray = jsonObject as? [[[Any]]] {
            guard nestedArray.count == shape[0] else { return nil }
            for a in 0..<nestedArray.count {
                guard nestedArray[a].count == shape[1] else { return nil }
                for b in 0..<nestedArray[a].count {
                    guard nestedArray[a][b].count == shape[2] else { return nil }
                    for c in 0..<nestedArray[a][b].count {
                        if let n = nestedArray[a][b][c] as? Double {
                            aa[aaIndex] = Float(n)
                            aaIndex += 1
                        } else {
                            fatalError("Cannot load value \(a), \(b), \(c) as double")
                        }
                    }
                }
            }
        } else {
            return nil
        }

        guard aaIndex == shape[0] * shape[1] * shape[2] else {
            fatalError("Mismatch in array size: \(aaIndex) vs \(shape[0] * shape[1] * shape[2])")
        }

        return MLXArray(aa).reshaped(shape)
    }
}

// MARK: - KokoroWeightLoader

class KokoroWeightLoader {
    private init() {}

    static func loadWeights(url: URL? = nil) -> [String: MLXArray] {
        let modelURL = url ?? {
            let filePath = Bundle.module.path(forResource: "kokoro-v1_0", ofType: "safetensors")!
            return URL(fileURLWithPath: filePath)
        }()

        do {
            let weights = try MLX.loadArrays(url: modelURL)
            var sanitizedWeights: [String: MLXArray] = [:]

            for (key, value) in weights {
                if key.hasPrefix("bert") {
                    if key.contains("position_ids") {
                        continue
                    }
                    sanitizedWeights[key] = value
                } else if key.hasPrefix("predictor") {
                    if key.contains("F0_proj.weight") {
                        sanitizedWeights[key] = value.transposed(0, 2, 1)
                    } else if key.contains("N_proj.weight") {
                        sanitizedWeights[key] = value.transposed(0, 2, 1)
                    } else if key.contains("weight_v") {
                        if checkArrayShape(arr: value) {
                            sanitizedWeights[key] = value
                        } else {
                            sanitizedWeights[key] = value.transposed(0, 2, 1)
                        }
                    } else {
                        sanitizedWeights[key] = value
                    }
                } else if key.hasPrefix("text_encoder") {
                    if key.contains("weight_v") {
                        if checkArrayShape(arr: value) {
                            sanitizedWeights[key] = value
                        } else {
                            sanitizedWeights[key] = value.transposed(0, 2, 1)
                        }
                    } else {
                        sanitizedWeights[key] = value
                    }
                } else if key.hasPrefix("decoder") {
                    if key.contains("noise_convs"), key.hasSuffix(".weight") {
                        sanitizedWeights[key] = value.transposed(0, 2, 1)
                    } else if key.contains("weight_v") {
                        if checkArrayShape(arr: value) {
                            sanitizedWeights[key] = value
                        } else {
                            sanitizedWeights[key] = value.transposed(0, 2, 1)
                        }
                    } else {
                        sanitizedWeights[key] = value
                    }
                }
            }

            return sanitizedWeights
        } catch {
            print("Kokoro: Error loading weights: \(error)")
            return [:]
        }
    }

    private static func checkArrayShape(arr: MLXArray) -> Bool {
        guard arr.shape.count != 3 else { return false }

        let outChannels = arr.shape[0]
        let kH = arr.shape[1]
        let kW = arr.shape[2]

        return (outChannels >= kH) && (outChannels >= kW) && (kH == kW)
    }
}
