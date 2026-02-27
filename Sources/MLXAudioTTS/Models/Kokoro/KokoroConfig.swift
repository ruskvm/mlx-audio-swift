//
//  KokoroConfig.swift
//  MLXAudio
//
//  Ported from mlx-audio Kokoro implementation
//

import Foundation

// MARK: - Albert Configuration

/// Configuration for the ALBERT model used in Kokoro TTS
public struct AlbertModelArgs: Codable, Sendable {
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var maxPositionEmbeddings: Int
    public var modelType: String
    public var embeddingSize: Int
    public var innerGroupNum: Int
    public var numHiddenGroups: Int
    public var hiddenDropoutProb: Float
    public var attentionProbsDropoutProb: Float
    public var typeVocabSize: Int
    public var initializerRange: Float
    public var layerNormEps: Float
    public var vocabSize: Int
    public var dropout: Float

    enum CodingKeys: String, CodingKey {
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case modelType = "model_type"
        case embeddingSize = "embedding_size"
        case innerGroupNum = "inner_group_num"
        case numHiddenGroups = "num_hidden_groups"
        case hiddenDropoutProb = "hidden_dropout_prob"
        case attentionProbsDropoutProb = "attention_probs_dropout_prob"
        case typeVocabSize = "type_vocab_size"
        case initializerRange = "initializer_range"
        case layerNormEps = "layer_norm_eps"
        case vocabSize = "vocab_size"
        case dropout
    }

    public init(
        numHiddenLayers: Int = 12,
        numAttentionHeads: Int = 12,
        hiddenSize: Int = 768,
        intermediateSize: Int = 2048,
        maxPositionEmbeddings: Int = 512,
        modelType: String = "albert",
        embeddingSize: Int = 128,
        innerGroupNum: Int = 1,
        numHiddenGroups: Int = 1,
        hiddenDropoutProb: Float = 0.1,
        attentionProbsDropoutProb: Float = 0.1,
        typeVocabSize: Int = 2,
        initializerRange: Float = 0.02,
        layerNormEps: Float = 1e-12,
        vocabSize: Int = 178,
        dropout: Float = 0.1
    ) {
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.modelType = modelType
        self.embeddingSize = embeddingSize
        self.innerGroupNum = innerGroupNum
        self.numHiddenGroups = numHiddenGroups
        self.hiddenDropoutProb = hiddenDropoutProb
        self.attentionProbsDropoutProb = attentionProbsDropoutProb
        self.typeVocabSize = typeVocabSize
        self.initializerRange = initializerRange
        self.layerNormEps = layerNormEps
        self.vocabSize = vocabSize
        self.dropout = dropout
    }
}

// MARK: - Decoder Configuration

/// Configuration for the Kokoro decoder
public struct KokoroDecoderConfig: Codable, Sendable {
    public var dimIn: Int
    public var styleDim: Int
    public var dimOut: Int
    public var resblockKernelSizes: [Int]
    public var upsampleRates: [Int]
    public var upsampleInitialChannel: Int
    public var resblockDilationSizes: [[Int]]
    public var upsampleKernelSizes: [Int]
    public var genIstftNFft: Int
    public var genIstftHopSize: Int

    enum CodingKeys: String, CodingKey {
        case dimIn = "dim_in"
        case styleDim = "style_dim"
        case dimOut = "dim_out"
        case resblockKernelSizes = "resblock_kernel_sizes"
        case upsampleRates = "upsample_rates"
        case upsampleInitialChannel = "upsample_initial_channel"
        case resblockDilationSizes = "resblock_dilation_sizes"
        case upsampleKernelSizes = "upsample_kernel_sizes"
        case genIstftNFft = "gen_istft_n_fft"
        case genIstftHopSize = "gen_istft_hop_size"
    }

    public init(
        dimIn: Int = 512,
        styleDim: Int = 128,
        dimOut: Int = 80,
        resblockKernelSizes: [Int] = [3, 7, 11],
        upsampleRates: [Int] = [10, 6],
        upsampleInitialChannel: Int = 512,
        resblockDilationSizes: [[Int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsampleKernelSizes: [Int] = [20, 12],
        genIstftNFft: Int = 20,
        genIstftHopSize: Int = 5
    ) {
        self.dimIn = dimIn
        self.styleDim = styleDim
        self.dimOut = dimOut
        self.resblockKernelSizes = resblockKernelSizes
        self.upsampleRates = upsampleRates
        self.upsampleInitialChannel = upsampleInitialChannel
        self.resblockDilationSizes = resblockDilationSizes
        self.upsampleKernelSizes = upsampleKernelSizes
        self.genIstftNFft = genIstftNFft
        self.genIstftHopSize = genIstftHopSize
    }
}

// MARK: - Main Configuration

/// Main configuration for Kokoro TTS model
public struct KokoroConfiguration: Codable, Sendable {
    // Model parameters
    public var channels: Int
    public var kernelSize: Int
    public var depth: Int
    public var nSymbols: Int
    public var dModel: Int
    public var styleDim: Int
    public var predictorNLayers: Int

    // Albert config
    public var albertConfig: AlbertModelArgs

    // Decoder config
    public var decoderConfig: KokoroDecoderConfig

    // Audio config
    public var sampleRate: Int
    public var maxTokenCount: Int

    enum CodingKeys: String, CodingKey {
        case channels
        case kernelSize = "kernel_size"
        case depth
        case nSymbols = "n_symbols"
        case dModel = "d_model"
        case styleDim = "style_dim"
        case predictorNLayers = "predictor_n_layers"
        case albertConfig = "albert_config"
        case decoderConfig = "decoder_config"
        case sampleRate = "sample_rate"
        case maxTokenCount = "max_token_count"
    }

    public init(
        channels: Int = 512,
        kernelSize: Int = 5,
        depth: Int = 3,
        nSymbols: Int = 178,
        dModel: Int = 512,
        styleDim: Int = 128,
        predictorNLayers: Int = 6,
        albertConfig: AlbertModelArgs = AlbertModelArgs(),
        decoderConfig: KokoroDecoderConfig = KokoroDecoderConfig(),
        sampleRate: Int = 24000,
        maxTokenCount: Int = 510
    ) {
        self.channels = channels
        self.kernelSize = kernelSize
        self.depth = depth
        self.nSymbols = nSymbols
        self.dModel = dModel
        self.styleDim = styleDim
        self.predictorNLayers = predictorNLayers
        self.albertConfig = albertConfig
        self.decoderConfig = decoderConfig
        self.sampleRate = sampleRate
        self.maxTokenCount = maxTokenCount
    }
}

// MARK: - Voice Enum

/// Available voices for Kokoro TTS
public enum KokoroVoice: String, CaseIterable, Sendable {
    case afAlloy
    case afAoede
    case afBella
    case afHeart
    case afJessica
    case afKore
    case afNicole
    case afNova
    case afRiver
    case afSarah
    case afSky
    case amAdam
    case amEcho
    case amEric
    case amFenrir
    case amLiam
    case amMichael
    case amOnyx
    case amPuck
    case amSanta
    case bfAlice
    case bfEmma
    case bfIsabella
    case bfLily
    case bmDaniel
    case bmFable
    case bmGeorge
    case bmLewis
    case efDora
    case emAlex
    case ffSiwis
    case hfAlpha
    case hfBeta
    case hfOmega
    case hmPsi
    case ifSara
    case imNicola
    case jfAlpha
    case jfGongitsune
    case jfNezumi
    case jfTebukuro
    case jmKumo
    case pfDora
    case pmSanta
    case zfXiaobei
    case zfXiaoni
    case zfXiaoxiao
    case zfXiaoyi
    case zmYunjian
    case zmYunxi
    case zmYunxia
    case zmYunyang

    /// Check if this voice is Chinese (Mandarin)
    public var isChinese: Bool {
        switch self {
        case .zfXiaobei, .zfXiaoni, .zfXiaoxiao, .zfXiaoyi,
             .zmYunjian, .zmYunxi, .zmYunxia, .zmYunyang:
            return true
        default:
            return false
        }
    }

    /// Get the voice file name
    public var fileName: String {
        switch self {
        case .afAlloy: return "af_alloy"
        case .afAoede: return "af_aoede"
        case .afBella: return "af_bella"
        case .afHeart: return "af_heart"
        case .afJessica: return "af_jessica"
        case .afKore: return "af_kore"
        case .afNicole: return "af_nicole"
        case .afNova: return "af_nova"
        case .afRiver: return "af_river"
        case .afSarah: return "af_sarah"
        case .afSky: return "af_sky"
        case .amAdam: return "am_adam"
        case .amEcho: return "am_echo"
        case .amEric: return "am_eric"
        case .amFenrir: return "am_fenrir"
        case .amLiam: return "am_liam"
        case .amMichael: return "am_michael"
        case .amOnyx: return "am_onyx"
        case .amPuck: return "am_puck"
        case .amSanta: return "am_santa"
        case .bfAlice: return "bf_alice"
        case .bfEmma: return "bf_emma"
        case .bfIsabella: return "bf_isabella"
        case .bfLily: return "bf_lily"
        case .bmDaniel: return "bm_daniel"
        case .bmFable: return "bm_fable"
        case .bmGeorge: return "bm_george"
        case .bmLewis: return "bm_lewis"
        case .efDora: return "ef_dora"
        case .emAlex: return "em_alex"
        case .ffSiwis: return "ff_siwis"
        case .hfAlpha: return "hf_alpha"
        case .hfBeta: return "hf_beta"
        case .hfOmega: return "hm_omega"
        case .hmPsi: return "hm_psi"
        case .ifSara: return "if_sara"
        case .imNicola: return "im_nicola"
        case .jfAlpha: return "jf_alpha"
        case .jfGongitsune: return "jf_gongitsune"
        case .jfNezumi: return "jf_nezumi"
        case .jfTebukuro: return "jf_tebukuro"
        case .jmKumo: return "jm_kumo"
        case .pfDora: return "pf_dora"
        case .pmSanta: return "pm_santa"
        case .zfXiaobei: return "zf_xiaobei"
        case .zfXiaoni: return "zf_xiaoni"
        case .zfXiaoxiao: return "zf_xiaoxiao"
        case .zfXiaoyi: return "zf_xiaoyi"
        case .zmYunjian: return "zm_yunjian"
        case .zmYunxi: return "zm_yunxi"
        case .zmYunxia: return "zm_yunxia"
        case .zmYunyang: return "zm_yunyang"
        }
    }

    /// Get the language dialect for this voice
    public var languageDialect: LanguageDialect {
        switch self {
        case .afAlloy, .afAoede, .afBella, .afHeart, .afJessica, .afKore, .afNicole, .afNova, .afRiver, .afSarah, .afSky,
             .amAdam, .amEcho, .amEric, .amFenrir, .amLiam, .amMichael, .amOnyx, .amPuck, .amSanta:
            return .enUS
        case .bfAlice, .bfEmma, .bfIsabella, .bfLily, .bmDaniel, .bmFable, .bmGeorge, .bmLewis:
            return .enGB
        case .efDora, .emAlex:
            return .esES
        case .ffSiwis:
            return .frFR
        case .hfAlpha, .hfBeta, .hfOmega, .hmPsi:
            return .hiIN
        case .ifSara, .imNicola:
            return .itIT
        case .jfAlpha, .jfGongitsune, .jfNezumi, .jfTebukuro, .jmKumo:
            return .jaJP
        case .pfDora, .pmSanta:
            return .ptBR
        case .zfXiaobei, .zfXiaoni, .zfXiaoxiao, .zfXiaoyi, .zmYunjian, .zmYunxi, .zmYunxia, .zmYunyang:
            return .znCN
        }
    }
}

// MARK: - Language Dialect

/// Language dialects supported by Kokoro TTS
public enum LanguageDialect: String, CaseIterable, Sendable {
    case none = ""
    case enUS = "en-us"
    case enGB = "en-gb"
    case jaJP = "ja"
    case znCN = "yue"
    case frFR = "fr-fr"
    case hiIN = "hi"
    case itIT = "it"
    case esES = "es"
    case ptBR = "pt-br"
}
